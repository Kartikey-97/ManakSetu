"""
kartikey/analysis/test_profile_extractor.py

Tests for the deterministic procurement-profile extractor.

This module reads the "Understanding Procurement Requirements" screen — the
first thing an evaluator sees after uploading a tender. It replaced a single
blocking Gemini call that, on a quota error, returned two invented requirements
and the product name "Unknown Entity". So there are two properties worth
pinning, and they pull in opposite directions:

  1. It has to actually find things. A profile screen with four empty buckets
     is indistinguishable from a broken pipeline, and the whole point of the
     rewrite was that the deterministic path populates the screen on its own.

  2. It must never invent. Every value it prints is either a quote from the
     document or the literal `NOT_STATED`, and every location it cites is a
     number the document itself used. A fabricated clause reference is worse
     than a blank field, because the audit trail is the product.

The tests below are grouped along that split, plus the merge boundary where
model output is allowed in — under the rule that it may only fill gaps, never
overwrite what the document said.
"""

from __future__ import annotations

import pytest

from kartikey.analysis.profile_extractor import (
    NOT_STATED,
    LLM_REFINABLE_FIELDS,
    extract_profile,
    merge_llm_context,
)
from kartikey.document_processing.extractor import clause_at, iter_clauses

# The bundled LED tender, near enough. This is the document the demo runs on, so
# it is the one the assertions are written against — a synthetic sample tuned to
# the rules would prove only that the rules match themselves.
LED_TENDER = """Name of Work: Supply, installation, testing and commissioning of
90W LED street light luminaires for municipal arterial road lighting.

2.1 The luminaires shall conform to IS 10322 (Part 5/Sec 3) : 2012.
2.2 Efficacy shall not be less than 110 lm/W at rated input.
2.3 Ingress protection shall be IP66 and impact protection IK08.
2.4 The luminaire shall operate from -10 C to +50 C at relative humidity up to 95 %.

3.1 Bidders must submit a valid BIS registration certificate under the CRS scheme.
3.2 Test Certificate from an NABL accredited laboratory shall be furnished.

4.1 Surge protection of 10 kV shall be provided.
4.2 Warranty of 5 years from the date of commissioning.
"""

CABLE_TENDER = """Subject: Procurement of 1.1 kV grade XLPE insulated armoured
power cables for underground distribution.

1. Conductor shall be 3.5 core 185 sqmm aluminium conforming to IS 8130.
2. Insulation shall conform to IS 1554 : Part 1.
3. Test certificates from an NABL accredited laboratory shall be furnished.
"""


@pytest.fixture(scope="module")
def led():
    return extract_profile(LED_TENDER, fallback_category="")


# ===========================================================================
# It has to find things
# ===========================================================================

class TestTheScreenIsPopulated:
    """
    A demo-readiness gate. Each assertion names the panel it protects, because
    the failure mode is not an exception — it is a blank box on stage.
    """

    def test_the_prose_header_is_filled_in(self, led) -> None:
        for name in ("product", "category", "application", "environment"):
            value = getattr(led, name)
            assert value != NOT_STATED, f"header field '{name}' came back blank"
            assert value.strip(), f"header field '{name}' is whitespace"

    def test_every_requirement_bucket_has_content(self, led) -> None:
        buckets = {
            "technicalParameters": led.technical_parameters,
            "performanceRequirements": led.performance_requirements,
            "testingRequirements": led.testing_requirements,
            "regulatoryMentions": led.regulatory_mentions,
        }
        empty = [name for name, rows in buckets.items() if not rows]
        assert not empty, f"empty on the demo document: {empty}"

    def test_the_product_is_the_work_not_the_whole_clause(self, led) -> None:
        """
        "Name of Work: Supply, installation ... luminaires for municipal ..." —
        the product is the part before "for", and it has to be short enough to
        sit on one line of the header.
        """
        assert "LED street light luminaire" in led.product
        assert not led.product.lower().startswith("name of work")
        assert len(led.product) <= 90

    def test_the_ip_rating_reads_as_written(self, led) -> None:
        """
        Regression: the environment rule captured only the digits of "IP66" and
        printed "ingress protection 66", which is not a rating anyone can look
        up. Self-describing designations carry their own label.
        """
        assert "IP66" in led.environment
        assert "protection 66" not in led.environment

    def test_a_clause_stating_only_a_class_rating_is_kept(self, led) -> None:
        """
        Regression: clause 2.3 states IP66 and IK08 and no numeric quantity with
        a unit, so a figure test that only understood "number + unit" dropped
        it. It is the clause a lighting tender is most often judged on.
        """
        rows = [
            f for bucket in (led.technical_parameters, led.performance_requirements)
            for f in bucket
        ]
        assert any("IP66" in f.value or "IK08" in f.value for f in rows), (
            "the ingress/impact protection clause reached no bucket: "
            f"{[f.value for f in rows]}"
        )

    def test_the_cited_standard_is_reported_as_regulatory(self) -> None:
        profile = extract_profile(LED_TENDER)
        regulatory = " | ".join(f.value for f in profile.regulatory_mentions)
        assert "IS 10322" in regulatory
        assert "BIS" in regulatory.upper()

    def test_a_second_real_document_also_populates(self) -> None:
        """The rules are not tuned to one document."""
        profile = extract_profile(CABLE_TENDER)
        assert profile.product != NOT_STATED
        assert "cable" in profile.product.lower()
        assert profile.total_fields >= 3
        assert any("IS 8130" in f.value or "IS 1554" in f.value
                   for f in profile.regulatory_mentions)


# ===========================================================================
# It must never invent
# ===========================================================================

class TestNothingIsFabricated:

    def test_an_empty_document_states_nothing_rather_than_guessing(self) -> None:
        profile = extract_profile("")
        assert profile.product == NOT_STATED
        assert profile.application == NOT_STATED
        assert profile.environment == NOT_STATED
        assert profile.total_fields == 0

    def test_prose_with_no_specifications_yields_no_requirement_rows(self) -> None:
        """
        The old endpoint returned "Equipment must be IP66 rated for outdoor use"
        for input that said no such thing. Text with no figures and no citations
        has to produce an empty list, not a plausible one.
        """
        profile = extract_profile(
            "This letter confirms receipt of your correspondence dated last "
            "Tuesday. The committee will meet in due course and will revert."
        )
        assert profile.total_fields == 0

    def test_every_value_is_a_substring_of_the_document(self, led) -> None:
        """
        The strongest form of "did not invent it": each row's quoted text has to
        appear in the source, modulo whitespace collapsing.
        """
        flat = " ".join(LED_TENDER.split())
        for bucket in (
            led.technical_parameters, led.performance_requirements,
            led.testing_requirements, led.regulatory_mentions,
        ):
            for f in bucket:
                assert f.value in flat, (
                    f"quoted text not found in the document: {f.value!r}"
                )

    def test_locations_use_the_documents_own_numbering(self, led) -> None:
        """
        "Clause 2.3" is a citation; "Section 3.4" on a document with no section
        3.4 is a fabrication. Anything the extractor could not anchor has to
        degrade to a line number, which is always true of the text it was given.
        """
        for bucket in (
            led.technical_parameters, led.performance_requirements,
            led.testing_requirements, led.regulatory_mentions,
        ):
            for f in bucket:
                location = f.source_clause
                assert location, f"row {f.value!r} cites no location"
                assert f.as_dict()["sourceClause"] == location
                if location.startswith("Line "):
                    continue
                number = location.split()[-1]
                assert number in LED_TENDER, (
                    f"location {location!r} names a clause the document "
                    "does not contain"
                )

    def test_no_row_is_duplicated_across_buckets(self, led) -> None:
        values = [
            f.value
            for bucket in (
                led.technical_parameters, led.performance_requirements,
                led.testing_requirements, led.regulatory_mentions,
            )
            for f in bucket
        ]
        assert len(values) == len(set(values)), "the same clause was filed twice"

    def test_the_dict_the_endpoint_returns_has_every_key(self, led) -> None:
        payload = led.as_dict()
        assert set(payload) == {
            "product", "category", "application", "environment",
            "technicalParameters", "performanceRequirements",
            "testingRequirements", "regulatoryMentions",
        }


# ===========================================================================
# The boundary where model output is allowed in
# ===========================================================================

class TestMergingLLMContext:

    def test_it_only_fills_fields_the_document_did_not_state(self, led) -> None:
        stated = {name: getattr(led, name) for name in LLM_REFINABLE_FIELDS
                  if getattr(led, name) != NOT_STATED}
        assert stated, "nothing was stated, so this test proves nothing"

        profile = extract_profile(LED_TENDER)
        profile, refined = merge_llm_context(
            profile, {name: "MODEL OVERWROTE THIS" for name in LLM_REFINABLE_FIELDS},
        )
        for name, value in stated.items():
            assert getattr(profile, name) == value, (
                f"the model overwrote '{name}', which the document had stated"
            )
            assert name not in refined

    def test_a_gap_is_filled_and_declared(self) -> None:
        profile = extract_profile("2.1 Shall comply with IS 456.")
        assert profile.product == NOT_STATED

        profile, refined = merge_llm_context(profile, {"product": "Ready mix concrete"})
        assert profile.product == "Ready mix concrete"
        assert refined == ["product"], (
            "a field filled by the model has to be named in the response, or "
            "the screen presents model output and document quotes alike"
        )

    @pytest.mark.parametrize(
        "answer", ["Unknown", "unknown entity", "N/A", "not specified", "none", "  "],
    )
    def test_a_models_way_of_saying_nothing_is_not_information(self, answer) -> None:
        """This is the exact string the old endpoint printed as a product name."""
        profile = extract_profile("2.1 Shall comply with IS 456.")
        profile, refined = merge_llm_context(profile, {"product": answer})
        assert profile.product == NOT_STATED
        assert refined == []

    @pytest.mark.parametrize("context", [None, {}, "not a dict", 42, []])
    def test_a_missing_or_malformed_response_changes_nothing(self, context) -> None:
        """
        Every LLM failure mode reaches this function as some flavour of nothing;
        none of them may disturb the profile that is already built.
        """
        before = extract_profile(LED_TENDER)
        after, refined = merge_llm_context(extract_profile(LED_TENDER), context)
        assert after.as_dict() == before.as_dict()
        assert refined == []

    def test_requirement_rows_are_not_refinable(self) -> None:
        """
        The buckets quote the document. A model rewriting a quote breaks the one
        guarantee the screen makes, so they are outside the refinable set by
        construction rather than by convention.
        """
        assert set(LLM_REFINABLE_FIELDS) == {"product", "application", "environment"}


# ===========================================================================
# iter_clauses — the splitter both the profile screen and the analysis use
# ===========================================================================

class TestIterClauses:
    """
    `iter_clauses` and `clause_at` share `_clause_boundaries` on purpose. If they
    drifted apart, the profile screen and the analysis would disagree about
    where a requirement begins on documents where they read identical text.
    """

    def test_it_agrees_with_clause_at_on_every_clause(self) -> None:
        for offset, clause in iter_clauses(LED_TENDER):
            assert clause_at(LED_TENDER, offset, len(clause)) == clause, (
                f"the two splitters disagree at offset {offset}"
            )

    def test_offsets_point_into_the_original_text(self) -> None:
        for offset, clause in iter_clauses(LED_TENDER):
            assert 0 <= offset < len(LED_TENDER)
            head = clause.split()[0]
            assert LED_TENDER[offset:offset + len(head) + 4].lstrip().startswith(head)

    def test_numbered_clauses_are_split_apart(self) -> None:
        clauses = [c for _, c in iter_clauses(LED_TENDER)]
        assert any(c.startswith("2.2") for c in clauses)
        assert any(c.startswith("2.3") for c in clauses)
        assert not any(c.startswith("2.2") and "2.3" in c for c in clauses), (
            "two numbered clauses were returned as one"
        )

    def test_a_decimal_quantity_does_not_end_a_clause(self) -> None:
        """
        The period in "1.1 kV" is not a full stop. Splitting there is how a
        specification loses the value it specifies.
        """
        text = "The cable shall be 1.1 kV grade with 3.5 core conductors."
        assert [c for _, c in iter_clauses(text)] == [text]

    def test_an_abbreviation_does_not_end_a_clause(self) -> None:
        text = "Conform to IS 2062 Amd. 4 for the plate material."
        assert [c for _, c in iter_clauses(text)] == [text]

    def test_empty_text_yields_no_clauses(self) -> None:
        assert iter_clauses("") == []
        assert iter_clauses("   \n\n  ") == []

    def test_no_clause_is_blank_or_untrimmed(self) -> None:
        for _, clause in iter_clauses(LED_TENDER):
            assert clause == clause.strip()
            assert clause
            assert "  " not in clause, "whitespace was not collapsed"
