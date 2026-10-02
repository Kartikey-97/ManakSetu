"""
kartikey/analysis/test_query_quality.py

Tests for the input-quality gate.

The gate has two ways to be wrong and they are not symmetric. A false accept
costs a few seconds and an API call. A false reject tells a procurement officer
their real requirement is meaningless, and there is no way for them to argue with
it — so the accept cases below are the ones that matter most, and they outnumber
the reject cases deliberately.
"""

from __future__ import annotations

import pytest

from kartikey.analysis.query_quality import (
    REJECTION_MESSAGE,
    assess_query,
)


# ===========================================================================
# Must be rejected
# ===========================================================================

class TestRejected:

    @pytest.mark.parametrize(
        "text",
        [
            "fhbjfbdjvfdjhbdf",
            "dhdjfhfjfjfjfjsjsjshshsjsjs",
            "asdfghjkl",
            "qwertyuiop",
            "zxcvbnmzxcvbnm",
            "aaaaaaaaaaaa",
            "abababababab",
            "kjhgfdsakjhgfdsa",
            "xkcdvbnmqwrtzp",
            "sdfsdfsdfsdfsdf",
        ],
    )
    def test_character_soup_is_rejected(self, text: str) -> None:
        result = assess_query(text)

        assert result.is_meaningful is False
        assert result.message == REJECTION_MESSAGE

    @pytest.mark.parametrize("text", ["", "   ", "\n\t  \n", None])
    def test_empty_input_is_rejected(self, text) -> None:
        result = assess_query(text)

        assert result.is_meaningful is False
        assert result.message == REJECTION_MESSAGE

    @pytest.mark.parametrize("text", ["...", "!!!", "-- // --", "12345"])
    def test_input_with_no_requirement_in_it_is_rejected(self, text: str) -> None:
        """
        Not gibberish, but there is nothing to analyse either. A bare number is
        the interesting case: it could be a quantity, but with no product and no
        designation attached there is no requirement to check it against.
        """
        result = assess_query(text)

        assert result.is_meaningful is False

    def test_a_wall_of_noise_is_not_rescued_by_one_real_word(self) -> None:
        """
        The aggregate rule. A recognisable token inside pure noise is a
        coincidence, and analysing the noise around it produces a confident
        report about nothing.
        """
        result = assess_query(
            "fhbjfbdjvfdjhbdf dhdjfhfjfjfj kjhgfdsaqwe cable zxcvbnmasdfg "
            "poiuytrewqas mnbvcxzlkjhg"
        )

        assert result.is_meaningful is False
        assert result.signals["reason"] == "dominated_by_gibberish"

    @pytest.mark.parametrize("text", ["kjh gfd saq zxv", "hjk lkj mnb vcx"])
    def test_short_chunk_mashing_is_rejected(self, text: str) -> None:
        """
        Mashing broken into three-letter chunks slips under the six-letter floor
        the shape tests need. One of those chunks usually contains a vowel by
        chance, so short tokens only count as evidence when they outnumber the
        malformed ones beside them.
        """
        result = assess_query(text)

        assert result.is_meaningful is False

    def test_the_rejection_explains_itself(self) -> None:
        result = assess_query("fhbjfbdjvfdjhbdf")

        assert result.rejected_tokens == ["fhbjfbdjvfdjhbdf"]
        assert result.signals["reason"] == "no_meaningful_content"


# ===========================================================================
# Must be accepted
# ===========================================================================

class TestAccepted:

    @pytest.mark.parametrize(
        "text",
        [
            "IS 10322",
            "IS 10322 for LED street lighting",
            "LED street lighting procurement requirement",
            "IS10322",
            "IS:10322-2012",
            "IS 10322 : Part 5 : Sec 3",
            "IS/IEC 60034 Part 30",
            "IEC 60598",
        ],
    )
    def test_the_required_valid_inputs_pass(self, text: str) -> None:
        result = assess_query(text)

        assert result.is_meaningful is True
        assert result.message is None

    @pytest.mark.parametrize(
        "text",
        [
            "cable",
            "LED",
            "pump",
            "luminaire",
            "PVC cable",
            "IP66",
            "230V",
            "2.5 sqmm cable",
            "IE3 motor",
            "XLPE",
            "MCB",
            "QCO",
        ],
    )
    def test_short_legitimate_technical_queries_pass(self, text: str) -> None:
        """
        The reason this gate cannot be a length check. Several of these are
        shorter than the strings it must reject, and two are all-consonant.
        """
        result = assess_query(text)

        assert result.is_meaningful is True

    def test_ordinary_tender_prose_passes(self) -> None:
        result = assess_query(
            "The supply and installation of LED street light luminaires shall "
            "conform to IS 10322 (Part 5/Sec 3) : 2012 and the luminaire "
            "efficacy shall not be less than 110 lm/W. Bidders must submit a "
            "valid BIS registration certificate along with the technical bid."
        )

        assert result.is_meaningful is True
        assert result.signals["designation"] is True
        assert result.signals["vocabulary_hits"] > 5

    def test_prose_with_no_designation_still_passes(self) -> None:
        """
        Nothing in the instruction says a query must name a standard. Finding the
        right standard is the system's job, not the user's.
        """
        result = assess_query(
            "Procurement of energy efficient submersible pumps for rural water "
            "supply schemes with three phase motors"
        )

        assert result.is_meaningful is True
        assert result.signals["designation"] is False

    def test_an_unrecognised_but_well_formed_word_passes(self) -> None:
        """
        The rule that keeps the gate honest: an unknown product is not a
        meaningless one. Rejecting these would be rejecting a valid query merely
        because we cannot match it.
        """
        for text in ("borewell", "geotextile", "spandrel", "vermicompost"):
            assert assess_query(text).is_meaningful is True, text

    @pytest.mark.parametrize(
        "text",
        [
            "N95",
            "N95 mask",
            "KN95 respirator",
            "DN300 valve",
            "M20 concrete",
            "Fe500 rebar",
            "Bitumen VG30",
            "ACSR",
            "GI wire",
        ],
    )
    def test_product_and_grade_codes_pass(self, text: str) -> None:
        """
        Procurement runs on alphanumeric codes and they are frequently the whole
        query. Mixing letters with digits is what separates them from a bare
        number, which still says nothing.
        """
        assert assess_query(text).is_meaningful is True

    @pytest.mark.parametrize(
        "text",
        [
            "supply of 100 nos LED street lights 90W",
            "Providing and fixing of vitrified tiles",
            "Distribution transformer 250 kVA 11/0.433 kV",
            "Solar photovoltaic module 540 Wp mono PERC",
            "ductile iron double flanged sluice valve DN300 PN10",
            "Aluminium conductor steel reinforced ACSR Panther",
            "polypropylene woven sacks",
            "geosynthetic clay liner",
            "SITC of 33/11 kV substation",
            "what standards apply to submersible pumps",
        ],
    )
    def test_real_procurement_line_items_pass(self, text: str) -> None:
        """
        Drawn from the shapes Indian tender line items actually take. Several
        contain no standard designation and no word this gate has vocabulary for.
        """
        assert assess_query(text).is_meaningful is True

    def test_a_typo_beside_a_real_term_does_not_sink_the_query(self) -> None:
        result = assess_query("luminaire specfcatn for strreet lghtng")

        assert result.is_meaningful is True

    @pytest.mark.parametrize(
        "text",
        [
            "एलईडी स्ट्रीट लाइट LED street light IS 10322",
            "श्रेणी 1 विद्युत केबल",
            "एलईडी स्ट्रीट लाइट",
        ],
    )
    def test_indic_script_prose_passes(self, text: str) -> None:
        """
        Indian tender documents are routinely bilingual, and the shape tests are
        Latin-only. Without a signal for non-Latin letters a requirement written
        in Hindi is invisible to every test and gets rejected as empty.
        """
        assert assess_query(text).is_meaningful is True


# ===========================================================================
# Properties of the gate itself
# ===========================================================================

class TestGateProperties:

    def test_it_is_deterministic(self) -> None:
        text = "IS 16107 LED luminaire performance requirements"
        first = assess_query(text)
        second = assess_query(text)

        assert (first.is_meaningful, first.signals) == (second.is_meaningful, second.signals)

    def test_it_makes_no_model_or_network_call(self, monkeypatch) -> None:
        """
        The gate's whole purpose is to run before anything expensive. If it ever
        acquires a dependency on the LLM client or an HTTP call, it stops being
        a gate and becomes another cost.
        """
        import httpx

        def _forbidden(*args, **kwargs):
            raise AssertionError("the quality gate performed network I/O")

        monkeypatch.setattr(httpx, "post", _forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "request", _forbidden, raising=False)

        assert assess_query("LED street lighting IS 10322").is_meaningful is True
        assert assess_query("fhbjfbdjvfdjhbdf").is_meaningful is False

    def test_it_does_not_read_a_whole_document_to_decide(self) -> None:
        """A gate that exists to be cheap must not scan a 200-page PDF."""
        text = "IS 10322 luminaire " + ("filler prose about lighting " * 5000)

        result = assess_query(text)

        assert result.is_meaningful is True
        assert result.signals["tokens_examined"] < 1000

    def test_a_long_real_document_is_accepted(self) -> None:
        clause = (
            "The bidder shall supply LED street light luminaires conforming to "
            "IS 16107 with IP66 ingress protection and a minimum efficacy of "
            "110 lm/W. "
        )
        result = assess_query(clause * 40)

        assert result.is_meaningful is True

    def test_signals_are_reported_for_accepted_input_too(self) -> None:
        """
        Not only for rejections. The accepted-path signals are what let the demo
        show why an input was considered analysable.
        """
        result = assess_query("IS 10322 for LED street lighting")

        assert result.signals["designation"] is True
        assert "lighting" in result.signals["sample_terms"]
