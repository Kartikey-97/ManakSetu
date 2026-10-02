"""
kartikey/document_processing/test_is_reference_scan.py

`scan_is_references` must find genuine IS citations and must not read ordinary
prose — "the warranty is 5 years" — as a citation of IS 5. IS 694 and IS 732 are
real, QCO-notified catalogue standards, so a phantom "is 694 mm" is not noise: it
becomes a finding and a high-confidence certification claim.
"""

from __future__ import annotations

import asyncio

import pytest

from kartikey.document_processing.extractor import scan_is_references


def _refs(text: str) -> list[tuple]:
    return [
        (r["matched_text"], r["is_number"], r["part_section"], r["year"], r["amendment_number"])
        for r in scan_is_references(text)
    ]


# ===========================================================================
# Genuine citations — unchanged
# ===========================================================================

@pytest.mark.parametrize("text, expected", [
    ("Steel conforming to IS 2062", [("IS 2062", "IS 2062", None, None, None)]),
    ("Cables as per IS 1554", [("IS 1554", "IS 1554", None, None, None)]),
    ("IS 10322 (Part 5/Sec 3):2012 luminaires",
     [("IS 10322 (Part 5/Sec 3):2012", "IS 10322", "Part 5/Sec 3", 2012, None)]),
    ("conforming to IS 1554 (Part 1) : 1988",
     [("IS 1554 (Part 1) : 1988", "IS 1554", "Part 1", 1988, None)]),
    ("IS 1180 (Part 1):2014", [("IS 1180 (Part 1):2014", "IS 1180", "Part 1", 2014, None)]),
    ("IS 2062:2011 Amd.4", [("IS 2062:2011 Amd.4", "IS 2062", None, 2011, 4)]),
    ("IS 2062:2011 AMD 4", [("IS 2062:2011 AMD 4", "IS 2062", None, 2011, 4)]),
    ("IS 269 (latest edition)", [("IS 269 (latest edition)", "IS 269", "latest edition", None, None)]),
    ("as per IS 694:2010 and IS 8130:2013",
     [("IS 694:2010", "IS 694", None, 2010, None), ("IS 8130:2013", "IS 8130", None, 2013, None)]),
    ("केबल IS 1554 (भाग 1):1988 के अनुरूप होनी चाहिए।",
     [("IS 1554 (भाग 1):1988", "IS 1554", "भाग 1", 1988, None)]),
    ("Pipes IS 1239 ke anusar hone chahiye.", [("IS 1239", "IS 1239", None, None, None)]),
    ("IS 2062 shall apply.", [("IS 2062", "IS 2062", None, None, None)]),
    # ALL-CAPS tenders: a word after the number is still a citation.
    ("STEEL CONFORMING TO IS 2062 GRADE E250", [("IS 2062", "IS 2062", None, None, None)]),
    ("PLATES TO IS 2062 MARKED WITH GRADE", [("IS 2062", "IS 2062", None, None, None)]),
])
def test_genuine_citations(text, expected) -> None:
    assert _refs(text) == expected


def test_char_offset_is_the_citation_start() -> None:
    text = "Cables shall conform to IS 694:2010."
    assert scan_is_references(text)[0]["char_offset"] == text.index("IS 694")


# ===========================================================================
# Prose — no citation
# ===========================================================================

@pytest.mark.parametrize("text", [
    "The rated voltage is 230 V.",
    "The warranty is 5 years.",
    "Price basis 230 per unit.",
    "this 5 year contract",
    "Analysis 2 shall follow.",
    "The overall cabinet height is 694 mm.",
    "The cable length is 732 m.",
    "WARRANTY IS 5 YEARS",
    "THE RATED VOLTAGE IS 230 V.",
    "OVERALL HEIGHT IS 694 MM",
    "CABLE LENGTH IS 732 M",
    "THIS 5 YEAR CONTRACT",
    "RATED CURRENT IS 16 A",
    "AMBIENT TEMPERATURE IS 40 °C",
    "EFFICIENCY IS 90 %",
    "QUANTITY IS 50 NOS",
    # punctuation and spacing around prose
    "The voltage is 230V.",
    "(warranty is 5 years)",
    "Warranty: is 5 years; voltage is  230 V",
    "Is 5 years enough?",
])
def test_prose_is_not_a_citation(text) -> None:
    assert scan_is_references(text) == []


def test_rejected_quantity_does_not_backtrack_to_a_shorter_number() -> None:
    """Without (?!\\d), "IS 230 V" would fail at 230 and then match "IS 23"."""
    assert scan_is_references("VOLTAGE IS 230 V AND HEIGHT IS 694 MM") == []


# ===========================================================================
# Separator and year formats
# ===========================================================================

@pytest.mark.parametrize("text, expected", [
    ("Steel conforming to IS:2062", [("IS:2062", "IS 2062", None, None, None)]),
    ("Steel conforming to IS-2062", [("IS-2062", "IS 2062", None, None, None)]),
    ("Steel conforming to IS2062", [("IS2062", "IS 2062", None, None, None)]),
    ("Steel shall conform to IS : 2062 - 2011.", [("IS : 2062 - 2011", "IS 2062", None, 2011, None)]),
    ("Steel shall conform to IS 2062-2011.", [("IS 2062-2011", "IS 2062", None, 2011, None)]),
    ("Steel shall conform to IS 2062 – 2011.", [("IS 2062 – 2011", "IS 2062", None, 2011, None)]),
    ("Steel shall conform to IS:2062:2011.", [("IS:2062:2011", "IS 2062", None, 2011, None)]),
])
def test_separator_and_year_formats(text, expected) -> None:
    assert _refs(text) == expected


def test_hyphenated_part_number_is_not_a_year() -> None:
    assert _refs("Pipes to IS 1239-1 shall be used.") == [("IS 1239", "IS 1239", None, None, None)]


@pytest.mark.parametrize("text", [
    "WARRANTY IS: 5 YEARS",
    "VOLTAGE IS - 230 V",
    "VOLTAGE IS-230V",
    "LENGTH IS:732 M",
    "THIS2062",
    "Analysis-2 shall follow.",
])
def test_separators_do_not_reopen_prose_false_positives(text) -> None:
    assert scan_is_references(text) == []


def test_spaced_colon_and_hyphen_citation_becomes_a_requirement() -> None:
    """Previously produced no requirement at all; is_reference stays the base number."""
    from kartikey.analysis.requirement_extractor import extract_requirements

    requirements, _ = extract_requirements(
        "a1", "Supply of structural steel sections. Steel shall conform to IS : 2062 - 2011.",
    )
    cited = [(r.is_reference, r.cited_year) for r in requirements if r.is_reference]

    assert cited == [("IS 2062", 2011)]


# ===========================================================================
# Part / section designation
# ===========================================================================

def _parts(text: str) -> list[tuple]:
    return [
        (r["is_number"], r["part_section"], r["year"], r["designation"], r["citation"])
        for r in scan_is_references(text)
    ]


@pytest.mark.parametrize("text, expected", [
    ("as per IS 10322 (Part 5/Sec 3)",
     [("IS 10322", "Part 5/Sec 3", None, "IS 10322 : Part 5 : Sec 3", "IS 10322 (Part 5/Sec 3)")]),
    ("as per IS 10322 (Part 5 : Sec 3)",
     [("IS 10322", "Part 5 : Sec 3", None, "IS 10322 : Part 5 : Sec 3", "IS 10322 (Part 5 : Sec 3)")]),
    ("as per IS 10322 (Part 5/Sec 3) : 2012",
     [("IS 10322", "Part 5/Sec 3", 2012, "IS 10322 : Part 5 : Sec 3", "IS 10322 (Part 5/Sec 3) : 2012")]),
    ("conforming to IS 1554 : Part 1 : 1988",
     [("IS 1554", "Part 1", 1988, "IS 1554 : Part 1", "IS 1554 : Part 1 : 1988")]),
    ("conforming to IS 1554 (Part 1) : 1988",
     [("IS 1554", "Part 1", 1988, "IS 1554 : Part 1", "IS 1554 (Part 1) : 1988")]),
    ("conforming to IS 1554 : Part 1",
     [("IS 1554", "Part 1", None, "IS 1554 : Part 1", "IS 1554 : Part 1")]),
    ("conforming to IS 1554 : Part 2 : 2010",
     [("IS 1554", "Part 2", 2010, "IS 1554 : Part 2", "IS 1554 : Part 2 : 2010")]),
    ("luminaires to IS 10322 : Part 5 : Sec 3 : 2012",
     [("IS 10322", "Part 5 : Sec 3", 2012, "IS 10322 : Part 5 : Sec 3", "IS 10322 : Part 5 : Sec 3 : 2012")]),
    ("CABLES TO IS 1554 : PART 1 : 1988",
     [("IS 1554", "PART 1", 1988, "IS 1554 : Part 1", "IS 1554 : PART 1 : 1988")]),
    ("luminaires to IS 10322 (Part 5/Section 3)",
     [("IS 10322", "Part 5/Section 3", None, "IS 10322 : Part 5 : Sec 3", "IS 10322 (Part 5/Section 3)")]),
    # Not translated: the Hindi part stays base-only.
    ("केबल IS 1554 (भाग 1) : 1988",
     [("IS 1554", "भाग 1", 1988, "IS 1554", "IS 1554 (भाग 1) : 1988")]),
    ("IS 269 (latest edition)",
     [("IS 269", "latest edition", None, "IS 269", "IS 269 (latest edition)")]),
    # A year after a colon is not mistaken for a part, and a dangling "Part"
    # leaves the citation at its base.
    ("as per IS 732 : 2019", [("IS 732", None, 2019, "IS 732", "IS 732 : 2019")]),
    ("as per IS 1554 : Part", [("IS 1554", None, None, "IS 1554", "IS 1554")]),
    ("to IS 1554 :\n Part 1 : 1988", [("IS 1554", "Part 1", 1988, "IS 1554 : Part 1", "IS 1554 : Part 1 : 1988")]),
    # Ranges are not narrowed to their first member.
    ("as per IS 9000 : Part 2 : Sec 1 to 4", [("IS 9000", "Part 2", None, "IS 9000 : Part 2", "IS 9000 : Part 2")]),
    ("as per IS 1944 : Part 1 and 2", [("IS 1944", None, None, "IS 1944", "IS 1944")]),
    ("as per IS 10322 (Part 5/Sec 1 to 3)",
     [("IS 10322", "Part 5/Sec 1 to 3", None, "IS 10322", "IS 10322 (Part 5/Sec 1 to 3)")]),
])
def test_part_and_section_designation(text, expected) -> None:
    assert _parts(text) == expected


@pytest.mark.parametrize("text, designation", [
    ("IS 2062", "IS 2062"), ("IS:2062", "IS 2062"), ("IS-2062", "IS 2062"),
    ("IS2062", "IS 2062"), ("IS : 2062 - 2011", "IS 2062"), ("IS 2062-2011", "IS 2062"),
])
def test_unqualified_citations_keep_the_base_designation(text, designation) -> None:
    assert [r["designation"] for r in scan_is_references(text)] == [designation]


@pytest.mark.parametrize("value, base", [
    ("IS 10322 : Part 5 : Sec 3", "IS 10322"),
    ("IS 1554 : Part 1", "IS 1554"),
    ("IS 2062", "IS 2062"),
    ("IS/IEC 60947", "IS/IEC 60947"),
])
def test_base_is_number(value, base) -> None:
    from kartikey.document_processing.extractor import base_is_number
    assert base_is_number(value) == base


# ===========================================================================
# Every caller sees the same result
# ===========================================================================

STEEL_CABINET = (
    "Supply of steel storage cabinets for the district office. "
    "The overall cabinet height is 694 mm and the rated voltage is 230 V. "
    "Warranty is 5 years. Shelves shall be of CRCA sheet conforming to IS 513."
)


def test_requirement_extraction_has_no_phantom_citations() -> None:
    from kartikey.analysis.requirement_extractor import extract_requirements

    requirements, _ = extract_requirements("a1", STEEL_CABINET)

    assert {r.is_reference for r in requirements if r.is_reference} == {"IS 513"}


def test_profile_has_no_phantom_regulatory_mentions() -> None:
    from kartikey.analysis.profile_extractor import extract_profile

    profile = extract_profile(STEEL_CABINET)
    labels = [f.label for f in profile.regulatory_mentions]

    assert not any(label.lower().startswith("is ") and label != "IS 513" for label in labels), labels


def test_pipeline_steel_cabinet_has_no_phantom_is_694(monkeypatch) -> None:
    from shared.models import Analysis, AnalysisStatus, InputType
    from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry
    from kartikey.orchestration.pipeline import run_analysis_pipeline

    registry = initialize_knowledge_registry()
    assert registry.standards_store.get_by_is_number("IS 694"), "IS 694 must be a real record"
    monkeypatch.setattr("shared.config.settings.enable_bis_sync", False)

    analysis = Analysis(
        input_type=InputType.TEXT, raw_text=STEEL_CABINET, status=AnalysisStatus.QUEUED,
    )
    asyncio.run(run_analysis_pipeline(analysis.id, {analysis.id: analysis}))

    assert analysis.status.value == "completed"
    assert "IS 694" not in {r.is_reference for r in analysis.requirements}
    assert not any(
        s.is_number == "IS 694"
        for f in analysis.findings
        for s in f.applicable_standards + f.cited_standards
    )
    assert "IS 694" not in [q["is_number"] for q in (analysis.qco_findings or [])]
    # The genuine citation in the same text is still found.
    assert "IS 513" in {r.is_reference for r in analysis.requirements}
