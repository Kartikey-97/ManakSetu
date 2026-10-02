"""
kartikey/analysis/test_scope_check.py

Tests for the deterministic scope/material check.

The gap this closes: a tender specifying "1.1 kV grade XLPE insulated armoured
power cables ... conforming to IS 1554 : Part 1" cites a standard that covers
*PVC* insulation. Nothing in the pipeline could see it. The year checks pass
(there is no year), the status checks pass (IS 1554 is active), the QCO check
passes, and the AI verdict for a resolved citation is JUSTIFIED. The defect is
not about the edition; it is about what the standard covers.

Two properties matter more than any single case here, and most of the file is
about them:

  1. It must not fire on a correct citation. A false mismatch tells a
     procurement officer to change a clause that was right, which is worse than
     silence because the report looks like it checked. So the tests below cover
     the near-misses — a material the catalogue has never heard of, a standard
     whose title states no material, a standard whose scope prose mentions the
     material even though its title leads with another.

  2. It must never claim to have checked something it did not. `checked=False`
     and `mismatch=False` mean completely different things, and a report that
     conflates them turns "we could not assess this" into "this is fine".

The vocabulary is not typed in. It is mined from the titles in whatever
catalogue is loaded, so these tests build small stores and assert against what
those stores imply — remove the XLPE standard and the module correctly stops
believing XLPE is a material, which is the behaviour asserted in
`TestVocabularyComesFromTheCatalogue`.
"""

from __future__ import annotations

import pytest

from shared.models import Requirement, Standard, StandardStatus

from kartikey.analysis import scope_check as S
from kartikey.analysis.scope_check import check_scope


# ===========================================================================
# Fixtures
# ===========================================================================

# The real catalogue records this scenario turns on, quoted as BIS writes them.
PVC_1554 = (
    "PVC Insulated (Heavy Duty) Electric Cables - Part 1: For Working Voltages "
    "up to and Including 1100 V"
)
XLPE_7098 = (
    "Cross Linked Polyethylene (XLPE) Insulated Thermoplastic Sheathed Cables - "
    "Part 1: For Working Voltages from 3.3 kV up to and Including 11 kV"
)
PVC_694 = "PVC Insulated Cables for Working Voltages up to and Including 1100 V"


def _std(
    is_number: str,
    title: str,
    year: int | None = 1988,
    status: StandardStatus = StandardStatus.ACTIVE,
    **kwargs,
) -> Standard:
    return Standard(
        is_number=is_number, title=title, year=year, status=status, **kwargs,
    )


def _req(text: str) -> Requirement:
    return Requirement(analysis_id="a1", text=text)


CABLE_TENDER = (
    "Supply of 1.1 kV grade XLPE insulated armoured power cables, 3.5 core "
    "185 sqmm aluminium conductor, conforming to IS 1554 : Part 1."
)


@pytest.fixture
def catalogue(monkeypatch):
    """
    Install a small catalogue as the vocabulary source.

    Patches `_vocabulary` rather than the knowledge registry: the registry is a
    process-wide singleton that other test modules share, and rebuilding it here
    would leak into them. The seam is the same one production uses — the mining
    itself is exercised, only the record set is controlled.
    """
    def _install(standards: list[Standard]):
        vocabulary = S._build_vocabulary(standards)
        monkeypatch.setattr(S, "_vocabulary", lambda: vocabulary)
        return vocabulary
    return _install


@pytest.fixture
def cable_catalogue(catalogue):
    """The three cable standards the XLPE-vs-PVC scenario turns on."""
    catalogue([
        _std("IS 1554 : Part 1", PVC_1554),
        _std("IS 7098 : Part 1", XLPE_7098),
        _std("IS 694", PVC_694, year=2010),
    ])


# ===========================================================================
# The scenario the module exists for
# ===========================================================================

class TestTheWrongStandardIsCaught:

    def test_xlpe_requirement_against_a_pvc_standard_is_a_mismatch(
        self, cable_catalogue,
    ) -> None:
        result = check_scope(
            _req(CABLE_TENDER), _std("IS 1554 : Part 1", PVC_1554),
        )
        assert result.checked
        assert result.mismatch
        assert result.attribute == "insulated"

    def test_the_mismatch_names_both_materials(self, cable_catalogue) -> None:
        """
        A finding that says "wrong standard" without saying wrong *how* is not
        actionable. Both sides have to be in the payload.
        """
        result = check_scope(
            _req(CABLE_TENDER), _std("IS 1554 : Part 1", PVC_1554),
        )
        assert "XLPE" in (result.required_material or "")
        assert "PVC" in (result.covered_material or "")
        assert "XLPE" in result.note
        assert "PVC" in result.note

    def test_the_mismatch_points_at_the_standard_that_does_cover_it(
        self, cable_catalogue,
    ) -> None:
        result = check_scope(
            _req(CABLE_TENDER), _std("IS 1554 : Part 1", PVC_1554),
        )
        assert result.alternatives == ["IS 7098 : Part 1:1988"]

    def test_the_full_material_name_is_recognised_as_well_as_the_acronym(
        self, cable_catalogue,
    ) -> None:
        """
        BIS writes "Cross Linked Polyethylene (XLPE)"; a tender may use either
        form, and they name the same material.
        """
        result = check_scope(
            _req("Cross Linked Polyethylene insulated cables to IS 1554 : Part 1."),
            _std("IS 1554 : Part 1", PVC_1554),
        )
        assert result.mismatch
        assert result.required_as_written == "cross linked polyethylene"

    def test_a_second_pvc_standard_is_also_flagged(self, cable_catalogue) -> None:
        """The check is about the material, not about one particular IS number."""
        result = check_scope(_req(CABLE_TENDER), _std("IS 694", PVC_694, year=2010))
        assert result.mismatch


# ===========================================================================
# What must NOT fire
# ===========================================================================

class TestCorrectCitationsAreLeftAlone:
    """
    Every case here would be a false accusation. They matter more than the
    positive case: a missed mismatch costs a finding, an invented one sends an
    officer to change a clause that was correct.
    """

    def test_the_matching_standard_is_not_a_mismatch(self, cable_catalogue) -> None:
        result = check_scope(_req(CABLE_TENDER), _std("IS 7098 : Part 1", XLPE_7098))
        assert result.checked
        assert not result.mismatch

    def test_pvc_requirement_against_the_pvc_standard(self, cable_catalogue) -> None:
        result = check_scope(
            _req("PVC insulated heavy duty cables 1100 V to IS 1554 : Part 1."),
            _std("IS 1554 : Part 1", PVC_1554),
        )
        assert result.checked
        assert not result.mismatch

    def test_a_material_the_catalogue_never_heard_of_is_not_a_mismatch(
        self, cable_catalogue,
    ) -> None:
        """
        "Teflon" precedes "insulated" and is not in the catalogue. That is not
        evidence the citation is wrong — it may be a trade name, a typo, or a
        material BIS has no standard for. Guessing here would invent findings
        from any unfamiliar word.
        """
        result = check_scope(
            _req("Teflon insulated wiring conforming to IS 1554 : Part 1."),
            _std("IS 1554 : Part 1", PVC_1554),
        )
        assert not result.checked
        assert not result.mismatch

    def test_a_requirement_with_no_material_qualifier_is_not_assessed(
        self, cable_catalogue,
    ) -> None:
        result = check_scope(
            _req("Underground laying of power cables as per IS 1554 : Part 1."),
            _std("IS 1554 : Part 1", PVC_1554),
        )
        assert not result.checked

    def test_a_standard_stating_no_material_is_not_assessed(
        self, cable_catalogue,
    ) -> None:
        """
        Most catalogue titles name no material at all. Comparing XLPE against
        nothing is not a mismatch — and it is not a match either, which is the
        bug this pins: the fall-through used to report agreement.
        """
        result = check_scope(
            _req(CABLE_TENDER),
            _std("IS 10322 : Part 1", "Luminaires - Part 1: General Requirements"),
        )
        assert not result.checked
        assert not result.mismatch
        assert "does not state one" in result.note

    def test_scope_prose_mentioning_the_material_prevents_a_mismatch(
        self, cable_catalogue,
    ) -> None:
        """
        A title is a summary. If the standard's own scope says it covers XLPE,
        the title leading with PVC is not grounds to call the citation wrong.
        """
        result = check_scope(
            _req(CABLE_TENDER),
            _std(
                "IS 1554 : Part 1", PVC_1554,
                scope=(
                    "This standard covers PVC and XLPE insulated cables for "
                    "working voltages up to 1100 V."
                ),
            ),
        )
        assert not result.mismatch

    def test_the_attributes_are_kept_separate(self, cable_catalogue) -> None:
        """
        Insulation and sheathing are different slots. A requirement naming a
        sheath material must not be compared against an insulation material —
        "PVC sheathed, XLPE insulated" is an ordinary cable, not a defect.
        """
        result = check_scope(
            _req("XLPE insulated, PVC sheathed cables to IS 7098 : Part 1."),
            _std("IS 7098 : Part 1", XLPE_7098),
        )
        assert not result.mismatch


# ===========================================================================
# Several materials for one attribute
# ===========================================================================

MIXED_CLAUSE = "PVC insulation to IS 694 and XLPE insulated cables to IS 7098 : Part 1."


class TestSeveralMaterialsForOneAttribute:
    """
    One clause can specify two products: "PVC insulation to IS 694 and XLPE
    insulated cables to IS 7098". Which material belongs to which standard is
    not something the text states, so neither standard may be flagged for the
    other's material — and neither may be reported as matching either.
    """

    @pytest.mark.parametrize("is_number, title", [
        ("IS 694", PVC_694),
        ("IS 7098 : Part 1", XLPE_7098),
    ])
    def test_neither_standard_is_flagged_or_matched(
        self, cable_catalogue, is_number, title,
    ) -> None:
        result = check_scope(_req(MIXED_CLAUSE), _std(is_number, title))

        assert result.checked is False
        assert result.mismatch is False

    def test_the_note_says_why_it_was_not_assessed(self, cable_catalogue) -> None:
        std = _std("IS 694", PVC_694, year=2010)
        note = check_scope(_req(MIXED_CLAUSE), std).note

        assert "more than one" in note
        assert "PVC" in note and "XLPE" in note
        assert std.designation in note
        assert "not assessed" in note

    def test_a_repeated_single_material_is_still_assessed(self, cable_catalogue) -> None:
        result = check_scope(
            _req("PVC insulated cables with PVC insulation to IS 694."),
            _std("IS 694", PVC_694),
        )
        assert result.checked is True
        assert result.mismatch is False

    def test_a_single_material_is_still_assessed_normally(self, cable_catalogue) -> None:
        match = check_scope(_req("PVC insulated cables to IS 694."), _std("IS 694", PVC_694))
        mismatch = check_scope(_req(CABLE_TENDER), _std("IS 1554 : Part 1", PVC_1554))

        assert (match.checked, match.mismatch) == (True, False)
        assert (mismatch.checked, mismatch.mismatch) == (True, True)
        assert mismatch.required_material and "XLPE" in mismatch.required_material.upper()

    def test_the_mixed_clause_does_not_produce_wrong_scope(self, cable_catalogue) -> None:
        from shared.models import Verdict
        from kartikey.analysis.compliance import run_compliance_checks

        for is_number, title in (("IS 694", PVC_694), ("IS 7098 : Part 1", XLPE_7098)):
            result = run_compliance_checks(
                Requirement(analysis_id="a1", text=MIXED_CLAUSE, is_reference=is_number),
                _std(is_number, title),
            )
            assert result.scope_check.mismatch is False
            assert result.suggested_verdict is not Verdict.WRONG_SCOPE


# ===========================================================================
# The vocabulary is the catalogue's, not the module's
# ===========================================================================

class TestVocabularyComesFromTheCatalogue:

    def test_removing_the_xlpe_standard_stops_the_claim(self, catalogue) -> None:
        """
        With no XLPE standard in the catalogue, "XLPE" is just a word before
        "insulated" and the module has no basis to call anything wrong. That is
        the intended behaviour, not a regression: the module's authority comes
        entirely from what BIS published.
        """
        catalogue([_std("IS 1554 : Part 1", PVC_1554)])
        result = check_scope(
            _req(CABLE_TENDER), _std("IS 1554 : Part 1", PVC_1554),
        )
        assert not result.checked

    def test_no_catalogue_means_no_scope_claim(self, monkeypatch) -> None:
        monkeypatch.setattr(S, "_vocabulary", lambda: None)
        result = check_scope(
            _req(CABLE_TENDER), _std("IS 1554 : Part 1", PVC_1554),
        )
        assert not result.checked
        assert "catalogue was not loaded" in result.note

    def test_a_withdrawn_standard_is_never_offered_as_the_alternative(
        self, catalogue,
    ) -> None:
        """
        Its title still teaches the module that XLPE exists, so the mismatch is
        still found — but telling an officer to cite a withdrawn standard would
        replace one defect with a worse one.
        """
        catalogue([
            _std("IS 1554 : Part 1", PVC_1554),
            _std("IS 7098 : Part 1", XLPE_7098, status=StandardStatus.WITHDRAWN),
        ])
        result = check_scope(
            _req(CABLE_TENDER), _std("IS 1554 : Part 1", PVC_1554),
        )
        assert result.mismatch, "the withdrawn record should still supply the alias"
        assert result.alternatives == []
        assert "standardsbis.gov.in" in result.note

    def test_the_cited_standard_is_not_offered_as_its_own_alternative(
        self, catalogue,
    ) -> None:
        catalogue([
            _std("IS 7098 : Part 1", XLPE_7098),
            _std("IS 7098 : Part 2", XLPE_7098.replace("Part 1", "Part 2"), year=1985),
        ])
        result = check_scope(
            _req("PVC insulated cables to IS 7098 : Part 1."),
            _std("IS 7098 : Part 1", XLPE_7098),
        )
        # PVC is unknown to this catalogue, so nothing is claimed at all.
        assert not result.checked


# ===========================================================================
# Title mining
# ===========================================================================

class TestTitleMining:
    """
    The three defects that made the first implementation return "not checked"
    for every XLPE case, each pinned so it cannot come back.
    """

    def test_a_parenthesised_acronym_survives_mining(self) -> None:
        """
        Treating ")" as a phrase break left nothing between "(XLPE)" and
        "Insulated", so the most important material in this catalogue was never
        mined at all.
        """
        materials = dict(S._materials_in(XLPE_7098))
        assert materials["insulated"] == "Cross Linked Polyethylene (XLPE)"

    def test_a_material_name_does_not_absorb_the_previous_attribute(self) -> None:
        """
        "... (XLPE) Insulated Thermoplastic Sheathed" states two separate facts.
        Without a stop at "Insulated" the sheath material read as
        "Polyethylene (XLPE) Insulated Thermoplastic".
        """
        materials = dict(S._materials_in(XLPE_7098))
        assert materials["sheathed"] == "Thermoplastic"

    def test_material_identity_ignores_case_and_spacing(self) -> None:
        """
        The catalogue spells one material both "... Free Electrical" and
        "... free Electrical". Treated as two, they collided on every alias and
        the alias was dropped as ambiguous — losing the material over a capital.
        """
        vocabulary = S._build_vocabulary([
            _std("IS 1", "Partial Discharge Free Electrical Insulated Wires"),
            _std("IS 2", "Partial Discharge free Electrical Insulated Wires"),
        ])
        keys = {m for (attr, m) in vocabulary.coverage if attr == "insulated"}
        assert len(keys) == 1, keys
        assert vocabulary.coverage[("insulated", keys.pop())] == [
            "IS 1:1988", "IS 2:1988",
        ]

    def test_function_words_are_not_mined_as_materials(self) -> None:
        """
        "Methods of tests for Covered ... wires" would otherwise mine "Methods
        of tests for" as a material name.
        """
        materials = dict(
            S._materials_in("Methods of tests for Covered Electrical Wires")
        )
        assert "covered" not in materials

    def test_an_ambiguous_alias_is_dropped_rather_than_guessed(self) -> None:
        """
        One word, two materials. Picking either would be a coin flip presented
        as a fact.
        """
        vocabulary = S._build_vocabulary([
            _std("IS 1", "Alpha Compound Insulated Cables"),
            _std("IS 2", "Alpha Polymer Insulated Cables"),
        ])
        assert vocabulary.material_for("insulated", "alpha") is None


# ===========================================================================
# Integration with the deterministic verdict
# ===========================================================================

class TestVerdictIntegration:
    """
    The check has to reach the verdict, and it has to reach it on the right
    axis. WRONG_SCOPE rather than INCORRECT_STANDARD: IS 1554 is real, live and
    correctly numbered, and INCORRECT_STANDARD is this codebase's verdict for a
    withdrawn standard.
    """

    def test_a_mismatch_produces_wrong_scope(self, cable_catalogue) -> None:
        from shared.models import Verdict
        from kartikey.analysis.compliance import run_compliance_checks

        result = run_compliance_checks(
            Requirement(
                analysis_id="a1", text=CABLE_TENDER,
                is_reference="IS 1554 : Part 1",
            ),
            _std("IS 1554 : Part 1", PVC_1554),
        )
        assert result.scope_check.mismatch
        assert result.suggested_verdict is Verdict.WRONG_SCOPE

    def test_the_scope_verdict_outranks_an_omitted_year(self, cable_catalogue) -> None:
        """
        The tender omits the year, which alone yields AMBIGUOUS. Reporting that
        first would tell the officer to pin the edition of a standard that does
        not cover their product — advice worse than silence, because it reads as
        having been checked.
        """
        from shared.models import Verdict
        from kartikey.analysis.compliance import run_compliance_checks

        result = run_compliance_checks(
            Requirement(
                analysis_id="a1", text=CABLE_TENDER,
                is_reference="IS 1554 : Part 1", cited_year=None,
            ),
            _std("IS 1554 : Part 1", PVC_1554),
        )
        assert result.version_check.is_year_omitted
        assert result.suggested_verdict is Verdict.WRONG_SCOPE

    def test_a_withdrawn_standard_still_outranks_the_scope_check(
        self, cable_catalogue,
    ) -> None:
        """
        A withdrawn standard cannot be used for procurement whatever it covers,
        so that fact leads.
        """
        from shared.models import Verdict
        from kartikey.analysis.compliance import run_compliance_checks

        result = run_compliance_checks(
            Requirement(
                analysis_id="a1", text=CABLE_TENDER,
                is_reference="IS 1554 : Part 1",
            ),
            _std("IS 1554 : Part 1", PVC_1554, status=StandardStatus.WITHDRAWN),
        )
        assert result.scope_check.mismatch
        assert result.suggested_verdict is Verdict.INCORRECT_STANDARD


# ===========================================================================
# Degenerate input
# ===========================================================================

class TestDegenerateInput:

    def test_empty_requirement_text(self, cable_catalogue) -> None:
        result = check_scope(
            Requirement(analysis_id="a1", text=""),
            _std("IS 1554 : Part 1", PVC_1554),
        )
        assert not result.checked

    def test_standard_with_no_title(self, cable_catalogue) -> None:
        result = check_scope(_req(CABLE_TENDER), _std("IS 1554 : Part 1", ""))
        assert not result.checked

    def test_the_requirement_is_not_mutated(self, cable_catalogue) -> None:
        requirement = _req(CABLE_TENDER)
        before = requirement.model_dump()
        check_scope(requirement, _std("IS 1554 : Part 1", PVC_1554))
        assert requirement.model_dump() == before
