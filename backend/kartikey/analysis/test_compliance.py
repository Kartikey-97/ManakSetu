"""
kartikey/analysis/test_compliance.py

Tests for the deterministic compliance rules.

The reason this file exists: currentness used to be decided twice. There was a
fully built, fully tested `kshiraj/enrichment/version_checker.py` that nothing
in the pipeline called, and a private `_check_version` here that the pipeline
did call — and the two disagreed in exactly the two cases where the answer is
not obvious:

    | case                       | VersionChecker | old _check_version |
    | cited > current            | not current    | CURRENT            |
    | standard.year is None      | not current    | CURRENT            |

Both of the old answers assert something that was never established. The second
one meant a tender citing "IS 10322:2035" — a typo, or an edition that does not
exist — came back JUSTIFIED at 0.85 confidence, because the verdict rules only
looked for a *positive* gap.

`_check_version` now delegates to VersionChecker and keeps only the prose. The
tests below pin the delegation, the prose, and the two verdicts that the honest
answer newly makes reachable.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from shared.models import (
    CertificationScheme,
    Requirement,
    Standard,
    StandardStatus,
    Verdict,
)

from kartikey.analysis import compliance as C
from kartikey.analysis.compliance import run_compliance_checks
from kshiraj.enrichment.version_checker import VersionChecker


# ===========================================================================
# Fixtures
# ===========================================================================

def _req(cited_year: int | None = 2012, is_reference: str = "IS 10322") -> Requirement:
    return Requirement(
        analysis_id="a1",
        text="Luminaires shall conform to IS 10322.",
        is_reference=is_reference,
        cited_year=cited_year,
    )


def _std(
    year: int | None = 2022,
    status: StandardStatus = StandardStatus.ACTIVE,
    **kwargs,
) -> Standard:
    return Standard(
        is_number="IS 10322",
        year=year,
        title="Luminaires",
        status=status,
        **kwargs,
    )


# ===========================================================================
# 1. One authority for the currentness decision
# ===========================================================================

class TestSingleVersionAuthority:
    """
    Every field except `note` must come from VersionChecker. A report that shows
    one set of numbers while the verdict was derived from another is worse than
    no report.
    """

    @pytest.mark.parametrize(
        "cited,current",
        [
            (2022, 2022),   # exact match
            (2012, 2022),   # outdated
            (2035, 2022),   # newer than the catalogue
            (None, 2022),   # year omitted
            (2012, None),   # candidate year unknown
        ],
    )
    def test_decision_fields_match_version_checker_exactly(self, cited, current):
        requirement, standard = _req(cited_year=cited), _std(year=current)

        expected = VersionChecker().check(requirement, standard)
        actual = run_compliance_checks(requirement, standard).version_check

        assert actual.cited_year == expected.cited_year
        assert actual.current_year == expected.current_year
        assert actual.is_current == expected.is_current
        assert actual.is_year_omitted == expected.is_year_omitted
        assert actual.gap_years == expected.gap_years

    def test_inputs_are_not_mutated(self):
        requirement, standard = _req(cited_year=2012), _std(year=2022)
        before = (requirement.model_dump(), standard.model_dump())

        run_compliance_checks(requirement, standard)

        assert (requirement.model_dump(), standard.model_dump()) == before

    def test_the_module_does_not_reimplement_the_comparison(self):
        """
        Guards against the duplicate coming back. If someone reintroduces inline
        arithmetic here, this fails rather than silently drifting again.
        """
        called: list[tuple] = []
        real = VersionChecker.check

        def _spy(self, requirement, standard):
            called.append((requirement.cited_year, standard.year))
            return real(self, requirement, standard)

        C._version_checker.__class__.check = _spy
        try:
            run_compliance_checks(_req(cited_year=2012), _std(year=2022))
        finally:
            C._version_checker.__class__.check = real

        assert called == [(2012, 2022)]


# ===========================================================================
# 2. Verdicts the honest answer newly makes reachable
# ===========================================================================

class TestCitedEditionNewerThanCatalogue:
    """
    The regression. `is_current = cited >= current` treated a future edition as
    current, and `gap_years > 0` never fired on a negative gap, so the rules fell
    all the way through to JUSTIFIED.
    """

    def test_future_edition_is_not_current(self):
        result = run_compliance_checks(_req(cited_year=2035), _std(year=2022))

        assert result.version_check.is_current is False
        assert result.version_check.gap_years == -13

    def test_future_edition_is_not_justified(self):
        result = run_compliance_checks(_req(cited_year=2035), _std(year=2022))

        assert result.suggested_verdict == Verdict.REQUIRES_HUMAN_VERIFICATION
        assert result.confidence < 0.60

    def test_note_says_which_direction_the_gap_runs(self):
        note = run_compliance_checks(_req(cited_year=2035), _std(year=2022)).version_check.note

        assert "newer" in note
        assert "13 year(s)" in note
        # Must not read as an outdated reference — the officer's action differs.
        assert "behind" not in note


class TestCandidateYearUnknown:
    """A year we do not have is not a year that matches."""

    def test_unknown_candidate_year_is_not_reported_as_current(self):
        result = run_compliance_checks(_req(cited_year=2012), _std(year=None))

        assert result.version_check.is_current is False
        assert result.version_check.gap_years is None

    def test_unknown_candidate_year_cannot_be_justified(self):
        result = run_compliance_checks(_req(cited_year=2012), _std(year=None))

        assert result.suggested_verdict == Verdict.UNABLE_TO_DETERMINE

    def test_note_tells_the_officer_where_to_check(self):
        note = run_compliance_checks(_req(cited_year=2012), _std(year=None)).version_check.note

        assert "cannot be established" in note.lower()
        assert "standardsbis.gov.in" in note


# ===========================================================================
# 3. Behaviour that must not have changed
# ===========================================================================

class TestUnchangedVerdicts:
    """
    Delegating the comparison must not move any verdict that was already right.
    """

    def test_exact_year_match_is_justified(self):
        result = run_compliance_checks(_req(cited_year=2022), _std(year=2022))

        assert result.suggested_verdict == Verdict.JUSTIFIED
        assert result.version_check.is_current is True
        assert "matches current edition" in result.version_check.note

    def test_small_gap_is_outdated_with_a_transition_hint(self):
        result = run_compliance_checks(_req(cited_year=2020), _std(year=2022))

        assert result.suggested_verdict == Verdict.OUTDATED_REFERENCE
        assert result.confidence == 0.70
        assert "transition period" in result.version_check.note

    def test_large_gap_is_outdated_with_higher_confidence(self):
        result = run_compliance_checks(_req(cited_year=2012), _std(year=2022))

        assert result.suggested_verdict == Verdict.OUTDATED_REFERENCE
        assert result.confidence == 0.80
        assert "Significant outdated reference" in result.version_check.note

    def test_omitted_year_is_ambiguous_and_cites_bis_convention(self):
        result = run_compliance_checks(_req(cited_year=None), _std(year=2022))

        assert result.suggested_verdict == Verdict.AMBIGUOUS
        assert result.version_check.is_year_omitted is True
        assert "latest edition including amendments" in result.version_check.note

    def test_withdrawn_standard_still_outranks_everything(self):
        """Status is a harder fact than a year; it must keep the headline."""
        result = run_compliance_checks(
            _req(cited_year=2022),
            _std(year=2022, status=StandardStatus.WITHDRAWN),
        )

        assert result.suggested_verdict == Verdict.INCORRECT_STANDARD
        assert result.confidence == 0.95

    def test_superseded_past_transition_is_outdated_at_high_confidence(self):
        result = run_compliance_checks(
            _req(cited_year=2022),
            _std(
                year=2022,
                status=StandardStatus.SUPERSEDED,
                superseded_by="IS 10322 (Part 5/Sec 3)",
                transition_deadline=date.today() - timedelta(days=1),
            ),
        )

        assert result.suggested_verdict == Verdict.OUTDATED_REFERENCE
        assert result.confidence == 0.95
        assert result.status_check.within_transition is False

    def test_superseded_within_transition_lowers_confidence(self):
        result = run_compliance_checks(
            _req(cited_year=2022),
            _std(
                year=2022,
                status=StandardStatus.SUPERSEDED,
                superseded_by="IS 10322 (Part 5/Sec 3)",
                transition_deadline=date.today() + timedelta(days=30),
            ),
        )

        assert result.suggested_verdict == Verdict.OUTDATED_REFERENCE
        assert result.status_check.within_transition is True
        assert result.status_check.is_usable is True

    def test_qco_evidence_is_built_from_the_gazette_reference(self):
        result = run_compliance_checks(
            _req(cited_year=2022),
            _std(
                year=2022,
                qco_notified=True,
                qco_gazette_so_number="S.O. 219(E)",
                qco_issuing_ministry="DPIIT",
                required_certification_scheme=CertificationScheme.ISI_MARK,
            ),
        )

        assert result.qco_check.qco_notified is True
        assert any(ev.gazette_so_number == "S.O. 219(E)" for ev in result.evidence)
        assert "CM/L" in result.qco_check.note
