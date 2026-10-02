"""
Tests for kshiraj/knowledge/standards_store.py.

Run from the backend/ directory:
    pytest kshiraj/knowledge/test_standards_store.py -v

All tests are synchronous; the store itself is not async.
"""

from __future__ import annotations

import threading
import uuid

import pytest

from shared.models import Standard, StandardStatus
from kshiraj.knowledge.standards_store import DuplicateStandardIDError, StandardsStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_standard(
    is_number: str = "IS 10322",
    title: str = "Test Standard",
    status: StandardStatus = StandardStatus.ACTIVE,
    year: int | None = 2012,
    part: str | None = None,
    section: str | None = None,
) -> Standard:
    """Factory for test Standard objects with a fresh UUID each call."""
    return Standard(
        id=str(uuid.uuid4()),
        is_number=is_number,
        title=title,
        status=status,
        year=year,
        part=part,
        section=section,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> StandardsStore:
    """Fresh StandardsStore for each test."""
    return StandardsStore()


# ---------------------------------------------------------------------------
# add + get_by_id
# ---------------------------------------------------------------------------


class TestAddAndGetById:
    def test_add_then_get_returns_same_object(self, store: StandardsStore) -> None:
        std = _make_standard()
        store.add(std)
        result = store.get_by_id(std.id)
        assert result is not None
        assert result.id == std.id
        assert result.title == std.title

    def test_get_by_id_missing_returns_none(self, store: StandardsStore) -> None:
        assert store.get_by_id("nonexistent-id") is None

    def test_add_multiple_distinct_ids(self, store: StandardsStore) -> None:
        s1 = _make_standard(title="Alpha")
        s2 = _make_standard(title="Beta")
        store.add(s1)
        store.add(s2)
        assert store.get_by_id(s1.id).title == "Alpha"
        assert store.get_by_id(s2.id).title == "Beta"


# ---------------------------------------------------------------------------
# Normalised IS-number lookup
# ---------------------------------------------------------------------------


class TestGetByIsNumber:
    def test_exact_match(self, store: StandardsStore) -> None:
        std = _make_standard(is_number="IS 10322")
        store.add(std)
        results = store.get_by_is_number("IS 10322")
        assert len(results) == 1
        assert results[0].id == std.id

    def test_case_insensitive_lookup(self, store: StandardsStore) -> None:
        std = _make_standard(is_number="IS 10322")
        store.add(std)
        assert store.get_by_is_number("is 10322") != []
        assert store.get_by_is_number("IS 10322") != []
        assert store.get_by_is_number("Is 10322") != []

    def test_leading_trailing_whitespace_stripped(self, store: StandardsStore) -> None:
        std = _make_standard(is_number="IS 10322")
        store.add(std)
        assert len(store.get_by_is_number("  IS 10322  ")) == 1

    def test_no_match_returns_empty_list(self, store: StandardsStore) -> None:
        store.add(_make_standard(is_number="IS 10322"))
        assert store.get_by_is_number("IS 9999") == []

    def test_mixed_case_stored_number(self, store: StandardsStore) -> None:
        std = _make_standard(is_number="iS 10322")
        store.add(std)
        assert len(store.get_by_is_number("IS 10322")) == 1


# ---------------------------------------------------------------------------
# Multiple versions / parts sharing a standard number
# ---------------------------------------------------------------------------


class TestMultipleVersions:
    def test_two_years_same_is_number(self, store: StandardsStore) -> None:
        v1 = _make_standard(is_number="IS 10322", year=2000)
        v2 = _make_standard(is_number="IS 10322", year=2012)
        store.add(v1)
        store.add(v2)
        results = store.get_by_is_number("IS 10322")
        ids = {r.id for r in results}
        assert ids == {v1.id, v2.id}

    def test_parts_under_same_is_number(self, store: StandardsStore) -> None:
        p1 = _make_standard(is_number="IS 10322", part="Part 1")
        p2 = _make_standard(is_number="IS 10322", part="Part 2")
        p3 = _make_standard(is_number="IS 10322", part="Part 3")
        for p in (p1, p2, p3):
            store.add(p)
        results = store.get_by_is_number("IS 10322")
        assert len(results) == 3

    def test_parts_and_sections_under_same_is_number(self, store: StandardsStore) -> None:
        s1 = _make_standard(is_number="IS 10322", part="Part 5", section="Sec 3")
        s2 = _make_standard(is_number="IS 10322", part="Part 5", section="Sec 5")
        store.add(s1)
        store.add(s2)
        results = store.get_by_is_number("IS 10322")
        assert len(results) == 2

    def test_different_is_numbers_are_isolated(self, store: StandardsStore) -> None:
        a = _make_standard(is_number="IS 10322")
        b = _make_standard(is_number="IS 694")
        store.add(a)
        store.add(b)
        assert len(store.get_by_is_number("IS 10322")) == 1
        assert len(store.get_by_is_number("IS 694")) == 1


# ---------------------------------------------------------------------------
# Upsert behaviour
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_upsert_new_standard_is_stored(self, store: StandardsStore) -> None:
        std = _make_standard()
        store.upsert(std)
        assert store.get_by_id(std.id) is not None

    def test_upsert_replaces_existing(self, store: StandardsStore) -> None:
        std = _make_standard(title="Original")
        store.add(std)
        updated = std.model_copy(update={"title": "Updated"})
        store.upsert(updated)
        result = store.get_by_id(std.id)
        assert result.title == "Updated"

    def test_upsert_updates_number_index_when_is_number_changes(
        self, store: StandardsStore
    ) -> None:
        std = _make_standard(is_number="IS 10322")
        store.add(std)
        renamed = std.model_copy(update={"is_number": "IS 9999"})
        store.upsert(renamed)
        # Old number should no longer find it
        assert store.get_by_is_number("IS 10322") == []
        # New number should find it
        assert len(store.get_by_is_number("IS 9999")) == 1

    def test_upsert_count_does_not_grow_on_replace(self, store: StandardsStore) -> None:
        std = _make_standard()
        store.add(std)
        store.upsert(std.model_copy(update={"title": "Changed"}))
        assert store.count() == 1


# ---------------------------------------------------------------------------
# Remove behaviour
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_existing_returns_true(self, store: StandardsStore) -> None:
        std = _make_standard()
        store.add(std)
        assert store.remove(std.id) is True

    def test_remove_nonexistent_returns_false(self, store: StandardsStore) -> None:
        assert store.remove("ghost-id") is False

    def test_removed_standard_not_retrievable_by_id(self, store: StandardsStore) -> None:
        std = _make_standard()
        store.add(std)
        store.remove(std.id)
        assert store.get_by_id(std.id) is None

    def test_removed_standard_not_retrievable_by_number(
        self, store: StandardsStore
    ) -> None:
        std = _make_standard(is_number="IS 10322")
        store.add(std)
        store.remove(std.id)
        assert store.get_by_is_number("IS 10322") == []

    def test_remove_one_of_two_leaves_other(self, store: StandardsStore) -> None:
        s1 = _make_standard(is_number="IS 10322", year=2000)
        s2 = _make_standard(is_number="IS 10322", year=2012)
        store.add(s1)
        store.add(s2)
        store.remove(s1.id)
        remaining = store.get_by_is_number("IS 10322")
        assert len(remaining) == 1
        assert remaining[0].id == s2.id


# ---------------------------------------------------------------------------
# Status filtering
# ---------------------------------------------------------------------------


class TestStatusFiltering:
    def test_filter_by_active(self, store: StandardsStore) -> None:
        active = _make_standard(status=StandardStatus.ACTIVE)
        superseded = _make_standard(status=StandardStatus.SUPERSEDED)
        withdrawn = _make_standard(status=StandardStatus.WITHDRAWN)
        for s in (active, superseded, withdrawn):
            store.add(s)
        results = store.filter_by_status(StandardStatus.ACTIVE)
        assert len(results) == 1
        assert results[0].id == active.id

    def test_filter_by_superseded(self, store: StandardsStore) -> None:
        store.add(_make_standard(status=StandardStatus.ACTIVE))
        r1 = _make_standard(status=StandardStatus.SUPERSEDED)
        r2 = _make_standard(status=StandardStatus.SUPERSEDED)
        store.add(r1)
        store.add(r2)
        results = store.filter_by_status(StandardStatus.SUPERSEDED)
        ids = {s.id for s in results}
        assert ids == {r1.id, r2.id}

    def test_filter_by_under_revision(self, store: StandardsStore) -> None:
        under = _make_standard(status=StandardStatus.UNDER_REVISION)
        store.add(under)
        results = store.filter_by_status(StandardStatus.UNDER_REVISION)
        assert len(results) == 1
        assert results[0].id == under.id

    def test_filter_by_reaffirmed(self, store: StandardsStore) -> None:
        reaffirmed = _make_standard(status=StandardStatus.REAFFIRMED)
        store.add(reaffirmed)
        results = store.filter_by_status(StandardStatus.REAFFIRMED)
        assert len(results) == 1
        assert results[0].id == reaffirmed.id

    def test_filter_empty_when_no_match(self, store: StandardsStore) -> None:
        store.add(_make_standard(status=StandardStatus.ACTIVE))
        assert store.filter_by_status(StandardStatus.WITHDRAWN) == []

    def test_get_active_is_alias_for_active_filter(self, store: StandardsStore) -> None:
        a = _make_standard(status=StandardStatus.ACTIVE)
        w = _make_standard(status=StandardStatus.WITHDRAWN)
        store.add(a)
        store.add(w)
        active = store.get_active()
        assert len(active) == 1
        assert active[0].id == a.id

    def test_all_current_statuses_representable(self, store: StandardsStore) -> None:
        """Every member of the current StandardStatus enum can be stored and filtered."""
        for status in StandardStatus:
            std = _make_standard(status=status)
            store.add(std)
        for status in StandardStatus:
            assert len(store.filter_by_status(status)) == 1


# ---------------------------------------------------------------------------
# Count + list_all behaviour
# ---------------------------------------------------------------------------


class TestCountAndListAll:
    def test_count_starts_at_zero(self, store: StandardsStore) -> None:
        assert store.count() == 0

    def test_count_increments_on_add(self, store: StandardsStore) -> None:
        store.add(_make_standard())
        assert store.count() == 1
        store.add(_make_standard())
        assert store.count() == 2

    def test_count_decrements_on_remove(self, store: StandardsStore) -> None:
        std = _make_standard()
        store.add(std)
        store.remove(std.id)
        assert store.count() == 0

    def test_list_all_returns_all_standards(self, store: StandardsStore) -> None:
        stds = [_make_standard(title=f"S{i}") for i in range(5)]
        for s in stds:
            store.add(s)
        listed = store.list_all()
        assert len(listed) == 5
        assert {s.id for s in listed} == {s.id for s in stds}

    def test_list_all_empty_store(self, store: StandardsStore) -> None:
        assert store.list_all() == []

    def test_list_all_returns_snapshot_not_live_view(
        self, store: StandardsStore
    ) -> None:
        std = _make_standard()
        store.add(std)
        snapshot = store.list_all()
        store.remove(std.id)
        # Snapshot taken before removal should still contain the standard
        assert len(snapshot) == 1

    def test_clear_resets_count_to_zero(self, store: StandardsStore) -> None:
        for _ in range(3):
            store.add(_make_standard())
        store.clear()
        assert store.count() == 0
        assert store.list_all() == []


# ---------------------------------------------------------------------------
# Duplicate-ID behaviour
# ---------------------------------------------------------------------------


class TestDuplicateIDHandling:
    def test_add_duplicate_id_raises(self, store: StandardsStore) -> None:
        std = _make_standard()
        store.add(std)
        duplicate = std.model_copy()  # same id
        with pytest.raises(DuplicateStandardIDError):
            store.add(duplicate)

    def test_store_unchanged_after_duplicate_add_fails(
        self, store: StandardsStore
    ) -> None:
        std = _make_standard(title="Original")
        store.add(std)
        duplicate = std.model_copy(update={"title": "Intruder"})
        with pytest.raises(DuplicateStandardIDError):
            store.add(duplicate)
        # The original must be intact
        assert store.get_by_id(std.id).title == "Original"
        assert store.count() == 1

    def test_upsert_does_not_raise_on_existing_id(self, store: StandardsStore) -> None:
        std = _make_standard()
        store.add(std)
        # Should not raise
        store.upsert(std.model_copy(update={"title": "Replaced"}))

    def test_number_index_not_duplicated_after_repeated_upsert(
        self, store: StandardsStore
    ) -> None:
        std = _make_standard(is_number="IS 10322")
        store.upsert(std)
        store.upsert(std.model_copy(update={"title": "Again"}))
        results = store.get_by_is_number("IS 10322")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Basic thread-safety smoke test
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_adds_produce_correct_count(self, store: StandardsStore) -> None:
        """Concurrent inserts from multiple threads should not lose records."""
        standards = [_make_standard(title=f"S{i}") for i in range(50)]
        errors: list[Exception] = []

        def _add(s: Standard) -> None:
            try:
                store.add(s)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_add, args=(s,)) for s in standards]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors during concurrent add: {errors}"
        assert store.count() == 50


# ---------------------------------------------------------------------------
# Family (part-stripped) resolution
# ---------------------------------------------------------------------------


class TestFamilyResolution:
    """
    BIS publishes many standards only as parts. There is no document numbered
    plain "IS 10322" — only "IS 10322 : Part 1", "IS 10322 : Part 5 : Sec 3" and
    so on. Tenders and normative-reference lists nevertheless cite the family
    ("shall conform to IS 10322"), so exact-key lookup alone returned nothing for
    a perfectly valid citation and the requirement lost its candidate standard.

    The exact key still wins wherever it hits, so nothing that resolved before
    resolves differently now.
    """

    def test_family_citation_resolves_to_every_part(self, store: StandardsStore) -> None:
        for part in ("IS 10322 : Part 1", "IS 10322 : Part 5 : Sec 3",
                     "IS 10322 : Part 5 : Sec 5"):
            store.add(_make_standard(is_number=part))

        results = store.get_by_is_number("IS 10322")

        assert {s.is_number for s in results} == {
            "IS 10322 : Part 1", "IS 10322 : Part 5 : Sec 3", "IS 10322 : Part 5 : Sec 5",
        }

    def test_an_exact_match_is_never_widened(self, store: StandardsStore) -> None:
        """
        The distinction that matters: a tender citing IS 694 wants IS 694, not
        every standard whose number starts with 694.
        """
        store.add(_make_standard(is_number="IS 694"))
        store.add(_make_standard(is_number="IS 694 : Part 1"))

        results = store.get_by_is_number("IS 694")

        assert [s.is_number for s in results] == ["IS 694"]

    def test_a_part_citation_stays_a_part_citation(self, store: StandardsStore) -> None:
        store.add(_make_standard(is_number="IS 7098 : Part 1"))
        store.add(_make_standard(is_number="IS 7098 : Part 2"))

        results = store.get_by_is_number("IS 7098 : Part 2")

        assert [s.is_number for s in results] == ["IS 7098 : Part 2"]

    def test_neighbouring_numbers_do_not_collide(self, store: StandardsStore) -> None:
        """
        The failure mode a prefix match would have: IS 1554 and IS 15544 are
        unrelated standards, and IS 15544 must not answer a query for IS 1554.
        """
        store.add(_make_standard(is_number="IS 1554 : Part 1"))
        store.add(_make_standard(is_number="IS 15544"))

        assert [s.is_number for s in store.get_by_is_number("IS 1554")] == [
            "IS 1554 : Part 1"
        ]
        assert [s.is_number for s in store.get_by_is_number("IS 15544")] == ["IS 15544"]

    @pytest.mark.parametrize(
        "cited",
        [
            "IS 10322 : Part 5 : Sec 3",
            "IS 10322 (Part 5/Sec 3)",
            "IS 10322: Part 5: Sec 3",
            "is 10322 : part 5 : sec 3",
        ],
    )
    def test_part_separator_styles_all_reduce_to_the_family(
        self, store: StandardsStore, cited: str,
    ) -> None:
        """The sources are not consistent about spacing or brackets."""
        store.add(_make_standard(is_number="IS 10322 : Part 1"))

        assert store.get_by_is_number(cited)[0].is_number == "IS 10322 : Part 1"

    def test_an_absent_standard_is_still_absent(self, store: StandardsStore) -> None:
        """
        Widening must not manufacture a hit. IS 8130 is genuinely not in the
        catalogue, and reporting a near-miss as a match would be worse than
        reporting nothing.
        """
        store.add(_make_standard(is_number="IS 1554 : Part 1"))

        assert store.get_by_is_number("IS 8130") == []

    def test_get_family_always_widens(self, store: StandardsStore) -> None:
        store.add(_make_standard(is_number="IS 10322 : Part 1"))
        store.add(_make_standard(is_number="IS 10322 : Part 5 : Sec 3"))

        family = store.get_family("IS 10322 : Part 1")

        assert len(family) == 2

    def test_the_family_index_survives_upsert(self, store: StandardsStore) -> None:
        std = _make_standard(is_number="IS 10322 : Part 1")
        store.add(std)
        store.upsert(std.model_copy(update={"is_number": "IS 16107"}))

        assert store.get_by_is_number("IS 10322") == []
        assert [s.is_number for s in store.get_by_is_number("IS 16107")] == ["IS 16107"]

    def test_the_family_index_survives_remove(self, store: StandardsStore) -> None:
        std = _make_standard(is_number="IS 10322 : Part 1")
        store.add(std)
        store.remove(std.id)

        assert store.get_by_is_number("IS 10322") == []

    def test_the_family_index_survives_clear(self, store: StandardsStore) -> None:
        store.add(_make_standard(is_number="IS 10322 : Part 1"))
        store.clear()

        assert store.get_by_is_number("IS 10322") == []
