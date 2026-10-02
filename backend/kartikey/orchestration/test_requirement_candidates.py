"""
kartikey/orchestration/test_requirement_candidates.py

`_step_retrieve` records which standards were retrieved for which requirement.
`_step_analyze` sends each requirement only its own candidates (with its own
scores), and `_step_enrich` refuses to attach any standard outside them. The
global `retrieved_standards` list that other steps depend on is unchanged.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from shared.contracts import AimlFinding, AimlResponse
from shared.models import (
    Analysis,
    AnalysisStatus,
    InputType,
    Requirement,
    Standard,
    StandardStatus,
    Verdict,
)
from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry
from kartikey.orchestration.pipeline import _step_analyze, _step_enrich, _step_retrieve


@pytest.fixture
def registry(monkeypatch):
    registry = initialize_knowledge_registry()
    registry.standards_store.clear()
    registry.evidence_store.clear()
    # Keep the fire-and-forget BIS sync off the network.
    monkeypatch.setattr("shared.config.settings.enable_bis_sync", False)
    return registry


def _std(is_number: str) -> Standard:
    return Standard(
        is_number=is_number, year=2020, title=f"Standard {is_number}",
        status=StandardStatus.ACTIVE,
    )


def _stub_search(results_by_query: dict[str, list[tuple[Standard, float]]]):
    """Retrieval stub: the query whose key appears in the query text wins."""
    def search_standards(query):
        for key, hits in results_by_query.items():
            if key in query.query_text:
                return SimpleNamespace(candidates=[
                    SimpleNamespace(standard=std.model_copy(), score=score)
                    for std, score in hits
                ])
        return SimpleNamespace(candidates=[])
    return search_standards


def _setup(registry, monkeypatch):
    """
    Requirement A cites IS 1001 and its search finds IS 1002 and (weakly) IS 1003.
    Requirement B has no citation and its search finds IS 1003 and IS 1004.
    """
    stds = {n: _std(n) for n in ("IS 1001", "IS 1002", "IS 1003", "IS 1004")}
    for std in stds.values():
        registry.standards_store.add(std)

    monkeypatch.setattr(
        registry.retrieval_service,
        "search_standards",
        _stub_search({
            "IS 1001": [(stds["IS 1002"], 0.6), (stds["IS 1003"], 0.2)],
            "warranty": [(stds["IS 1003"], 0.9), (stds["IS 1004"], 0.5)],
        }),
    )

    analysis = Analysis(
        input_type=InputType.TEXT, raw_text="tender", status=AnalysisStatus.QUEUED,
    )
    req_a = Requirement(
        analysis_id=analysis.id, text="Shall conform to IS 1001.", is_reference="IS 1001",
    )
    req_b = Requirement(analysis_id=analysis.id, text="Five-year warranty required.")
    analysis.requirements = [req_a, req_b]
    return analysis, req_a, req_b, stds


def test_step_retrieve_records_per_requirement_candidates(registry, monkeypatch) -> None:
    analysis, req_a, req_b, stds = _setup(registry, monkeypatch)

    asyncio.run(_step_retrieve(analysis, "tender"))

    candidates = analysis.metadata["requirement_candidates"]
    assert set(candidates) == {req_a.id, req_b.id}
    assert candidates[req_a.id] == {
        stds["IS 1001"].id: 1.0,
        stds["IS 1002"].id: 0.6,
        stds["IS 1003"].id: 0.2,
    }
    # IS 1003 keeps each requirement's own score, not the global best.
    assert candidates[req_b.id] == {
        stds["IS 1003"].id: 0.9,
        stds["IS 1004"].id: 0.5,
    }
    # The cited standard is not a candidate of the requirement that did not cite it.
    assert stds["IS 1001"].id not in candidates[req_b.id]


def test_step_retrieve_global_list_unchanged(registry, monkeypatch) -> None:
    """6. The global retrieved_standards list keeps its shape and ordering."""
    analysis, _, _, stds = _setup(registry, monkeypatch)

    retrieved = asyncio.run(_step_retrieve(analysis, "tender"))

    # Exact matches first, then distinct search hits by best score across all
    # requirements — exactly as before.
    assert [s.is_number for s in retrieved] == ["IS 1001", "IS 1003", "IS 1002", "IS 1004"]
    assert retrieved[0].relevance_score == 1.0
    assert {s.is_number: s.relevance_score for s in retrieved[1:]} == {
        "IS 1003": 0.9, "IS 1002": 0.6, "IS 1004": 0.5,
    }


def test_step_enrich_enforces_candidates_from_retrieve(registry, monkeypatch) -> None:
    """AI output proposing every retrieved standard for both requirements."""
    analysis, req_a, req_b, stds = _setup(registry, monkeypatch)

    async def _run():
        retrieved = await _step_retrieve(analysis, "tender")
        every_id = [s.id for s in retrieved] + ["unknown-id"]
        response = AimlResponse(
            analysis_id=analysis.id,
            findings=[
                AimlFinding(
                    finding_id=f"f-{req.id[:8]}", requirement_id=req.id,
                    verdict=Verdict.JUSTIFIED.value, reason="stub", confidence=0.9,
                    applicable_standard_ids=every_id,
                )
                for req in (req_a, req_b)
            ],
        )
        await _step_enrich(analysis, retrieved, response)

    asyncio.run(_run())

    attached = {
        f.requirement_id: {s.is_number for s in f.applicable_standards + f.cited_standards}
        for f in analysis.findings
    }
    assert attached[req_a.id] == {"IS 1001", "IS 1002", "IS 1003"}
    assert attached[req_b.id] == {"IS 1003", "IS 1004"}


def test_step_enrich_without_retrieve_attaches_nothing(registry) -> None:
    """No candidate map means no requirement has candidates — no unscoped fallback."""
    std = _std("IS 1001")
    registry.standards_store.add(std)
    analysis = Analysis(
        input_type=InputType.TEXT, raw_text="tender", status=AnalysisStatus.QUEUED,
    )
    req = Requirement(analysis_id=analysis.id, text="Five-year warranty required.")
    analysis.requirements = [req]
    response = AimlResponse(
        analysis_id=analysis.id,
        findings=[AimlFinding(
            finding_id="f1", requirement_id=req.id, verdict=Verdict.JUSTIFIED.value,
            reason="stub", confidence=0.9, applicable_standard_ids=[std.id],
        )],
    )

    asyncio.run(_step_enrich(analysis, [std], response))

    assert "requirement_candidates" not in analysis.metadata
    assert analysis.findings[0].applicable_standards == []
    assert analysis.findings[0].cited_standards == []


# ===========================================================================
# Stage 2 — the AI/ML request carries each requirement's own candidates
# ===========================================================================

def _capture_request(monkeypatch):
    """Run the backend mock engine, keeping the AimlRequest it was given."""
    from kshiraj.aiml_client.client import AimlClient

    captured = {}
    original = AimlClient._run_mock_analysis

    def _spy(self, request):
        captured["request"] = request
        return original(self, request)

    monkeypatch.setattr("shared.config.settings.aiml_service_url", "")
    monkeypatch.setattr(AimlClient, "_run_mock_analysis", _spy)
    return captured


def test_step_analyze_sends_per_requirement_candidates(registry, monkeypatch) -> None:
    """A, B, C: own candidates only, own scores, pooled list kept for compatibility."""
    analysis, req_a, req_b, stds = _setup(registry, monkeypatch)
    captured = _capture_request(monkeypatch)

    async def _run():
        retrieved = await _step_retrieve(analysis, "tender")
        await _step_analyze(analysis, "tender", retrieved)
        return retrieved

    retrieved = asyncio.run(_run())
    request = captured["request"]
    per_req = {
        rid: {s.is_number: (s.relevance_score, s.semantic_score) for s in cands}
        for rid, cands in request.requirement_candidates.items()
    }

    assert per_req[req_a.id] == {
        "IS 1001": (1.0, 1.0), "IS 1002": (0.6, None), "IS 1003": (0.2, None),
    }
    assert per_req[req_b.id] == {"IS 1003": (0.9, None), "IS 1004": (0.5, None)}
    # Standards retrieved only for A never reach B's input.
    assert not {"IS 1001", "IS 1002"} & set(per_req[req_b.id])
    # Highest requirement-specific score first.
    assert [s.is_number for s in request.requirement_candidates[req_b.id]] == ["IS 1003", "IS 1004"]
    # The pooled list is still sent, unchanged, for compatibility.
    assert [s.id for s in request.retrieved_standards] == [s.id for s in retrieved]
    # The shared catalogue objects are not mutated by the per-requirement copies.
    assert registry.standards_store.get_by_id(stds["IS 1003"].id).relevance_score is None


def test_exact_citation_keeps_its_score_and_leads(registry, monkeypatch) -> None:
    """Lexical scores are unnormalised; they must not displace an exact citation."""
    analysis, req_a, _, stds = _setup(registry, monkeypatch)
    captured = _capture_request(monkeypatch)
    monkeypatch.setattr(
        registry.retrieval_service,
        "search_standards",
        _stub_search({"IS 1001": [(stds["IS 1002"], 30.0), (stds["IS 1001"], 26.5)]}),
    )

    async def _run():
        retrieved = await _step_retrieve(analysis, "tender")
        await _step_analyze(analysis, "tender", retrieved)

    asyncio.run(_run())

    assert analysis.metadata["requirement_candidates"][req_a.id][stds["IS 1001"].id] == 1.0
    ordered = [
        (s.is_number, s.relevance_score)
        for s in captured["request"].requirement_candidates[req_a.id]
    ]
    assert ordered == [("IS 1001", 1.0), ("IS 1002", 30.0)]


def test_step_analyze_requirement_without_candidates_gets_none(registry, monkeypatch) -> None:
    """F: an empty candidate set is sent as empty — nothing borrowed from the pool."""
    analysis, req_a, req_b, stds = _setup(registry, monkeypatch)
    captured = _capture_request(monkeypatch)
    req_c = Requirement(analysis_id=analysis.id, text="Unmatched clause.")
    analysis.requirements.append(req_c)

    async def _run():
        retrieved = await _step_retrieve(analysis, "tender")
        response = await _step_analyze(analysis, "tender", retrieved)
        await _step_enrich(analysis, retrieved, response)

    asyncio.run(_run())

    assert captured["request"].requirement_candidates[req_c.id] == []
    finding_c = next(f for f in analysis.findings if f.requirement_id == req_c.id)
    assert finding_c.applicable_standards == []
    assert finding_c.verdict == Verdict.REQUIRES_HUMAN_VERIFICATION


def test_step_analyze_without_map_sends_no_candidates(registry, monkeypatch) -> None:
    """Called without a retrieval pass, no requirement is given the pooled list."""
    captured = _capture_request(monkeypatch)
    std = _std("IS 1001")
    registry.standards_store.add(std)
    analysis = Analysis(
        input_type=InputType.TEXT, raw_text="tender", status=AnalysisStatus.QUEUED,
    )
    req = Requirement(analysis_id=analysis.id, text="Conform to IS 1001.", is_reference="IS 1001")
    analysis.requirements = [req]

    response = asyncio.run(_step_analyze(analysis, "tender", [std]))

    assert captured["request"].requirement_candidates == {req.id: []}
    assert response.findings[0].applicable_standard_ids == []


def test_mock_engine_end_to_end_keeps_requirements_isolated(registry, monkeypatch) -> None:
    """
    Through retrieve → analyze (mock engine) → enrich: the citing requirement maps
    to its own cited standard, the non-citing one is not handed the global top
    standard as it used to be.
    """
    analysis, req_a, req_b, stds = _setup(registry, monkeypatch)
    _capture_request(monkeypatch)

    async def _run():
        retrieved = await _step_retrieve(analysis, "tender")
        response = await _step_analyze(analysis, "tender", retrieved)
        await _step_enrich(analysis, retrieved, response)

    asyncio.run(_run())
    attached = {
        f.requirement_id: {s.is_number for s in f.applicable_standards + f.cited_standards}
        for f in analysis.findings
    }

    assert attached[req_a.id] == {"IS 1001"}
    assert attached[req_b.id] == set()


# ---------------------------------------------------------------------------
# Display scaling of the shared relevance_score (lexical score-scale fix)
#
# Lexical scores are unbounded weighted sums (e.g. 20.0); the frontend reads
# relevance_score as [0, 1] or a percentage, so 20.0 was shown as 20%. Only the
# pooled/display copy is scaled, by raw / max(1.0, top score of that query);
# the AI/ML engine's per-requirement input keeps the raw scores.
# ---------------------------------------------------------------------------


def _lexical_setup(registry, monkeypatch, hits_by_query, requirements):
    """Register the standards named in hits_by_query and stub the search."""
    stds: dict[str, Standard] = {}
    for hits in hits_by_query.values():
        for number, _ in hits:
            if number not in stds:
                stds[number] = _std(number)
                registry.standards_store.add(stds[number])
    monkeypatch.setattr(
        registry.retrieval_service,
        "search_standards",
        _stub_search({
            key: [(stds[number], score) for number, score in hits]
            for key, hits in hits_by_query.items()
        }),
    )
    analysis = Analysis(
        input_type=InputType.TEXT, raw_text="tender", status=AnalysisStatus.QUEUED,
    )
    analysis.requirements = [
        Requirement(analysis_id=analysis.id, text=text, is_reference=ref)
        for text, ref in requirements
    ]
    return analysis, stds


def test_lexical_scores_are_scaled_on_the_shared_copy(registry, monkeypatch) -> None:
    """1. Lexical {20, 10, 1.0} → shared {1.0, 0.5, 0.05}, in the same order."""
    analysis, _ = _lexical_setup(
        registry, monkeypatch,
        {"luminaire": [("IS 2001", 20.0), ("IS 2002", 10.0), ("IS 2003", 1.0)]},
        [("LED luminaire for street lighting.", None)],
    )

    retrieved = asyncio.run(_step_retrieve(analysis, "tender"))

    assert [(s.is_number, s.relevance_score) for s in retrieved] == [
        ("IS 2001", 1.0), ("IS 2002", 0.5), ("IS 2003", 0.05),
    ]


def test_lexical_scaling_leaves_ai_candidate_scores_raw(registry, monkeypatch) -> None:
    """2. requirement_candidates and the AI/ML request keep the raw lexical scores."""
    analysis, stds = _lexical_setup(
        registry, monkeypatch,
        {"luminaire": [("IS 2001", 20.0), ("IS 2002", 10.0), ("IS 2003", 1.0)]},
        [("LED luminaire for street lighting.", None)],
    )
    req = analysis.requirements[0]
    captured = _capture_request(monkeypatch)

    async def _run():
        retrieved = await _step_retrieve(analysis, "tender")
        await _step_analyze(analysis, "tender", retrieved)

    asyncio.run(_run())

    assert analysis.metadata["requirement_candidates"][req.id] == {
        stds["IS 2001"].id: 20.0, stds["IS 2002"].id: 10.0, stds["IS 2003"].id: 1.0,
    }
    assert [
        (s.is_number, s.relevance_score)
        for s in captured["request"].requirement_candidates[req.id]
    ] == [("IS 2001", 20.0), ("IS 2002", 10.0), ("IS 2003", 1.0)]


def test_lexical_scaling_keeps_exact_citation_at_one(registry, monkeypatch) -> None:
    """3. An exact citation stays 1.0 and first, whatever the lexical scale."""
    analysis, stds = _lexical_setup(
        registry, monkeypatch,
        {"IS 2001": [("IS 2002", 30.0), ("IS 2001", 26.5), ("IS 2003", 15.0)]},
        [("Shall conform to IS 2001.", "IS 2001")],
    )
    # get_by_is_number finds the cited standard; the stub search returns the rest.
    retrieved = asyncio.run(_step_retrieve(analysis, "tender"))

    assert [(s.is_number, s.relevance_score) for s in retrieved] == [
        ("IS 2001", 1.0), ("IS 2002", 1.0), ("IS 2003", 0.5),
    ]
    assert retrieved[0].semantic_score == 1.0
    req = analysis.requirements[0]
    assert analysis.metadata["requirement_candidates"][req.id][stds["IS 2001"].id] == 1.0


def test_hybrid_scores_are_unchanged(registry, monkeypatch) -> None:
    """4. Scores already in [0, 1] (hybrid) pass through untouched."""
    analysis, stds = _lexical_setup(
        registry, monkeypatch,
        {"luminaire": [("IS 2001", 0.86), ("IS 2002", 0.43), ("IS 2003", 0.72)]},
        [("LED luminaire for street lighting.", None)],
    )
    req = analysis.requirements[0]

    retrieved = asyncio.run(_step_retrieve(analysis, "tender"))

    assert [(s.is_number, s.relevance_score) for s in retrieved] == [
        ("IS 2001", 0.86), ("IS 2003", 0.72), ("IS 2002", 0.43),
    ]
    assert analysis.metadata["requirement_candidates"][req.id] == {
        stds["IS 2001"].id: 0.86, stds["IS 2002"].id: 0.43, stds["IS 2003"].id: 0.72,
    }


def test_lexical_scaling_keeps_raw_retrieval_order(registry, monkeypatch) -> None:
    """
    5. Order across requirements still follows the raw score. IS 2004 tops its
    own query (display 1.0) but has the lowest raw score, so it stays last.
    """
    analysis, _ = _lexical_setup(
        registry, monkeypatch,
        {
            "luminaire": [("IS 2001", 30.0), ("IS 2002", 25.0)],
            "warranty": [("IS 2004", 20.0)],
        },
        [("LED luminaire for street lighting.", None), ("Five-year warranty required.", None)],
    )

    retrieved = asyncio.run(_step_retrieve(analysis, "tender"))

    assert [s.is_number for s in retrieved] == ["IS 2001", "IS 2002", "IS 2004"]
    assert [s.relevance_score for s in retrieved] == pytest.approx([1.0, 25.0 / 30.0, 1.0])
