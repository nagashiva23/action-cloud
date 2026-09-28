"""
Integration tests against a real Postgres (+pgvector). Each covers a bug that
existed in the previous version; the test name says which.
"""

from __future__ import annotations

import uuid

import pytest

from actioncloud.metrics import MetricCalculator
from actioncloud.policy import MemorySelectionPolicy
from actioncloud.schema import AgentRole, Experience, ExperienceCreate, MemoryTier, SystemCondition
from actioncloud.service import MemoryService

pytestmark = pytest.mark.usefixtures("database")


def _create(run_id: str, **kw) -> ExperienceCreate:
    base = dict(
        agent_id="author", agent_role=AgentRole.CODING,
        task="Configure pgvector HNSW index for cosine search",
        action="STEP: Create an HNSW index\nSTEP: Use vector_cosine_ops",
        solution="STEP: Create an HNSW index\nSTEP: Use vector_cosine_ops",
        result="done", success=True, run_id=run_id, system=SystemCondition.ACTIONCLOUD,
        technologies=["postgres", "pgvector"], task_key="k-hnsw",
    )
    base.update(kw)
    return ExperienceCreate(**base)


@pytest.fixture
def svc():
    return MemoryService(sync_write=True)


@pytest.fixture
def run_id():
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.mark.parametrize("role", list(AgentRole))
def test_every_role_can_be_stored(svc, run_id, role):
    """001_init.sql used to accept only 6 of the 12 roles."""
    eid = svc.store_experience(_create(run_id, agent_role=role))["id"]
    assert svc.get_experience(eid)["agent_role"] == role.value


def test_sync_write_runs_full_pipeline(svc, run_id):
    """SYNC_WRITE used to skip tiering, extraction and embedding."""
    row = svc.get_experience(svc.store_experience(_create(run_id))["id"])
    assert row["tier"] == "agent"
    assert row["embedded"] is True
    assert row["workflow"]["steps"] == ["Create an HNSW index", "Use vector_cosine_ops"]


def test_duplicate_delivery_is_a_noop(run_id):
    from actioncloud import db
    from actioncloud.pipeline import process_experience

    exp = Experience(**_create(run_id).model_dump())
    assert process_experience(exp) is True
    assert process_experience(exp) is False
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM tier_transitions WHERE experience_id = %s", (exp.id,))
        assert cur.fetchone()["n"] == 1


def _visible(svc, run_id, **who):
    return {r.id for r in svc.search_memory("pgvector HNSW cosine index", limit=20,
                                            scope_run_id=run_id, technologies=["postgres", "pgvector"], **who)}


def test_visibility_rules(svc, run_id):
    """AGENT-tier rows used to be fleet-visible; anonymous callers saw PRIVATE failures."""
    ok = svc.store_experience(_create(run_id))["id"]                         # AGENT tier
    failed = svc.store_experience(_create(run_id, success=False, solution=None))["id"]  # PRIVATE

    assert _visible(svc, run_id) == set()                                    # anonymous
    assert _visible(svc, run_id, agent_id="other", agent_role=AgentRole.SECURITY) == set()
    assert _visible(svc, run_id, agent_id="other", agent_role=AgentRole.CODING) == {ok}
    assert _visible(svc, run_id, agent_id="author", agent_role=AgentRole.CODING) == {ok, failed}


def test_run_scoping_isolates_arms(svc, run_id):
    svc.store_experience(_create(run_id))
    other = f"{run_id}-other"
    assert _visible(svc, other, agent_id="x", agent_role=AgentRole.CODING) == set()


def test_positive_self_report_not_counted_negative_is(svc, run_id):
    eid = svc.store_experience(_create(run_id))["id"]
    r = svc.record_reuse(eid, success=True, agent_id="author")
    assert r["counted"] is False and r["tier"] == "agent" and r["reuse_count"] == 0
    r = svc.record_reuse(eid, success=False, agent_id="author")
    assert r["counted"] is True and r["reuse_count"] == 1
    assert r["confidence"] == pytest.approx(0.4)


def test_promotion_by_another_agent_is_audited(svc, run_id):
    from actioncloud import db

    eid = svc.store_experience(_create(run_id))["id"]
    _inject(svc, run_id, "peer")
    r = svc.record_reuse(eid, success=True, agent_id="peer")
    assert r["transition"] == "shared"
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT from_tier::text, to_tier::text FROM tier_transitions "
                    "WHERE experience_id = %s ORDER BY id", (eid,))
        assert [(t["from_tier"], t["to_tier"]) for t in cur.fetchall()] == [(None, "agent"), ("agent", "shared")]


def _inject(svc, run_id, agent_id, role=AgentRole.CODING):
    return svc.prepare_context("Configure pgvector HNSW index for cosine search",
                               agent_id=agent_id, role=role,
                               technologies=["postgres", "pgvector"], scope_run_id=run_id,
                               policy=MemorySelectionPolicy())


def test_report_without_injection_not_counted(svc, run_id):
    """Exploit: reporting on a memory you were never shown."""
    eid = svc.store_experience(_create(run_id))["id"]
    r = svc.record_reuse(eid, success=True, agent_id="stranger")
    assert r["counted"] is False and r["tier"] == "agent"
    assert "no unreported injection" in r["reason"]


def test_one_injection_one_counted_report(svc, run_id):
    """Exploit: one agent reporting success over and over to self-promote a peer's memory."""
    eid = svc.store_experience(_create(run_id))["id"]
    _inject(svc, run_id, "peer")
    results = [svc.record_reuse(eid, success=True, agent_id="peer") for _ in range(10)]
    assert [r["counted"] for r in results] == [True] + [False] * 9
    assert results[-1]["reuse_count"] == 1 and results[-1]["tier"] == "shared"


def test_each_injection_earns_one_report(svc, run_id):
    eid = svc.store_experience(_create(run_id))["id"]
    for _ in range(3):
        _inject(svc, run_id, "peer")
    counted = [svc.record_reuse(eid, success=True, agent_id="peer")["counted"] for _ in range(4)]
    assert counted == [True, True, True, False]


def test_author_negative_self_report_counts_once(svc, run_id):
    eid = svc.store_experience(_create(run_id))["id"]
    a = svc.record_reuse(eid, success=False, agent_id="author")
    b = svc.record_reuse(eid, success=False, agent_id="author")
    assert a["counted"] is True and b["counted"] is False
    assert b["reuse_count"] == 1


def test_reuse_requires_reporter(svc, run_id):
    eid = svc.store_experience(_create(run_id))["id"]
    with pytest.raises(ValueError):
        svc.record_reuse(eid, success=True, agent_id=None)


def test_low_confidence_memory_not_injected(svc, run_id):
    eid = svc.store_experience(_create(run_id))["id"]
    ctx = lambda: svc.prepare_context("Configure pgvector HNSW index for cosine search",
                                      agent_id="peer", role=AgentRole.CODING,
                                      technologies=["postgres", "pgvector"], scope_run_id=run_id,
                                      policy=MemorySelectionPolicy())
    assert ctx()["injected_experience_ids"] == [str(eid)]
    svc.record_reuse(eid, success=False, agent_id="peer")                    # confidence 0.4
    assert ctx()["injected_experience_ids"] == []


def test_irrelevant_memory_not_injected(svc, run_id):
    """The builder used to inject the top match even below the threshold."""
    svc.store_experience(_create(run_id, task="Bake sourdough bread", technologies=["oven"]))
    res = svc.prepare_context("Configure pgvector HNSW index", agent_id="peer",
                              role=AgentRole.CODING, scope_run_id=run_id,
                              policy=MemorySelectionPolicy())
    assert res["final_count"] == 0 and res["context"] == ""


def test_metrics_endpoint_shape(svc, run_id):
    svc.store_experience(_create(run_id))
    m = MetricCalculator.calculate_run_metrics(run_id)
    assert m["systems"]["actioncloud"]["total_tasks"] == 1
    assert "agent" in m["governance_tier_distribution"]


def test_author_prior_off_by_default(svc, run_id):
    eid = svc.store_experience(_create(run_id, agent_id=f"a-{uuid.uuid4().hex[:6]}"))["id"]
    row = svc.get_experience(eid)
    assert row["prior"] == pytest.approx(0.6) and row["confidence"] == pytest.approx(0.6)


def test_author_prior_tracks_the_authors_record(run_id):
    """An author whose memories keep failing starts new memories below the injection floor."""
    svc = MemoryService(sync_write=True, author_prior=True)
    author = f"shaky-{uuid.uuid4().hex[:6]}"
    for i in range(3):
        eid = svc.store_experience(_create(run_id, agent_id=author, task_key=f"k{i}"))["id"]
        svc.record_reuse(eid, success=False, agent_id=author)       # counted once each
    fresh = svc.get_experience(svc.store_experience(_create(run_id, agent_id=author))["id"])
    assert fresh["prior"] == pytest.approx((0 + 1.2) / (3 + 2))     # 0.24
    assert fresh["confidence"] < MemorySelectionPolicy().min_confidence
    newcomer = svc.get_experience(svc.store_experience(_create(run_id, agent_id=f"new-{uuid.uuid4().hex[:6]}"))["id"])
    assert newcomer["prior"] == pytest.approx(0.6)                  # no record -> global prior


def test_confidence_posterior_uses_the_memory_prior(run_id):
    svc = MemoryService(sync_write=True, author_prior=True)
    author = f"good-{uuid.uuid4().hex[:6]}"
    eid = svc.store_experience(_create(run_id, agent_id=author))["id"]
    _inject(svc, run_id, "peer")
    r = svc.record_reuse(eid, success=True, agent_id="peer")
    assert r["confidence"] == pytest.approx((1 + 2 * 0.6) / 3, abs=1e-3)


def test_ann_mode_matches_exact_on_small_store(svc, run_id):
    eid = svc.store_experience(_create(run_id))["id"]
    svc.store_experience(_create(run_id, task="Bake sourdough bread", technologies=["oven"], task_key="bread"))
    for ann in (0, 50):
        res = svc.prepare_context("Configure pgvector HNSW index for cosine search", agent_id="peer",
                                  role=AgentRole.CODING, technologies=["postgres", "pgvector"],
                                  scope_run_id=run_id, policy=MemorySelectionPolicy(ann_candidates=ann))
        assert res["injected_experience_ids"] == [str(eid)]


def test_ann_mode_falls_back_when_neighbours_are_invisible(run_id):
    """Index neighbours all belong to someone else -> exact fallback still finds the visible one."""
    from actioncloud import db
    from actioncloud.embeddings import experience_embedding_text, get_embedding_provider

    svc = MemoryService(sync_write=True)
    mine = svc.store_experience(_create(run_id, task="Rotate the RDS master password with Secrets Manager",
                                        technologies=["aws", "rds"], task_key="rds"))["id"]
    vec = get_embedding_provider().embed(experience_embedding_text(
        "Rotate the RDS master password with Secrets Manager", ["aws", "rds"]))
    rows = db.hybrid_search_experiences("Rotate the RDS master password with Secrets Manager", vec,
                                        limit=5, agent_id="author", agent_role="coding",
                                        ann_candidates=1)   # top-1 neighbour may be anyone's
    assert str(mine) in {str(r["id"]) for r in rows}
