#!/usr/bin/env python3
"""
Experiments for the ActionCloud paper. Every number in the paper comes from
the JSON files this script writes to ../data/.

  PYTHONPATH=../../src python paper_experiments.py main        # paired 5-seed run, per-task outcomes
  PYTHONPATH=../../src python paper_experiments.py sensitivity # simulator-assumption grid
  PYTHONPATH=../../src python paper_experiments.py adversary   # poisoning / collusion sweep
  PYTHONPATH=../../src python paper_experiments.py calibration # retrieval score distributions
  DB_NAME=ac_bench PYTHONPATH=../../src python paper_experiments.py microbench  # real latency/scale

The agent loop mirrors actioncloud.benchmark.harness.run_arm exactly (same
agents, verifier, policy, seeds); it additionally records per-instance
outcomes and supports adversarial agents.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from actioncloud import db
from actioncloud.agents import TaskSpec, build_agent
from actioncloud.benchmark import load_procedures, load_tasks
from actioncloud.benchmark.harness import ACTIONCLOUD, BASELINE, FLAT, ArmConfig, Workload
from actioncloud.benchmark.simulator import SimParams, SimulatedAgentLLM
from actioncloud.benchmark.verifier import TaskVerifier
from actioncloud.client import InProcessClient
from actioncloud.metrics import compute_metrics
from actioncloud.schema import AgentRole
from actioncloud.service import MemoryService

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# Adversarial agents
# --------------------------------------------------------------------------

class AdversaryClient:
    """
    Wraps the in-process client for an adversarial agent that
      * stores a poisoned procedure (ground truth minus two steps) as a success,
      * reports every memory it was given as a success (collusion / lying),
      * never admits its own memory failed.
    """

    def __init__(self, inner: InProcessClient, procedures: dict, own_ids: set) -> None:
        self.inner = inner
        self.procedures = procedures
        self.own_ids = own_ids

    def context(self, **kw):
        return self.inner.context(**kw)

    def store(self, **fields):
        steps = [s["step"] for s in self.procedures[fields["task_key"]]][:-2]
        text = "Procedure applied:\n" + "\n".join(f"STEP: {s}" for s in steps)
        fields.update(action=text, solution=text, success=True, result="Completed")
        exp_id = self.inner.store(**fields)
        self.own_ids.add(str(exp_id))
        return exp_id

    def report_reuse(self, experience_id, success, agent_id=None):
        if str(experience_id) in self.own_ids:
            return {}                                   # never self-incriminate
        return self.inner.report_reuse(experience_id, success=True, agent_id=agent_id)


def run_arm_detailed(arm: ArmConfig, wl: Workload, sim: SimParams, exp_id: str,
                     adversary_rate: float = 0.0, author_prior: bool = False) -> dict[str, Any]:
    # Unique per call: sweeps reuse (arm, seed) across settings, and settings
    # must never read each other's memories.
    run_id = f"{exp_id}-{arm.name}-s{wl.seed}-{uuid.uuid4().hex[:8]}"
    service = MemoryService(sync_write=True, governance=arm.governance, author_prior=author_prior)
    base_client = InProcessClient(service=service)
    llm = SimulatedAgentLLM(seed=wl.seed, params=sim)
    verifier = TaskVerifier()
    procedures = load_procedures()
    policy = arm.policy()
    n_adv = round(adversary_rate * wl.agents_per_role)
    adversary_owned: set[str] = set()
    author_is_adversary: dict[str, bool] = {}
    memory_correct: dict[str, bool] = {}

    rows = []
    for i, t in enumerate(wl.instances()):
        role = AgentRole(t["role"])
        k = i % wl.agents_per_role
        is_adv = k < n_adv
        client = AdversaryClient(base_client, procedures, adversary_owned) if is_adv else base_client
        agent = build_agent(
            # run-unique ids: author reputation is global per agent id
            role=role, agent_id=f"{role.value}-{arm.name}-{k}-{run_id[-8:]}", client=client, run_id=run_id,
            use_memory=arm.use_memory, llm=llm, policy=policy, verifier=verifier,
            report_reuse=arm.governance, feedback_rate=arm.feedback_rate, scope_run_id=run_id,
        )
        o = agent.run_task(TaskSpec(task=t["task"], task_key=t["task_key"],
                                    technologies=t["technologies"],
                                    expected_tools=t.get("expected_tools", []),
                                    instance_id=t["instance_id"]))
        poisoned = sum(
            1 for e, key in zip(o.retrieved_ids, o.injected_task_keys)
            if key == t["task_key"] and (not memory_correct.get(str(e), True)
                                          or author_is_adversary.get(str(e), False))
        )
        relevant = sum(1 for key in o.injected_task_keys if key == t["task_key"])
        if o.experience_id is not None:
            eid = str(o.experience_id)
            author_is_adversary[eid] = is_adv
            memory_correct[eid] = (not is_adv) and o.success
        rows.append({
            "instance_id": t["instance_id"], "task_key": o.task_key, "role": role.value,
            "adversary": is_adv, "success": o.success,
            "tokens_input": o.tokens_input, "tokens_output": o.tokens_output,
            "execution_time_ms": o.execution_time_ms, "cost_usd": o.cost_usd,
            "injected_task_keys": o.injected_task_keys, "context_tokens": o.context_tokens,
            "relevant_injections": relevant, "poisoned_injections": poisoned, "mode": o.mode,
        })

    honest = [r for r in rows if not r["adversary"]]
    m = compute_metrics(honest)
    rel = sum(r["relevant_injections"] for r in honest)
    m["poisoned_injection_pct"] = round(100 * sum(r["poisoned_injections"] for r in honest) / rel, 2) if rel else 0.0
    m["avg_context_tokens"] = round(statistics.mean(r["context_tokens"] for r in honest), 1)
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT tier::text AS tier, count(*) AS n FROM experiences WHERE run_id=%s GROUP BY tier", (run_id,))
        m["tiers"] = {r["tier"]: r["n"] for r in cur.fetchall()}
    return {"arm": arm.name + ("+author" if author_prior else ""), "seed": wl.seed, "adversary_rate": adversary_rate,
            "sim": asdict(sim), "metrics": m, "rows": rows}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(rows_a: list[dict], rows_b: list[dict]) -> dict:
    a = {r["instance_id"]: r["success"] for r in rows_a}
    b = {r["instance_id"]: r["success"] for r in rows_b}
    both = [k for k in a if k in b]
    only_a = sum(1 for k in both if a[k] and not b[k])
    only_b = sum(1 for k in both if b[k] and not a[k])
    return {"n": len(both), "a_only": only_a, "b_only": only_b, "p": mcnemar_exact(only_a, only_b)}


def bootstrap_ci(values: list[float], iters: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def exp_main(seeds=(1, 2, 3, 4, 5)):
    exp = f"paper-main-{uuid.uuid4().hex[:6]}"
    arms = [BASELINE, FLAT, ACTIONCLOUD]
    runs = {a.name: [] for a in arms}
    for s in seeds:
        for a in arms:
            r = run_arm_detailed(a, Workload(epochs=3, seed=s), SimParams(), exp)
            runs[a.name].append(r)
            log(f"main seed={s} {a.name}: success={r['metrics']['success_rate_pct']}")
    tests = {}
    for x, y in [("actioncloud", "baseline"), ("actioncloud", "flat_memory"), ("flat_memory", "baseline")]:
        rows_x = [row for r in runs[x] for row in r["rows"]]
        rows_y = [row for r in runs[y] for row in r["rows"]]
        # instance ids repeat across seeds; make them unique per seed
        def tag(rs, runs_):
            out = []
            for r in runs_:
                out += [{**row, "instance_id": f"s{r['seed']}:{row['instance_id']}"} for row in r["rows"]]
            return out
        tests[f"{x}_vs_{y}"] = paired(tag(None, runs[x]), tag(None, runs[y]))
    # per-seed paired token savings with bootstrap CI over task instances
    savings = []
    for rb, ra in zip(runs["baseline"], runs["actioncloud"]):
        tb = {row["instance_id"]: row["tokens_input"] + row["tokens_output"] for row in rb["rows"]}
        for row in ra["rows"]:
            b = tb[row["instance_id"]]
            savings.append(100 * (b - row["tokens_input"] - row["tokens_output"]) / b)
    # learning curve: success by epoch
    curve = {}
    for name, rs in runs.items():
        per_epoch = {}
        for r in rs:
            for row in r["rows"]:
                ep = int(row["instance_id"].split(":")[1])
                per_epoch.setdefault(ep, []).append(row["success"])
        curve[name] = {ep: round(100 * sum(v) / len(v), 2) for ep, v in sorted(per_epoch.items())}
    # per-role success
    by_role = {}
    for name in ("baseline", "actioncloud"):
        acc = {}
        for r in runs[name]:
            for row in r["rows"]:
                acc.setdefault(row["role"], []).append(row["success"])
        by_role[name] = {k: round(100 * sum(v) / len(v), 2) for k, v in sorted(acc.items())}
    summary = {
        name: {k: {"mean": round(statistics.mean(r["metrics"][k] for r in rs), 3),
                   "sd": round(statistics.stdev(r["metrics"][k] for r in rs), 3)}
               for k in ("success_rate_pct", "retrieval_precision_pct", "misled_rate_pct",
                         "poisoned_injection_pct", "redundancy_index", "avg_tokens_per_task",
                         "avg_context_tokens", "cost_per_success_usd", "injection_rate_pct")
               if all(r["metrics"].get(k) is not None for r in rs)}
        for name, rs in runs.items()
    }
    lo, hi = bootstrap_ci(savings)
    out = {"summary": summary, "tests": tests,
           "token_savings_pct": {"mean": round(statistics.mean(savings), 2), "ci95": [round(lo, 2), round(hi, 2)]},
           "learning_curve": curve, "by_role": by_role,
           "tiers": {n: [r["metrics"]["tiers"] for r in rs] for n, rs in runs.items()}}
    (DATA / "main_paired.json").write_text(json.dumps(out, indent=1))
    log(json.dumps({"tests": tests, "token_savings": out["token_savings_pct"]}, indent=1))


def exp_author_benign(seeds=(1, 2, 3, 4, 5)):
    """Does the author prior cost anything when every agent is honest?"""
    exp = f"paper-authb-{uuid.uuid4().hex[:6]}"
    out = {}
    for ap in (False, True):
        vals = [run_arm_detailed(ACTIONCLOUD, Workload(epochs=3, seed=s), SimParams(), exp, author_prior=ap)["metrics"]
                for s in seeds]
        out["author" if ap else "memory_only"] = {
            k: {"mean": round(statistics.mean(v[k] for v in vals), 2), "sd": round(statistics.stdev(v[k] for v in vals), 2)}
            for k in ("success_rate_pct", "poisoned_injection_pct", "retrieval_precision_pct", "injection_rate_pct",
                      "avg_tokens_per_task")}
        log("benign", "author" if ap else "memory_only", out["author" if ap else "memory_only"]["success_rate_pct"])
    (DATA / "author_benign.json").write_text(json.dumps(out, indent=1))


def exp_sensitivity(seeds=(1, 2)):
    exp = f"paper-sens-{uuid.uuid4().hex[:6]}"
    grid = []
    for p_ok in (0.45, 0.65, 0.85):
        for p_mis in (0.5, 0.85, 1.0):
            sim = replace(SimParams(), p_scratch_success=p_ok, p_misled=p_mis)
            cell = {"p_scratch_success": p_ok, "p_misled": p_mis}
            for a in (BASELINE, FLAT, ACTIONCLOUD):
                vals = [run_arm_detailed(a, Workload(epochs=3, seed=s), sim, exp)["metrics"] for s in seeds]
                cell[a.name] = {k: round(statistics.mean(v[k] for v in vals), 3)
                                for k in ("success_rate_pct", "poisoned_injection_pct", "avg_tokens_per_task")}
            log(f"sens p_ok={p_ok} p_mis={p_mis}: base={cell['baseline']['success_rate_pct']} "
                f"flat={cell['flat_memory']['success_rate_pct']} ac={cell['actioncloud']['success_rate_pct']}")
            grid.append(cell)
    # token-model sensitivity: shrink the assumed follow-vs-scratch output gap
    token = []
    for ratio in (1.0, 0.75, 0.5, 0.33):
        base = SimParams()
        sim = replace(base,
                      follow_out_base=int(base.scratch_out_base - (base.scratch_out_base - base.follow_out_base) * ratio),
                      follow_out_per_step=int(base.scratch_out_per_step - (base.scratch_out_per_step - base.follow_out_per_step) * ratio))
        r = {"gap_fraction": ratio}
        for a in (BASELINE, ACTIONCLOUD):
            vals = [run_arm_detailed(a, Workload(epochs=3, seed=s), sim, exp)["metrics"] for s in seeds]
            r[a.name] = round(statistics.mean(v["avg_tokens_per_task"] for v in vals), 1)
        r["savings_pct"] = round(100 * (r["baseline"] - r["actioncloud"]) / r["baseline"], 2)
        log(f"token gap={ratio}: savings={r['savings_pct']}")
        token.append(r)
    (DATA / "sensitivity.json").write_text(json.dumps({"grid": grid, "token_gap": token}, indent=1))


def exp_adversary(seeds=(1, 2, 3)):
    exp = f"paper-adv-{uuid.uuid4().hex[:6]}"
    res = []
    for rate in (0.0, 0.1, 0.2, 0.3, 0.4):
        cell = {"adversary_rate": rate}
        for a, ap in ((BASELINE, False), (FLAT, False), (ACTIONCLOUD, False), (ACTIONCLOUD, True)):
            vals = [run_arm_detailed(a, Workload(epochs=3, agents_per_role=10, seed=s), SimParams(), exp,
                                     adversary_rate=rate, author_prior=ap)["metrics"] for s in seeds]
            cell[a.name + ("_author" if ap else "")] = {k: {"mean": round(statistics.mean(v[k] for v in vals), 2),
                                "sd": round(statistics.stdev(v[k] for v in vals), 2)}
                            for k in ("success_rate_pct", "poisoned_injection_pct", "misled_rate_pct")}
        log(f"adv {rate}: base={cell['baseline']['success_rate_pct']['mean']} "
            f"flat={cell['flat_memory']['success_rate_pct']['mean']} ac={cell['actioncloud']['success_rate_pct']['mean']} "
            f"ac+author={cell['actioncloud_author']['success_rate_pct']['mean']} "
            f"| poison flat={cell['flat_memory']['poisoned_injection_pct']['mean']} ac={cell['actioncloud']['poisoned_injection_pct']['mean']} "
            f"ac+author={cell['actioncloud_author']['poisoned_injection_pct']['mean']}")
        res.append(cell)
    (DATA / "adversary.json").write_text(json.dumps(res, indent=1))


def exp_calibration():
    from actioncloud.embeddings import cosine, experience_embedding_text, get_embedding_provider
    tasks = list(load_tasks())
    emb = get_embedding_provider()
    vecs = [emb.embed(experience_embedding_text(t["task"], t["technologies"])) for t in tasks]
    q_bare = [emb.embed(t["task"]) for t in tasks]
    same, diff, same_bare, diff_bare = [], [], [], []
    top1 = top1_bare = 0
    for i, ti in enumerate(tasks):
        best = best_b = (-2.0, -1)
        for j, tj in enumerate(tasks):
            if i == j:
                continue
            c, cb = cosine(vecs[i], vecs[j]), cosine(q_bare[i], vecs[j])
            (same if ti["task_key"] == tj["task_key"] else diff).append(c)
            (same_bare if ti["task_key"] == tj["task_key"] else diff_bare).append(cb)
            best, best_b = max(best, (c, j)), max(best_b, (cb, j))
        top1 += tasks[best[1]]["task_key"] == ti["task_key"]
        top1_bare += tasks[best_b[1]]["task_key"] == ti["task_key"]
    ths = [round(0.05 * i, 2) for i in range(0, 17)]
    curve = [{"t": th, "recall": sum(c >= th for c in same) / len(same),
              "fpr": sum(c >= th for c in diff) / len(diff)} for th in ths]
    out = {"n_tasks": len(tasks), "n_families": len({t["task_key"] for t in tasks}),
           "top1": top1, "top1_task_only": top1_bare, "curve": curve,
           "same": same, "diff": random.Random(0).sample(diff, 3000)}
    (DATA / "calibration.json").write_text(json.dumps(out))
    log(f"top1 {top1}/{len(tasks)} (task-only query {top1_bare}/{len(tasks)})")


def exp_microbench():
    """Real latency/throughput on Postgres 16 + pgvector (run against an empty DB)."""
    from actioncloud.embeddings import experience_embedding_text, get_embedding_provider
    from actioncloud.judge import DECIDED_BY, MemoryJudge
    from actioncloud.pipeline import process_experience
    from actioncloud.policy import MemorySelectionPolicy
    from actioncloud.schema import Experience, MemoryTier, SystemCondition

    db.require_database()
    tasks = list(load_tasks())
    rng = random.Random(0)
    filler = ("cache retry timeout config schema index worker queue deploy build test alert "
              "metric token parser router client server auth session batch stream").split()
    emb = get_embedding_provider()
    svc = MemoryService(sync_write=True)
    pol = MemorySelectionPolicy()

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    # 1) ingest pipeline latency (full: tier + extract + embed + transactional insert)
    lat = []
    for i in range(300):
        t = rng.choice(tasks)
        exp = Experience(agent_id=f"bench-{i % 12}", agent_role=AgentRole(t["role"]),
                         task=t["task"] + " " + " ".join(rng.sample(filler, 3)),
                         action="STEP: a\nSTEP: b\nSTEP: c", solution="STEP: a\nSTEP: b\nSTEP: c",
                         result="ok", success=True, run_id="bench", system=SystemCondition.ACTIONCLOUD,
                         technologies=t["technologies"], task_key=t["task_key"])
        t0 = time.perf_counter()
        process_experience(exp)
        lat.append(1000 * (time.perf_counter() - t0))
    ingest = {"p50_ms": round(pct(lat, .5), 2), "p95_ms": round(pct(lat, .95), 2),
              "throughput_per_s": round(1000 / statistics.mean(lat), 1)}
    log("ingest", ingest)

    # embedding cost alone
    t0 = time.perf_counter()
    for t in tasks:
        emb.embed(experience_embedding_text(t["task"], t["technologies"]))
    embed_ms = 1000 * (time.perf_counter() - t0) / len(tasks)

    # 2) retrieval latency as the store grows (all rows SHARED = worst case: everything visible)
    sizes = [1000, 5000, 10000, 25000, 50000]
    scaling = []
    inserted = 300
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE experiences SET tier='shared'")
        conn.commit()
    for n in sizes:
        batch = []
        while inserted + len(batch) < n:
            t = rng.choice(tasks)
            text = t["task"] + " " + " ".join(rng.sample(filler, 4))
            v = emb.embed(experience_embedding_text(text, t["technologies"]))
            batch.append((str(uuid.uuid4()), t["role"], text, "STEP: a", "ok", t["technologies"], db._vec(v),
                          t["task_key"], json.dumps({"steps": ["a", "b", "c"]})))
        with db.get_conn() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO experiences (id, agent_id, agent_role, task, action, result, success,
                       technologies, run_id, system, tier, confidence, embedding, embedded, task_key, workflow)
                   VALUES (%s, 'bench', %s, %s, %s, %s, true, %s, 'bench', 'actioncloud', 'shared', 0.6,
                           %s::vector, true, %s, %s::jsonb)""", batch)
            conn.commit()
            cur.execute("ANALYZE experiences")
            conn.commit()
        inserted += len(batch)
        qs = [rng.choice(tasks) for _ in range(60)]
        pol_ann = MemorySelectionPolicy(ann_candidates=200)
        exact_lat, ann_lat, agree = [], [], 0
        for t in qs:
            kw = dict(agent_id="reader", role=AgentRole(t["role"]), technologies=t["technologies"])
            t0 = time.perf_counter()
            ex = svc.prepare_context(t["task"], policy=pol, **kw)
            exact_lat.append(1000 * (time.perf_counter() - t0))
            t0 = time.perf_counter()
            an = svc.prepare_context(t["task"], policy=pol_ann, **kw)
            ann_lat.append(1000 * (time.perf_counter() - t0))
            agree += ex["injected_experience_ids"] == an["injected_experience_ids"]
        row = {"n": n, "context_p50_ms": round(pct(exact_lat, .5), 2), "context_p95_ms": round(pct(exact_lat, .95), 2),
               "ann_p50_ms": round(pct(ann_lat, .5), 2), "ann_p95_ms": round(pct(ann_lat, .95), 2),
               "ann_agreement_pct": round(100 * agree / len(qs), 1)}
        log("scale", row)
        scaling.append(row)

    # 3) outcome-report latency (row-locked read-decide-write + injection ledger)
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM experiences WHERE agent_id LIKE 'bench-%%' LIMIT 200")
        ids = [r["id"] for r in cur.fetchall()]
    rep = []
    for e in ids:
        db.record_injections([e], "reporter")
        t0 = time.perf_counter()
        svc.record_reuse(e, success=True, agent_id="reporter")
        rep.append(1000 * (time.perf_counter() - t0))
    report = {"p50_ms": round(pct(rep, .5), 2), "p95_ms": round(pct(rep, .95), 2)}
    log("report", report)

    import platform
    import subprocess
    cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | cut -d: -f2"], capture_output=True, text=True).stdout.strip()
    out = {"hardware": {"cpu": cpu, "vcpus": __import__("os").cpu_count(),
                        "python": platform.python_version(), "postgres": "16 + pgvector"},
           "embed_ms": round(embed_ms, 2), "ingest": ingest, "scaling": scaling, "report": report}
    (DATA / "microbench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"main": exp_main, "sensitivity": exp_sensitivity, "adversary": exp_adversary, "author_benign": exp_author_benign,
     "calibration": exp_calibration, "microbench": exp_microbench}[sys.argv[1]]()
