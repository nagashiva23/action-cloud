#!/usr/bin/env python3
"""
Calibrate MemorySelectionPolicy.similarity_threshold for the current
EMBEDDING_PROVIDER on the benchmark tasks.

For every ordered pair of tasks it embeds the query side (task + technologies,
what agents send) and the memory side (what the pipeline stores), then reports
top-1 precision and, per threshold, recall on same-family pairs vs the
false-positive rate on different-family pairs. No database needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from actioncloud.benchmark import load_tasks  # noqa: E402
from actioncloud.embeddings import cosine, experience_embedding_text, get_embedding_provider  # noqa: E402

tasks = list(load_tasks())
emb = get_embedding_provider()
vecs = [emb.embed(experience_embedding_text(t["task"], t["technologies"])) for t in tasks]

same, diff, top1 = [], [], 0
for i, ti in enumerate(tasks):
    best = (-2.0, -1)
    for j, tj in enumerate(tasks):
        if i == j:
            continue
        c = cosine(vecs[i], vecs[j])
        (same if ti["task_key"] == tj["task_key"] else diff).append(c)
        best = max(best, (c, j))
    top1 += tasks[best[1]]["task_key"] == ti["task_key"]

print(f"provider={emb.name}  tasks={len(tasks)}  top-1 precision={top1}/{len(tasks)} ({100*top1/len(tasks):.1f}%)")
print(f"{'threshold':>9} {'recall(same)':>13} {'FP rate(diff)':>14}")
for th in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
    r = sum(c >= th for c in same) / len(same)
    fp = sum(c >= th for c in diff) / len(diff)
    print(f"{th:>9.2f} {r:>13.2%} {fp:>14.3%}")
