#!/usr/bin/env python3
"""Builds every data figure in the paper from ../data/*.json and the repo's results/*.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIG = HERE.parent / "figures"
REPO_RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "results"

# Validated categorical palette (fixed order) + chart chrome.
AC, FLAT, BASE, EXTRA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ARMS = [("baseline", "No memory", BASE, "s"), ("flat_memory", "Flat shared memory", FLAT, "^"),
        ("actioncloud", "ActionCloud", AC, "o")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7.5,
    "lines.linewidth": 1.6, "lines.markersize": 5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
COL1, COL2 = 3.5, 7.2   # single / double column width (inches) for elsarticle 5p


def load(name):
    return json.loads((DATA / name).read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def bars(ax, vals, errs, ylabel, fmt):
    xs = range(len(ARMS))
    for x, (key, label, color, _), v, e in zip(xs, ARMS, vals, errs):
        ax.bar(x, v, 0.62, color=color, yerr=e, capsize=2.5, error_kw={"elinewidth": 0.7, "ecolor": INK2},
               edgecolor="white", linewidth=0)
        ax.text(x, v + (e or 0) + max(vals) * 0.03, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(list(xs), ["No\nmemory", "Flat\nshared", "Action-\nCloud"])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="x", visible=False)


def fig_main():
    s = load("main_paired.json")["summary"]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.1))
    specs = [("success_rate_pct", "Task success (%)", "{:.1f}"),
             ("poisoned_injection_pct", "Flawed injections (%)", "{:.1f}"),
             ("avg_tokens_per_task", "Tokens per task", "{:.0f}")]
    for ax, (k, lab, fmt) in zip(axs, specs):
        bars(ax, [s[a][k]["mean"] for a, *_ in ARMS], [s[a][k]["sd"] for a, *_ in ARMS], lab, fmt)
    fig.tight_layout(w_pad=2.2)
    save(fig, "main_results")


def fig_learning():
    c = load("main_paired.json")["learning_curve"]
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for key, label, color, m in ARMS:
        ys = [c[key][str(e)] for e in range(3)]
        ax.plot([1, 2, 3], ys, color=color, marker=m, label=label)
        ax.annotate(f"{ys[-1]:.1f}", (3, ys[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.8, 3.45)
    ax.set_xlabel("Epoch (pass over the 136-task workload)")
    ax.set_ylabel("Task success (%)")
    ax.set_ylim(55, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, handlelength=1.5, columnspacing=1.0)
    save(fig, "learning_curve")


def fig_ablations():
    k = json.loads((REPO_RESULTS / "k.json").read_text())["summary"]
    fb = json.loads((REPO_RESULTS / "feedback.json").read_text())["summary"]
    fig, axs = plt.subplots(1, 3, figsize=(COL2 - 0.1, 2.15))
    ks = [1, 2, 3, 5]
    ax = axs[0]
    ax.plot(ks, [k[f"k{i}"]["retrieval_precision_pct"]["mean"] for i in ks], color=AC, marker="o", label="Retrieval precision")
    ax.plot(ks, [k[f"k{i}"]["success_rate_pct"]["mean"] for i in ks], color=EXTRA, marker="D", label="Task success")
    ax.set_xlabel("Memories injected per task (k)")
    ax.set_ylabel("%")
    ax.set_ylim(50, 100)
    ax.set_xticks(ks)
    ax.text(3.0, 89.5, "Task success", color=INK, fontsize=7.5, ha="center")
    ax.text(3.2, 63.0, "Retrieval precision", color=INK, fontsize=7.5, ha="left")
    ax = axs[1]
    ax.plot(ks, [k[f"k{i}"]["avg_context_tokens"]["mean"] for i in ks], color=AC, marker="o")
    for i in ks:
        v = k[f"k{i}"]["avg_context_tokens"]["mean"]
        ax.annotate(f"{v:.0f}", (i, v), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xlabel("Memories injected per task (k)")
    ax.set_ylabel("Context tokens per task")
    ax.set_ylim(90, 180)
    ax.set_xticks(ks)
    ax = axs[2]
    rates = [30, 70, 100]
    flat_p = fb["flat_memory"]["poisoned_injection_pct"]["mean"]
    ax.plot(rates, [fb[f"feedback_{r}"]["poisoned_injection_pct"]["mean"] for r in rates], color=AC, marker="o",
            label="ActionCloud")
    ax.axhline(flat_p, color=FLAT, linestyle="--", linewidth=1.2, label="Flat shared memory")
    ax.set_xlabel("Outcomes observed (%)")
    ax.set_ylabel("Flawed injections (%)")
    ax.set_xticks(rates)
    ax.set_xlim(20, 110)
    ax.set_ylim(-1, 40)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout(w_pad=2.0)
    save(fig, "ablations")


def fig_adversary():
    rows = load("adversary.json")
    rates = [100 * r["adversary_rate"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.2))
    for ax, key, lab in ((axs[0], "success_rate_pct", "Honest-agent success (%)"),
                         (axs[1], "poisoned_injection_pct", "Flawed injections (%)")):
        for arm, label, color, m in ARMS + [("actioncloud_author", "ActionCloud + author rep.", EXTRA, "D")]:
            if key == "poisoned_injection_pct" and arm == "baseline":
                continue
            ys = [r[arm][key]["mean"] for r in rows]
            es = [r[arm][key]["sd"] for r in rows]
            ax.errorbar(rates, ys, yerr=es, color=color, marker=m, label=label, capsize=2, elinewidth=0.7)
        ax.set_xlabel("Adversarial agents (%)")
        ax.set_ylabel(lab)
        ax.set_xticks(rates)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=7.5)
    fig.tight_layout(w_pad=2.5)
    save(fig, "adversary")


def fig_sensitivity():
    d = load("sensitivity.json")
    grid = d["grid"]
    oks = sorted({g["p_scratch_success"] for g in grid})
    mis = sorted({g["p_misled"] for g in grid})
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.35), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    import numpy as np
    m = np.zeros((len(mis), len(oks)))
    for g in grid:
        m[mis.index(g["p_misled"]), oks.index(g["p_scratch_success"])] = (
            g["actioncloud"]["success_rate_pct"] - g["flat_memory"]["success_rate_pct"])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=max(1, m.max()), origin="lower", aspect="auto")
    for i in range(len(mis)):
        for j in range(len(oks)):
            ax.text(j, i, f"+{m[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.55 * m.max() else INK)
    ax.set_xticks(range(len(oks)), [f"{v:.2f}" for v in oks])
    ax.set_yticks(range(len(mis)), [f"{v:.2f}" for v in mis])
    ax.set_xlabel(r"Unaided success probability $p_{\mathrm{solve}}$")
    ax.set_ylabel(r"Misled probability $p_{\mathrm{mislead}}$")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Success gain over flat (pp)", fontsize=7.5)
    cb.outline.set_visible(False)
    ax = axs[1]
    t = d["token_gap"]
    xs = [100 * r["gap_fraction"] for r in t]
    ax.plot(xs, [r["savings_pct"] for r in t], color=AC, marker="o")
    for x, r in zip(xs, t):
        ax.annotate(f"{r['savings_pct']:.1f}%", (x, r["savings_pct"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7.5)
    ax.set_xlabel("Assumed output-token gap kept (%)")
    ax.set_ylabel("Token savings (%)")
    ax.set_xlim(108, 25)
    ax.set_ylim(min(0, min(r["savings_pct"] for r in t) - 3), max(r["savings_pct"] for r in t) + 8)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    fig.tight_layout(w_pad=2.5)
    save(fig, "sensitivity")


def fig_calibration():
    c = load("calibration.json")
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.15))
    ax = axs[0]
    bins = [i * 0.025 for i in range(-4, 34)]
    ax.hist(c["diff"], bins=bins, density=True, color=MUTED, alpha=0.55, label="Different family")
    ax.hist(c["same"], bins=bins, density=True, color=AC, alpha=0.75, label="Same family")
    ax.axvline(0.30, color=INK, linestyle="--", linewidth=0.9)
    ax.text(0.31, ax.get_ylim()[1] * 0.55, r"$\tau=0.30$", fontsize=7.5)
    ax.set_xlabel("Cosine similarity (query vs. stored memory)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(-0.12, 0.85)
    ax = axs[1]
    cur = c["curve"]
    ax.plot([p["t"] for p in cur], [100 * p["recall"] for p in cur], color=AC, marker="o", markersize=3.5,
            label="Recall (same family)")
    ax.plot([p["t"] for p in cur], [100 * p["fpr"] for p in cur], color=FLAT, marker="^", markersize=3.5,
            label="False-positive rate")
    ax.axvline(0.30, color=INK, linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Relevance threshold $\tau$")
    ax.set_ylabel("%")
    ax.set_xlim(0, 0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "calibration")


def fig_scaling():
    d = load("microbench.json")["scaling"]
    fig, ax = plt.subplots(figsize=(COL1, 2.25))
    ns = [r["n"] for r in d]
    ax.plot(ns, [r["context_p50_ms"] for r in d], color=FLAT, marker="^", label="Exact scoring, p50")
    ax.plot(ns, [r["context_p95_ms"] for r in d], color=FLAT, marker="^", linestyle="--", markerfacecolor="white",
            label="Exact scoring, p95")
    ax.plot(ns, [r["ann_p50_ms"] for r in d], color=AC, marker="o", label="HNSW pre-selection, p50")
    ax.plot(ns, [r["ann_p95_ms"] for r in d], color=AC, marker="o", linestyle="--", markerfacecolor="white",
            label="HNSW pre-selection, p95")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Memories visible to the query")
    ax.set_ylabel("Context build latency (ms)")
    ax.set_ylim(8, 4000)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=6.8)
    save(fig, "scaling")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (fig_main, fig_learning, fig_ablations, fig_calibration, fig_adversary, fig_sensitivity, fig_scaling):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, "(missing", Path(e.filename).name + ")")
