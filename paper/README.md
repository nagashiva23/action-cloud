# ActionCloud: paper

Manuscript for *Future Generation Computer Systems* (Elsevier, `elsarticle`, 5p two-column).

```
main.tex          manuscript
refs.bib          bibliography (29 entries, each checked against arXiv / the publisher)
highlights.txt    Elsevier highlights (5 bullets, ≤ 85 characters each)
figures/          every figure used by main.tex (PDF)
diagrams/         draw.io sources for Figures 1, 2, 3 and 5 + the script that generates them
scripts/          paper_experiments.py (all new experiments) and make_plots.py (all data figures)
data/             raw JSON results behind every number and plot in the paper
```

## Build

```bash
make                      # main.pdf, needs TeX Live with elsarticle (e.g. MacTeX, or texlive-publishers)
```

Overleaf works too: upload `main.tex`, `refs.bib` and `figures/`, then compile with pdfLaTeX.

## Regenerate figures

```bash
make figures DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io   # macOS with draw.io desktop
```

The architecture, lifecycle, governance and pipeline diagrams are ordinary `.drawio` files: open them in draw.io, edit, and re-export (File → Export as → PDF, "Crop" ticked), or run `make diagrams`. `diagrams/build_diagrams.py` regenerates the originals and overwrites manual edits, so once you start editing in draw.io, stop running it.

## Reproduce the numbers

The main, ablation and fleet results come from `../results/` (see the repository README). The additional experiments are in `scripts/paper_experiments.py`:

```bash
cd scripts
PYTHONPATH=../../src python paper_experiments.py main           # paired 5-seed run + McNemar tests (~3 min)
PYTHONPATH=../../src python paper_experiments.py calibration    # retrieval score distributions (no DB needed)
PYTHONPATH=../../src python paper_experiments.py adversary      # adversarial / collusion sweep (~10 min)
PYTHONPATH=../../src python paper_experiments.py author_benign  # cost of author reputation, honest fleet
PYTHONPATH=../../src python paper_experiments.py sensitivity    # simulator-assumption grid (~10 min)
DB_NAME=ac_bench PYTHONPATH=../../src python paper_experiments.py microbench   # latency; run on an EMPTY database
python make_plots.py
```

Run the simulation experiments against a database used only for them. Retrieval in exact mode scans every row, so a large shared table makes runs slow, although results stay correct because every run is isolated by its run ID. The latency benchmark must run on an empty database on an otherwise idle machine. The numbers in the paper came from a 2-vCPU Intel Xeon 2.8 GHz VM with PostgreSQL 16 and pgvector 0.6.

## Before submission

- [ ] Replace the `[Author n]`, affiliation, e-mail and CRediT placeholders.
- [ ] Complete the **generative-AI declaration** at the end of `main.tex`. Elsevier requires it: name the tool and what it was used for.
- [ ] Rewrite the text in your own words and check every claim yourself; you are responsible for all of it.
- [ ] The main results are from a **simulated** agent environment (Section 6.2). The paper says so throughout. A real-LLM run with the same harness (`LLM_PROVIDER=anthropic|gemini|groq`) would considerably strengthen it.
- [ ] Check FGCS's current Guide for Authors for length and format; switch to `\documentclass[preprint,12pt]{elsarticle}` if the editor asks for the review layout.
