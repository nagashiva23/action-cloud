# ActionCloud paper — build everything from source.
#   make            compile main.pdf (needs TeX Live with elsarticle)
#   make figures    re-export the draw.io diagrams and re-plot the data figures
#   make clean
#
# draw.io desktop CLI. On macOS:
#   make figures DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io
# Headless Linux (as root / in CI):
#   xvfb-run -a make figures DRAWIO_FLAGS="--no-sandbox --disable-gpu"
REPO ?= ..
DRAWIO ?= drawio
DRAWIO_FLAGS ?=
PYTHON ?= python3

.PHONY: all figures diagrams plots clean

all: main.pdf

main.pdf: main.tex refs.bib $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures: diagrams plots

diagrams:
	cd diagrams && $(PYTHON) build_diagrams.py
	for f in diagrams/*.drawio; do \
	  $(DRAWIO) -x -f pdf --crop -o figures/$$(basename $${f%.drawio}).pdf $$f $(DRAWIO_FLAGS) || exit 1; \
	done

plots:
	$(PYTHON) scripts/make_plots.py $(REPO)/results

clean:
	latexmk -C
