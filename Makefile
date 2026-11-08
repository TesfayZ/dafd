# One-command reproduction:  make reproduce   (experiments -> tables -> figures -> claim check -> paper)
#
#   make quick      reduced smoke run of every benchmark -> Results/quick/ (a few minutes, one core)
#   make all        full run of every benchmark -> Results/ (historical measured run: 834 s; see README); always reruns everything
#   make resume     continue an interrupted full run (skips benchmarks that finished; markers in Results/logs/)
#   make tables     LaTeX tables generated from Results/*.csv -> doc/tables/
#   make figures    figures generated from Results/*.csv -> doc/fig/
#   make check      assert that every headline number quoted in the paper matches Results/*.csv
#   make paper      tables + figures + claim check + doc/DAFD-paper.pdf
#   make reproduce  all + paper
#   make source-audit  fetch/build/audit the pinned upstream engine (network/CMake)
#   make reproduce-complete  all + source-audit + paper
#   make predecessor  validate a supplied prior-work checkout (needed by 3 benchmarks)
#
# Keep JOBS=1: the sensitivity study re-scores trial files written by two earlier benchmarks.
PY   ?= python3
JOBS ?= 1

# Reproduction stages consume files produced by earlier stages, even with make -j.
.NOTPARALLEL:

quick:
	$(PY) run_all_experiments.py --quick --jobs $(JOBS)

all:
	rm -rf Results/logs
	$(PY) run_all_experiments.py --jobs $(JOBS)

resume:
	$(PY) run_all_experiments.py --resume --jobs $(JOBS)

tables:
	$(PY) make_tables.py

figures:
	$(PY) make_figures.py

check:
	$(PY) check_claims.py

# latexmk reruns pdflatex until the .aux/.out files converge; -halt-on-error stops on the first real error;
# latexmk -c removes auxiliary files afterwards so the next build starts clean.
paper: tables figures check
	cd doc && latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error DAFD-paper.tex
	cd doc && latexmk -c

reproduce: all paper

source-audit:
	$(PY) reproduce_mujoco_source.py

reproduce-complete: all source-audit paper

watch:
	cd doc && latexmk -pdf -pvc -interaction=nonstopmode DAFD-paper.tex

predecessor:
	$(PY) predecessor.py

.PHONY: quick all resume tables figures check paper reproduce source-audit reproduce-complete watch predecessor
