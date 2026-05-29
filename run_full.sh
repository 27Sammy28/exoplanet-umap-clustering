#!/bin/bash

set -e
export PYTHONPATH=.

echo "Running full NeurIPS submission pipeline..."

python scripts/run_experiment.py
python scripts/make_table.py
python scripts/make_neurips_figures.py
python scripts/make_report.py
python scripts/build_paper.py
python scripts/make_zip.py

echo "DONE: submission_package.zip ready"
