#!/bin/bash

set -e
export PYTHONPATH=.

echo "🚀 Starting NeurIPS Submission Pipeline..."

# 1. Run experiments
python scripts/run_experiment.py

# 2. Build evaluation table
python scripts/make_table.py

# 3. Generate figures
python scripts/make_neurips_figures.py

# 4. Create reproducibility report
python scripts/make_report.py

echo "✅ Core pipeline complete"
