#!/bin/bash

export PYTHONPATH=.

python scripts/run_experiment.py
python scripts/make_table.py
python scripts/make_neurips_figures.py


