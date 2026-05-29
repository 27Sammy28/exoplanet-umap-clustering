#!/bin/bash

set -e

export PYTHONPATH=.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

LOG_DIR="results/logs"
REPORT_DIR="results/reports"

mkdir -p $LOG_DIR $REPORT_DIR

LOG_FILE="$LOG_DIR/run_$TIMESTAMP.log"

echo "==============================" | tee -a $LOG_FILE
echo "NeurIPS Pipeline Started" | tee -a $LOG_FILE
echo "Timestamp: $TIMESTAMP" | tee -a $LOG_FILE
echo "==============================" | tee -a $LOG_FILE

echo "[1/4] Running experiment..." | tee -a $LOG_FILE
python scripts/run_experiment.py >> $LOG_FILE 2>&1

echo "[2/4] Generating table..." | tee -a $LOG_FILE
python scripts/make_table.py >> $LOG_FILE 2>&1

echo "[3/4] Generating figures..." | tee -a $LOG_FILE
python scripts/make_neurips_figures.py >> $LOG_FILE 2>&1

echo "[4/4] Creating reproducibility report..." | tee -a $LOG_FILE
python scripts/make_report.py >> $LOG_FILE 2>&1 || echo "Report failed" | tee -a $LOG_FILE

echo "==============================" | tee -a $LOG_FILE
echo "Pipeline Completed Successfully" | tee -a $LOG_FILE
echo "Log saved to: $LOG_FILE"
echo "=============================="
