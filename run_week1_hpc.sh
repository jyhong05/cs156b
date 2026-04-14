#!/usr/bin/env bash
set -euo pipefail

# Minimal Week 1 runner for HPC.
# Usage:
#   bash run_week1_hpc.sh
# Optional custom path usage:
#   bash run_week1_hpc.sh /path/to/train2023.csv /path/to/test_ids.csv

TRAIN_CSV="${1:-/groups/CS156b/from_central/data/train/student_labels/train2023.csv}"
TEST_CSV="${2:-/groups/CS156b/from_central/data/student_labels/test_ids.csv}"

echo "Host: $(hostname)"
echo "CWD:  $(pwd)"

echo "Checking dataset files..."
ls -l "$TRAIN_CSV" "$TEST_CSV"

echo "Running Week 1 data exploration..."
python3 code.py \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --sample-images 200 \
  --uncertain-policy ignore \
  --nan-policy ignore

echo "Done. Generated:"
ls -l week1_report.txt week1_raw_label_stats.csv week1_policy_stats.csv
