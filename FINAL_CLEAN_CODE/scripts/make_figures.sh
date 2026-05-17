#!/bin/bash

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p plots report/figures

echo "=== Regenerating available figures ==="

if [ -f results/final_main_results.csv ] || [ -f results/roofline_combined.csv ]; then
    python3 scripts/generate_results.py
fi

if [ -f results/roofline_combined.csv ]; then
    python3 scripts/roofline_combined_analysis.py
fi

echo "Figures are in plots/. Copy final selected assets into report/figures/ when locking the report."
