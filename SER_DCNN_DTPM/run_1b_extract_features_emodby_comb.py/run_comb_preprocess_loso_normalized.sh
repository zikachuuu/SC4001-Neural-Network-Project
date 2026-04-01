#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
python 1b_extract_features_emodb_comb.py \
  --split-mode loso \
  --normalize-speaker \
  --output-dir ./processed_emodb_comb_loso_normalized
