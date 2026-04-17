#!/usr/bin/env bash
set -euo pipefail

# Run from SER_DCNN_DTPM root, or this script auto-jumps there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Activate env if needed:
# conda activate sc4001-project

# 1) Build combined audio first (if not already generated).
# python 1a_generate_dynamic_emodb_combinations.py \
#   --data-dir ../emo_db \
#   --output-dir ../emo_db_comb

# 2) Soft-label extraction with original split.
# python B2b_extract_features_emodb_comb_soft_labels.py \
#   --data-dir ../emo_db_comb \
#   --output-dir ./processed_emodb_comb_soft \
#   --soft-label-decimals 3

# 3) Soft-label extraction with LOSO split.
python B2b_extract_features_emodb_comb_soft_labels.py \
  --normalize-speaker \
  --data-dir ../emo_db_comb \
  --output-dir ./processed_emodb_comb_norm_loso_soft \
  --split-mode loso \
  --soft-label-decimals 3

echo "Soft-label preprocessing finished."
