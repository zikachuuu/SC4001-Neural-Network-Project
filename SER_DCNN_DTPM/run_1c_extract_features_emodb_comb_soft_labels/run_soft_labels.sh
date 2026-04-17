#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1) Build combined audio first, if needed.
# python 1a_generate_dynamic_emodb_combinations.py \
#   --data-dir ../emo_db \
#   --output-dir ../emo_db_comb

# 2) Soft-label preprocessing with LOSO split.
python B2b_extract_features_emodb_comb_soft_labels.py \
  --normalize-speaker \
  --data-dir ../emo_db_comb \
  --output-dir ./processed_emodb_comb_norm_loso_soft \
  --split-mode loso \
  --soft-label-decimals 3

echo "Soft-label preprocessing finished."
