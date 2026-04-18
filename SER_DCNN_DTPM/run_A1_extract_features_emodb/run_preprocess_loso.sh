#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" A1_extract_features_emodb.py \
  --split-mode loso \
  --output-dir ./processed_emodb_og_loso

echo "Done: LOSO split, no normalization, no augmentation -> processed_emodb_og_loso"
