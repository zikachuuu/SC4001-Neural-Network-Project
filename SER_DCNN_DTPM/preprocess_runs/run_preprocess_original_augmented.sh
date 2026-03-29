#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" 1_extract_features_emodb.py \
  --split-mode original \
  --augment-training \
  --output-dir ./processed_emodb_og_aug

echo "Done: original split, no normalization, augmentation enabled -> processed_emodb_og_aug"
