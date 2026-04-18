"""
Mel-spectrogram preprocessing pipeline for EmoDB.

Re-implements the same feature extraction as the baseline
(SC4001-Neural-Network-Project/SER_DCNN_DTPM/1_extract_features_emodb.py)
so that Track 1 (Transformer) operates on identical features.

Usage:
    python ser_preprocess.py \
        --data-dir SC4001-Neural-Network-Project/emodb \
        --output-dir processed_data/emodb_norm_loso \
        --normalize-speaker --split-mode loso
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from ser_utils import (
    EMOTION_CODE_MAP,
    DEFAULT_SPEAKER_ORDER,
    SplitConfig,
    build_loso_split_configs,
    collect_utterances,
)


# ──────────────────────────────────────────────────────────────────────
# Feature extraction config (matches baseline exactly)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureConfig:
    sampling_rate: int = 16000
    n_mels: int = 64
    window_ms: float = 0.025       # 25 ms → win_length = 400
    hop_ms: float = 0.010          # 10 ms → hop_length = 160
    segment_frames: int = 64       # segment width in time frames
    frame_shift: int = 30          # overlap shift


# ──────────────────────────────────────────────────────────────────────
# Core feature extraction functions
# ──────────────────────────────────────────────────────────────────────

def extract_log_mel(file_path: str, cfg: FeatureConfig) -> np.ndarray:
    """Load audio, compute log-mel spectrogram. Returns (n_mels, T)."""
    y, sr = librosa.load(file_path, sr=cfg.sampling_rate)
    hop_length = int(sr * cfg.hop_ms)
    win_length = int(sr * cfg.window_ms)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=win_length,
        hop_length=hop_length,
        n_mels=cfg.n_mels,
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def build_feature_tensor(log_mel: np.ndarray) -> np.ndarray:
    """Stack log-mel + delta + delta2 → (3, n_mels, T)."""
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    return np.stack([log_mel, delta, delta2], axis=0)


def slice_segments(features: np.ndarray, cfg: FeatureConfig) -> List[np.ndarray]:
    """Slice (3, n_mels, T) into overlapping (3, n_mels, segment_frames) segments."""
    total_frames = features.shape[2]
    segments: List[np.ndarray] = []
    for start in range(0, total_frames - cfg.segment_frames + 1, cfg.frame_shift):
        end = start + cfg.segment_frames
        segments.append(features[:, :, start:end])
    return segments


# ──────────────────────────────────────────────────────────────────────
# Speaker normalization
# ──────────────────────────────────────────────────────────────────────

def compute_speaker_stats(
    utterance_mels: List[Tuple[str, str, int, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute per-speaker mean/std across all mel bins (axis=1 = time)."""
    speaker_mels: Dict[str, List[np.ndarray]] = {}
    for _, speaker_id, _, log_mel in utterance_mels:
        speaker_mels.setdefault(speaker_id, []).append(log_mel)

    speaker_stats: Dict[str, Dict[str, np.ndarray]] = {}
    for speaker_id, mels in speaker_mels.items():
        concatenated = np.concatenate(mels, axis=1)
        mean = np.mean(concatenated, axis=1, keepdims=True)
        std = np.std(concatenated, axis=1, keepdims=True) + 1e-8
        speaker_stats[speaker_id] = {"mean": mean, "std": std}
    return speaker_stats


def apply_speaker_normalization(
    log_mel: np.ndarray,
    speaker_id: str,
    speaker_stats: Optional[Dict[str, Dict[str, np.ndarray]]],
) -> np.ndarray:
    if speaker_stats is None:
        return log_mel
    stats = speaker_stats.get(speaker_id)
    if stats is None:
        raise ValueError(f"No stats for speaker '{speaker_id}'")
    return (log_mel - stats["mean"]) / stats["std"]


# ──────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────

def preprocess_fold(
    data_dir: str,
    split_config: SplitConfig,
    feature_config: FeatureConfig,
    normalize_speaker: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run the full preprocessing pipeline for one LOSO fold."""
    utterances = collect_utterances(data_dir)
    print(f"  Found {len(utterances)} valid utterances")

    # Extract log-mel for every utterance
    utterance_mels: List[Tuple[str, str, int, np.ndarray]] = []
    for filename, speaker_id, label, file_path in utterances:
        log_mel = extract_log_mel(file_path, feature_config)
        utterance_mels.append((filename, speaker_id, label, log_mel))

    # Speaker normalization stats
    speaker_stats = None
    if normalize_speaker:
        speaker_stats = compute_speaker_stats(utterance_mels)

    # Build segments per split
    datasets: Dict[str, Dict[str, List]] = {
        s: {"X": [], "y": [], "utterance_ids": []}
        for s in ("train", "validation", "test")
    }

    for filename, speaker_id, label, raw_log_mel in utterance_mels:
        split_name = split_config.get_split_name(speaker_id)
        if split_name is None:
            continue

        normalized_mel = apply_speaker_normalization(
            raw_log_mel, speaker_id, speaker_stats
        )
        stacked = build_feature_tensor(normalized_mel)
        segments = slice_segments(stacked, feature_config)

        for seg in segments:
            datasets[split_name]["X"].append(seg)
            datasets[split_name]["y"].append(label)
            datasets[split_name]["utterance_ids"].append(filename)

    # Convert to numpy
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in ("train", "validation", "test"):
        d = datasets[split_name]
        result[split_name] = {
            "X": np.array(d["X"]) if d["X"] else np.empty((0, 3, 64, 64)),
            "y": np.array(d["y"]),
            "utterance_ids": np.array(d["utterance_ids"]),
        }
    return result


def save_datasets(
    output_dir: str, datasets: Dict[str, Dict[str, np.ndarray]]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        d = datasets[split_name]
        np.save(os.path.join(output_dir, f"X_{split_name}.npy"), d["X"])
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), d["y"])
        np.save(
            os.path.join(output_dir, f"utt_ids_{split_name}.npy"),
            d["utterance_ids"],
        )
        print(f"    {split_name}: {len(d['X'])} segments")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EmoDB mel-spectrogram preprocessing")
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to EmoDB wav directory",
    )
    parser.add_argument(
        "--output-dir", default="processed_data/emodb_norm_loso",
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--normalize-speaker", action="store_true",
        help="Apply speaker-wise z-score normalization",
    )
    parser.add_argument(
        "--split-mode", choices=["original", "loso"], default="loso",
        help="Split mode: 'original' (fixed) or 'loso' (10-fold LOSO CV)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    feature_config = FeatureConfig()

    print("=" * 60)
    print(f"  data_dir:           {data_dir}")
    print(f"  output_dir:         {output_dir}")
    print(f"  normalize_speaker:  {args.normalize_speaker}")
    print(f"  split_mode:         {args.split_mode}")
    print(f"  sr={feature_config.sampling_rate}, n_mels={feature_config.n_mels}, "
          f"win={feature_config.window_ms}s, hop={feature_config.hop_ms}s, "
          f"seg={feature_config.segment_frames}, shift={feature_config.frame_shift}")
    print("=" * 60)

    if args.split_mode == "original":
        split_config = SplitConfig()
        datasets = preprocess_fold(
            data_dir, split_config, feature_config, args.normalize_speaker
        )
        save_datasets(output_dir, datasets)
    else:
        folds = build_loso_split_configs(DEFAULT_SPEAKER_ORDER)
        for idx, split_cfg in enumerate(folds, start=1):
            fold_dir = os.path.join(output_dir, f"fold{idx:02d}")
            print(
                f"\nFold {idx:02d}: test={split_cfg.test_speakers[0]}, "
                f"val={split_cfg.validation_speakers[0]}, "
                f"train={len(split_cfg.train_speakers)} speakers"
            )
            datasets = preprocess_fold(
                data_dir, split_cfg, feature_config, args.normalize_speaker
            )
            save_datasets(fold_dir, datasets)

    print("\nDone.")


if __name__ == "__main__":
    main()
