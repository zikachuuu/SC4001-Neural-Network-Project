"""
Step 1c: Extract segment features from combined EMO-DB audio with soft labels.

What it does:
- Loads generated multi-emotion wav files from emo_db_comb speaker folders.
- Uses timeline CSV intervals (or filename fallback) to compute overlap per emotion in each segment.
- Writes a 7-dim soft label vector per segment, where values sum to 1.0.
- Also writes hard labels (argmax of soft labels) for backward compatibility.

Recommended sequence:
1) Run 1a_generate_dynamic_emodb_combinations.py first.
2) Run this file to build transition-aware targets.
3) Train with y_soft_*.npy if you want soft-label learning.

Quick run examples:
- python 1c_extract_features_emodb_comb_soft_labels.py --output-dir ./processed_emodb_comb_soft
- python 1c_extract_features_emodb_comb_soft_labels.py --normalize-speaker --split-mode loso --output-dir ./processed_emodb_comb_soft_loso
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

from utility import EMOTION_CODE_MAP


np.random.seed(42)

DEFAULT_SPEAKER_ORDER = ["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
FILENAME_PATTERN = re.compile(r"^(\d{2})([A-Z]+)(\d+)\.wav$")


@dataclass(frozen=True)
class SplitConfig:
    train_speakers: Sequence[str] = field(default_factory=lambda: ["03", "08", "09", "10", "11", "12", "13"])
    validation_speakers: Sequence[str] = field(default_factory=lambda: ["14"])
    test_speakers: Sequence[str] = field(default_factory=lambda: ["15", "16"])

    def get_split_name(self, speaker_id: str) -> Optional[str]:
        if speaker_id in self.train_speakers:
            return "train"
        if speaker_id in self.validation_speakers:
            return "validation"
        if speaker_id in self.test_speakers:
            return "test"
        return None


@dataclass(frozen=True)
class FeatureConfig:
    sampling_rate: int = 16000
    n_mels: int = 64
    window_ms: float = 0.025
    hop_ms: float = 0.010
    segment_frames: int = 64
    frame_shift: int = 30


@dataclass(frozen=True)
class PreprocessConfig:
    data_dir: str
    output_dir: str
    normalize_speaker: bool = False
    strict_unknown_emotion: bool = True
    soft_label_decimals: int = 3
    split_config: SplitConfig = field(default_factory=SplitConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass(frozen=True)
class CombinedUtterance:
    filename: str
    speaker_id: str
    file_path: str
    timeline_labels: Optional[List[Tuple[float, float, int]]] = None
    fallback_label_sequence: Tuple[int, ...] = ()


def build_loso_split_configs(speaker_order: Sequence[str]) -> List[SplitConfig]:
    if len(speaker_order) < 3:
        raise ValueError("Need at least 3 speakers to build LOSO folds with train/validation/test.")

    folds: List[SplitConfig] = []
    n = len(speaker_order)
    for i in range(n):
        test_speaker = speaker_order[i]
        validation_speaker = speaker_order[(i + 1) % n]
        train_speakers = [spk for spk in speaker_order if spk not in {test_speaker, validation_speaker}]
        folds.append(
            SplitConfig(
                train_speakers=train_speakers,
                validation_speakers=[validation_speaker],
                test_speakers=[test_speaker],
            )
        )
    return folds


def parse_label_sequence_from_filename(filename: str, strict_unknown_emotion: bool) -> Tuple[int, ...]:
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        return ()

    emotion_codes = list(match.group(2))
    labels: List[int] = []
    for code in emotion_codes:
        if code not in EMOTION_CODE_MAP:
            if strict_unknown_emotion:
                raise ValueError(f"Unknown emotion code '{code}' in filename '{filename}'")
            return ()
        labels.append(EMOTION_CODE_MAP[code])
    return tuple(labels)


def load_speaker_timeline_map(
    speaker_csv_path: str,
    strict_unknown_emotion: bool,
) -> Dict[str, List[Tuple[float, float, int]]]:
    timeline_map: Dict[str, List[Tuple[float, float, int]]] = {}
    if not os.path.exists(speaker_csv_path):
        return timeline_map

    with open(speaker_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generated_file = row.get("generated_file", "").strip()
            code = row.get("emotion_code", "").strip()
            start_sec = float(row.get("start_sec", 0.0))
            end_sec = float(row.get("end_sec", 0.0))

            if generated_file == "":
                continue
            if code not in EMOTION_CODE_MAP:
                if strict_unknown_emotion:
                    raise ValueError(
                        f"Unknown emotion code '{code}' in timeline '{speaker_csv_path}' for file '{generated_file}'"
                    )
                continue

            timeline_map.setdefault(generated_file, []).append((start_sec, end_sec, EMOTION_CODE_MAP[code]))

    for generated_file in timeline_map:
        timeline_map[generated_file].sort(key=lambda x: x[0])

    return timeline_map


def collect_comb_utterances(
    data_dir: str,
    strict_unknown_emotion: bool = True,
) -> List[CombinedUtterance]:
    utterances: List[CombinedUtterance] = []

    for speaker_folder in sorted(os.listdir(data_dir)):
        speaker_dir = os.path.join(data_dir, speaker_folder)
        if not os.path.isdir(speaker_dir):
            continue

        if not speaker_folder.startswith("speaker_"):
            continue
        speaker_id = speaker_folder.split("_")[-1]

        timeline_path = os.path.join(speaker_dir, f"speaker_{speaker_id}_timeline.csv")
        timeline_map = load_speaker_timeline_map(timeline_path, strict_unknown_emotion)

        for filename in sorted(os.listdir(speaker_dir)):
            if not filename.endswith(".wav"):
                continue

            file_path = os.path.join(speaker_dir, filename)
            fallback_sequence = parse_label_sequence_from_filename(filename, strict_unknown_emotion)
            timeline_labels = timeline_map.get(filename)

            utterances.append(
                CombinedUtterance(
                    filename=filename,
                    speaker_id=speaker_id,
                    file_path=file_path,
                    timeline_labels=timeline_labels,
                    fallback_label_sequence=fallback_sequence,
                )
            )

    return utterances


def extract_log_mel(file_path: str, cfg: FeatureConfig) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=cfg.sampling_rate)
    hop_length = int(sr * cfg.hop_ms)
    win_length = int(sr * cfg.window_ms)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=win_length,
        hop_length=hop_length,
        n_mels=cfg.n_mels,
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def compute_speaker_stats(
    utterance_mels: List[Tuple[CombinedUtterance, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    speaker_mels: Dict[str, List[np.ndarray]] = {}
    for utt, log_mel in utterance_mels:
        speaker_mels.setdefault(utt.speaker_id, []).append(log_mel)

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
        raise ValueError(f"No speaker statistics found for speaker '{speaker_id}'")
    return (log_mel - stats["mean"]) / stats["std"]


def build_feature_tensor(log_mel: np.ndarray) -> np.ndarray:
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    return np.stack([log_mel, delta, delta2], axis=0)


def estimate_utterance_duration_sec(total_frames: int, cfg: FeatureConfig) -> float:
    if total_frames <= 0:
        return 0.0
    return cfg.window_ms + (total_frames - 1) * cfg.hop_ms


def build_equal_intervals_from_label_sequence(
    label_sequence: Sequence[int],
    utterance_duration_sec: float,
) -> List[Tuple[float, float, int]]:
    if len(label_sequence) == 0:
        return []
    chunk = utterance_duration_sec / len(label_sequence) if utterance_duration_sec > 0 else 0.0

    intervals: List[Tuple[float, float, int]] = []
    start = 0.0
    for i, label in enumerate(label_sequence):
        end = utterance_duration_sec if i == len(label_sequence) - 1 else start + chunk
        intervals.append((start, end, label))
        start = end
    return intervals


def choose_midpoint_label(
    intervals: Sequence[Tuple[float, float, int]],
    seg_start_sec: float,
    seg_end_sec: float,
) -> int:
    midpoint = 0.5 * (seg_start_sec + seg_end_sec)
    for start, end, label in intervals:
        if start <= midpoint < end:
            return int(label)
    return int(intervals[0][2])


def choose_segment_soft_label(
    intervals: Sequence[Tuple[float, float, int]],
    seg_start_sec: float,
    seg_end_sec: float,
    num_classes: int,
    decimals: int,
) -> np.ndarray:
    probs = np.zeros(num_classes, dtype=np.float32)

    for start, end, label in intervals:
        overlap = max(0.0, min(seg_end_sec, end) - max(seg_start_sec, start))
        if overlap > 0.0:
            probs[int(label)] += float(overlap)

    total_overlap = float(probs.sum())
    if total_overlap > 0.0:
        probs /= total_overlap
    else:
        fallback_label = choose_midpoint_label(intervals, seg_start_sec, seg_end_sec)
        probs[fallback_label] = 1.0

    if decimals >= 0:
        probs = np.round(probs, decimals=decimals)
        rounded_sum = float(probs.sum())
        if rounded_sum <= 0.0:
            fallback_idx = int(np.argmax(probs))
            probs[:] = 0.0
            probs[fallback_idx] = 1.0
        else:
            anchor = int(np.argmax(probs))
            probs[anchor] += float(1.0 - rounded_sum)
            probs = np.clip(probs, 0.0, 1.0)
            clipped_sum = float(probs.sum())
            if clipped_sum <= 0.0:
                probs[:] = 0.0
                probs[anchor] = 1.0
            else:
                probs /= clipped_sum

    return probs.astype(np.float32)


def choose_segment_pair_label(
    intervals: Sequence[Tuple[float, float, int]],
    seg_start_sec: float,
    seg_end_sec: float,
) -> Tuple[int, int]:
    ordered_labels: List[int] = []
    seen_labels = set()

    for start, end, label in intervals:
        overlap = max(0.0, min(seg_end_sec, end) - max(seg_start_sec, start))
        if overlap > 0.0 and int(label) not in seen_labels:
            ordered_labels.append(int(label))
            seen_labels.add(int(label))

    if len(ordered_labels) == 2:
        return ordered_labels[0], ordered_labels[1]

    return (-1, -1)


def slice_segments_with_soft_labels(
    stacked_features: np.ndarray,
    label_intervals: Sequence[Tuple[float, float, int]],
    cfg: FeatureConfig,
    num_classes: int,
    decimals: int,
) -> List[Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]]:
    total_frames = stacked_features.shape[2]
    out: List[Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]] = []
    segment_duration_sec = cfg.window_ms + (cfg.segment_frames - 1) * cfg.hop_ms

    for start_frame in range(0, total_frames - cfg.segment_frames + 1, cfg.frame_shift):
        end_frame = start_frame + cfg.segment_frames
        segment = stacked_features[:, :, start_frame:end_frame]

        seg_start_sec = start_frame * cfg.hop_ms
        seg_end_sec = seg_start_sec + segment_duration_sec
        soft = choose_segment_soft_label(
            intervals=label_intervals,
            seg_start_sec=seg_start_sec,
            seg_end_sec=seg_end_sec,
            num_classes=num_classes,
            decimals=decimals,
        )
        hard = int(np.argmax(soft))
        pair = choose_segment_pair_label(
            intervals=label_intervals,
            seg_start_sec=seg_start_sec,
            seg_end_sec=seg_end_sec,
        )
        out.append((segment, soft, hard, pair))

    return out


def initialize_dataset_buffers() -> Dict[str, Dict[str, List]]:
    return {
        "train": {"X": [], "y": [], "y_soft": [], "y_pair": [], "utterance_ids": []},
        "validation": {"X": [], "y": [], "y_soft": [], "y_pair": [], "utterance_ids": []},
        "test": {"X": [], "y": [], "y_soft": [], "y_pair": [], "utterance_ids": []},
    }


def preprocess_emodb_comb_soft_labels(config: PreprocessConfig) -> Dict[str, Dict[str, np.ndarray]]:
    utterances = collect_comb_utterances(
        data_dir=config.data_dir,
        strict_unknown_emotion=config.strict_unknown_emotion,
    )
    print(f"Found {len(utterances)} combined utterances in {config.data_dir}")

    utterance_mels: List[Tuple[CombinedUtterance, np.ndarray]] = []
    for utt in utterances:
        log_mel = extract_log_mel(utt.file_path, config.feature_config)
        utterance_mels.append((utt, log_mel))

    speaker_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    if config.normalize_speaker:
        print("Computing speaker-wise statistics...")
        speaker_stats = compute_speaker_stats(utterance_mels)

    datasets = initialize_dataset_buffers()
    num_classes = len(EMOTION_CODE_MAP)

    skipped_no_labels = 0
    for utt, raw_log_mel in utterance_mels:
        split_name = config.split_config.get_split_name(utt.speaker_id)
        if split_name is None:
            continue

        normalized_mel = apply_speaker_normalization(raw_log_mel, utt.speaker_id, speaker_stats)
        stacked = build_feature_tensor(normalized_mel)

        intervals = utt.timeline_labels
        if not intervals:
            total_duration = estimate_utterance_duration_sec(normalized_mel.shape[1], config.feature_config)
            intervals = build_equal_intervals_from_label_sequence(utt.fallback_label_sequence, total_duration)

        if not intervals:
            skipped_no_labels += 1
            print(f"[WARN] Skipping file without usable labels: {utt.filename}")
            continue

        segs_with_labels = slice_segments_with_soft_labels(
            stacked_features=stacked,
            label_intervals=intervals,
            cfg=config.feature_config,
            num_classes=num_classes,
            decimals=config.soft_label_decimals,
        )
        for segment, soft_label, hard_label, pair_label in segs_with_labels:
            datasets[split_name]["X"].append(segment)
            datasets[split_name]["y"].append(hard_label)
            datasets[split_name]["y_soft"].append(soft_label)
            datasets[split_name]["y_pair"].append(pair_label)
            datasets[split_name]["utterance_ids"].append(utt.filename)

    if skipped_no_labels > 0:
        print(f"[WARN] Skipped {skipped_no_labels} files due to missing/invalid label timeline information.")

    converted: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in ["train", "validation", "test"]:
        converted[split_name] = {
            "X": np.array(datasets[split_name]["X"]),
            "y": np.array(datasets[split_name]["y"], dtype=np.int64),
            "y_soft": np.array(datasets[split_name]["y_soft"], dtype=np.float32),
            "y_pair": np.array(datasets[split_name]["y_pair"], dtype=np.int64),
            "utterance_ids": np.array(datasets[split_name]["utterance_ids"]),
        }
    return converted


def save_datasets(output_dir: str, datasets: Dict[str, Dict[str, np.ndarray]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        split = datasets[split_name]
        np.save(os.path.join(output_dir, f"X_{split_name}.npy"), split["X"])
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), split["y"])
        np.save(os.path.join(output_dir, f"y_soft_{split_name}.npy"), split["y_soft"])
        np.save(os.path.join(output_dir, f"y_pair_{split_name}.npy"), split["y_pair"])
        np.save(os.path.join(output_dir, f"utterance_ids_{split_name}.npy"), split["utterance_ids"])
        print(
            f"Saved {len(split['X'])} segments for {split_name} to {output_dir} "
            f"(y: {split['y'].shape}, y_soft: {split['y_soft'].shape}, y_pair: {split['y_pair'].shape})"
        )


def parse_args(default_data_dir: str, default_output_dir: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess EMO-DB combined files into segment-level features with soft labels.",
        epilog=(
            "Examples:\n"
            "  python 1c_extract_features_emodb_comb_soft_labels.py --output-dir ./processed_emodb_comb_soft\n"
            "  python 1c_extract_features_emodb_comb_soft_labels.py --normalize-speaker --output-dir ./processed_emodb_comb_soft_norm\n"
            "  python 1c_extract_features_emodb_comb_soft_labels.py --split-mode loso --output-dir ./processed_emodb_comb_soft_loso\n"
            "  python 1c_extract_features_emodb_comb_soft_labels.py --soft-label-decimals 3 --interactive"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--data-dir", default=default_data_dir, help="Path to emo_db_comb folder.")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to save .npy outputs.")
    parser.add_argument(
        "--normalize-speaker",
        action="store_true",
        help="Apply speaker-wise z-score normalization before feature stacking.",
    )
    parser.add_argument("--interactive", action="store_true", help="Prompt for key options interactively.")
    parser.add_argument(
        "--skip-unknown-emotion",
        action="store_true",
        help="Skip files/rows with unknown emotion code instead of raising an error.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["original", "loso"],
        default="original",
        help="Use original fixed speaker split or generate rotating LOSO folds.",
    )
    parser.add_argument(
        "--soft-label-decimals",
        type=int,
        default=3,
        help="Round soft-label values to this many decimals and re-normalize to sum to 1.",
    )
    return parser.parse_args()


def ask_user_yes_no(question: str, default_yes: bool) -> bool:
    default_hint = "Y/n" if default_yes else "y/N"
    raw = input(f"{question} [{default_hint}]: ").strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"}


def prompt_interactive(args: argparse.Namespace) -> argparse.Namespace:
    print("Interactive mode enabled. Press Enter to keep defaults.")

    data_dir = input(f"Data directory [{args.data_dir}]: ").strip()
    if data_dir:
        args.data_dir = data_dir

    output_dir = input(f"Output directory [{args.output_dir}]: ").strip()
    if output_dir:
        args.output_dir = output_dir

    args.normalize_speaker = ask_user_yes_no(
        "Enable speaker-wise normalization?",
        default_yes=args.normalize_speaker,
    )

    skip_unknown = ask_user_yes_no(
        "Skip files with unknown emotion code?",
        default_yes=args.skip_unknown_emotion,
    )
    args.skip_unknown_emotion = skip_unknown

    use_loso = ask_user_yes_no(
        "Use LOSO mode (10 rotating folds)?",
        default_yes=(args.split_mode == "loso"),
    )
    args.split_mode = "loso" if use_loso else "original"

    decimals_raw = input(f"Soft-label decimals [{args.soft_label_decimals}]: ").strip()
    if decimals_raw:
        args.soft_label_decimals = int(decimals_raw)

    return args


def run_pipeline(args: argparse.Namespace) -> None:
    config = PreprocessConfig(
        data_dir=os.path.abspath(args.data_dir),
        output_dir=os.path.abspath(args.output_dir),
        normalize_speaker=args.normalize_speaker,
        strict_unknown_emotion=not args.skip_unknown_emotion,
        soft_label_decimals=args.soft_label_decimals,
    )

    print("=" * 70)
    print("EMO-DB combined preprocessing configuration (soft labels)")
    print(f"data_dir            : {config.data_dir}")
    print(f"output_dir          : {config.output_dir}")
    print(f"normalize_speaker   : {config.normalize_speaker}")
    print(f"strict_unknown_code : {config.strict_unknown_emotion}")
    print(f"soft_label_decimals : {config.soft_label_decimals}")
    print(f"split_mode          : {args.split_mode}")
    print("=" * 70)

    if args.split_mode == "original":
        datasets = preprocess_emodb_comb_soft_labels(config)
        save_datasets(config.output_dir, datasets)
        print("Preprocessing completed.")
        return

    loso_folds = build_loso_split_configs(DEFAULT_SPEAKER_ORDER)
    print(f"Generating {len(loso_folds)} LOSO folds in: {config.output_dir}")

    for idx, split_cfg in enumerate(loso_folds, start=1):
        fold_dir = os.path.join(config.output_dir, f"fold{idx}")
        fold_config = PreprocessConfig(
            data_dir=config.data_dir,
            output_dir=fold_dir,
            normalize_speaker=config.normalize_speaker,
            strict_unknown_emotion=config.strict_unknown_emotion,
            soft_label_decimals=config.soft_label_decimals,
            split_config=split_cfg,
            feature_config=config.feature_config,
        )

        print("-" * 70)
        print(
            f"fold{idx}: test={split_cfg.test_speakers[0]}, "
            f"validation={split_cfg.validation_speakers[0]}, "
            f"train={len(split_cfg.train_speakers)} speakers"
        )
        datasets = preprocess_emodb_comb_soft_labels(fold_config)
        save_datasets(fold_dir, datasets)

    print("LOSO preprocessing completed.")


def main() -> None:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(curr_dir, "../emo_db_comb/")
    default_output_dir = os.path.join(curr_dir, "./processed_emodb_comb_soft/")

    args = parse_args(default_data_dir, default_output_dir)
    if args.interactive:
        args = prompt_interactive(args)

    run_pipeline(args)


if __name__ == "__main__":
    main()
