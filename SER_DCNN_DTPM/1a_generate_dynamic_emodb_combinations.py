"""
Step 1a: Generate combined multi-emotion utterances from original EMO-DB wav files.

What it does:
- Groups files by speaker.
- Concatenates 2 to 4 source utterances from the same speaker.
- Saves generated wav files into speaker folders.
- Writes timeline CSV files that store start/end emotion intervals for each generated file.

Recommended sequence:
1) Start from original EMO-DB wav files.
2) Run this file to generate combined audio + timeline labels.
3) Run 1b_extract_features_emodb_comb.py for hard labels per segment.
4) Or run 1c_extract_features_emodb_comb_soft_labels.py for soft labels per segment.

Quick run examples:
- python 1a_generate_dynamic_emodb_combinations.py --output-dir ../emo_db_comb
- python 1a_generate_dynamic_emodb_combinations.py --min-concat 2 --max-concat 4 --unique-combinations
"""

import argparse
import itertools
import math
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf


RNG_SEED = 42

# Speaker metadata from EMO-DB documentation.
SPEAKER_AGE_MAP: Dict[str, int] = {
    "03": 31,
    "08": 34,
    "09": 21,
    "10": 32,
    "11": 26,
    "12": 30,
    "13": 32,
    "14": 35,
    "15": 25,
    "16": 31,
}

# EMO-DB filename position 5 stores the short emotion code.
EMOTION_CODE_TO_NAME = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}


@dataclass(frozen=True)
class GenerationConfig:
    data_dir            : str
    output_dir          : str
    min_concat          : int = 2
    max_concat          : int = 4
    seed                : int = RNG_SEED
    unique_combinations : bool = False


def parse_args(
        default_data_dir    : str, 
        default_output_dir  : str
    ) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dynamic multi-emotion EMO-DB files by concatenating 2-4 same-speaker files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir", 
        default = default_data_dir, 
        help    = "Input EMO-DB wav folder"
    )
    parser.add_argument(
        "--output-dir", 
        default = default_output_dir, 
        help    = "Output folder for combined wav files"
    )
    parser.add_argument(
        "--min-concat", 
        type    = int, 
        default = 2, 
        help    = "Minimum number of source files to concatenate"
    )
    parser.add_argument(
        "--max-concat", 
        type    = int, 
        default = 4, 
        help    = "Maximum number of source files to concatenate"
    )
    parser.add_argument(
        "--seed", 
        type    = int, 
        default = RNG_SEED, 
        help    = "Random seed"
    )
    parser.add_argument(
        "--unique-combinations",
        action="store_true",
        help=(
            "Guarantee no duplicate source-file combinations per speaker "
            "(combination identity ignores ordering)."
        ),
    )
    return parser.parse_args()


def collect_speaker_files(data_dir: str) -> Dict[str, List[str]]:
    speaker_files: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".wav"):
            continue
        speaker_id = name[:2]
        speaker_files.setdefault(speaker_id, []).append(name)
    return speaker_files


def read_audio_float32(path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr)
    return audio.astype(np.float32), sr


def extract_emotion_code(filename: str) -> str:
    if len(filename) < 6:
        return "X"
    return filename[5]


def build_output_name(
    speaker_id: str,
    sample_index: int,
    source_files: Sequence[str],
) -> str:
    emotion_codes = [extract_emotion_code(name) for name in source_files]
    emotion_seq = "".join(emotion_codes)
    # Requested format: <speaker code><emotions><index>
    # Example: 03WAF001.wav
    return f"{speaker_id}{emotion_seq}{sample_index:03d}.wav"


def write_speaker_log(
    log_path: str,
    speaker_id: str,
    rows: Sequence[Tuple[str, Sequence[str]]],
) -> None:
    age = SPEAKER_AGE_MAP.get(speaker_id, "unknown")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"speaker_id: {speaker_id}\n")
        f.write(f"speaker_age: {age}\n")
        f.write(f"generated_files: {len(rows)}\n")
        f.write("=" * 80 + "\n")
        for idx, (generated_name, sources) in enumerate(rows, start=1):
            codes = [extract_emotion_code(src) for src in sources]
            names = [EMOTION_CODE_TO_NAME.get(code, "unknown") for code in codes]
            f.write(f"[{idx}] generated: {generated_name}\n")
            f.write(f"    emotion_codes: {' -> '.join(codes)}\n")
            f.write(f"    emotion_names: {' -> '.join(names)}\n")
            f.write("    source_files:\n")
            for src in sources:
                f.write(f"      - {src}\n")
            f.write("-" * 80 + "\n")


def write_speaker_timeline_csv(
    csv_path: str,
    rows: Sequence[Dict[str, object]],
) -> None:
    fieldnames = [
        "generated_file",
        "speaker_id",
        "segment_order",
        "source_file",
        "emotion_code",
        "emotion_name",
        "start_sec",
        "end_sec",
        "duration_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_all_unique_combinations(files: Sequence[str], min_k: int, max_k: int) -> List[Tuple[str, ...]]:
    all_combos: List[Tuple[str, ...]] = []
    for k in range(min_k, max_k + 1):
        all_combos.extend(list(itertools.combinations(files, k)))
    return all_combos


def pick_unique_groups_diverse_length(
    files: Sequence[str],
    min_k: int,
    max_k: int,
    target_count: int,
    rng: random.Random,
) -> List[Tuple[str, ...]]:
    """
    Pick unique combinations while encouraging diversity of combination length.

    Strategy:
    1) Build all unique combinations grouped by k.
    2) If possible, allocate at least one sample per available k.
    3) Distribute remaining quota with capacity-aware weighted sampling so larger-k pools
       no longer dominate as aggressively as pure all-combo sampling.
    """
    combos_by_k: Dict[int, List[Tuple[str, ...]]] = {}
    for k in range(min_k, max_k + 1):
        combos = list(itertools.combinations(files, k))
        rng.shuffle(combos)
        if combos:
            combos_by_k[k] = combos

    if not combos_by_k:
        return []

    total_available = sum(len(v) for v in combos_by_k.values())
    target = min(target_count, total_available)
    if target <= 0:
        return []

    ks = sorted(combos_by_k.keys())
    selected_count_by_k = {k: 0 for k in ks}

    # Give at least one per available k when quota allows.
    remaining = target
    if target >= len(ks):
        for k in ks:
            selected_count_by_k[k] = 1
            remaining -= 1

    # Capacity-aware distribution for remaining slots.
    # We scale by sqrt(remaining capacity) to reduce large-k domination.
    while remaining > 0:
        candidate_ks = [k for k in ks if selected_count_by_k[k] < len(combos_by_k[k])]
        if not candidate_ks:
            break

        weights = [math.sqrt(len(combos_by_k[k]) - selected_count_by_k[k]) for k in candidate_ks]
        total_w = sum(weights)
        if total_w <= 0:
            chosen_k = rng.choice(candidate_ks)
        else:
            pick = rng.random() * total_w
            acc = 0.0
            chosen_k = candidate_ks[-1]
            for k, w in zip(candidate_ks, weights):
                acc += w
                if pick <= acc:
                    chosen_k = k
                    break

        selected_count_by_k[chosen_k] += 1
        remaining -= 1

    selected: List[Tuple[str, ...]] = []
    for k in ks:
        take_n = selected_count_by_k[k]
        if take_n > 0:
            selected.extend(combos_by_k[k][:take_n])

    rng.shuffle(selected)
    return selected


def generate_dynamic_audio(config: GenerationConfig) -> None:
    if config.min_concat < 2:
        raise ValueError("min_concat must be at least 2.")
    if config.max_concat < config.min_concat:
        raise ValueError("max_concat must be greater than or equal to min_concat.")

    random.seed(config.seed)
    np.random.seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)
    speaker_files = collect_speaker_files(config.data_dir)
    if not speaker_files:
        raise ValueError(f"No .wav files found in data directory: {config.data_dir}")

    total_generated = 0
    rng = random.Random(config.seed)

    for speaker_id in sorted(speaker_files.keys()):
        files = speaker_files[speaker_id]
        n_original = len(files)
        if n_original < 2:
            print(f"[WARN] Speaker {speaker_id} has fewer than 2 files. Skipping.")
            continue

        speaker_out_dir = os.path.join(config.output_dir, f"speaker_{speaker_id}")
        os.makedirs(speaker_out_dir, exist_ok=True)

        min_k = max(2, config.min_concat)
        max_k = min(config.max_concat, n_original)
        if min_k > max_k:
            print(
                f"[WARN] Speaker {speaker_id} does not have enough files for requested range "
                f"[{config.min_concat}, {config.max_concat}]. Skipping."
            )
            continue

        generated_rows: List[Tuple[str, Sequence[str]]] = []
        timeline_rows: List[Dict[str, object]] = []

        # Build selected source-file groups for this speaker.
        selected_groups: List[Sequence[str]] = []
        target_count = n_original
        if config.unique_combinations:
            all_unique = build_all_unique_combinations(files, min_k, max_k)

            if len(all_unique) < target_count:
                print(
                    f"[WARN] Speaker {speaker_id}: requested {target_count} unique combinations, "
                    f"but only {len(all_unique)} are possible in range [{min_k}, {max_k}]. "
                    "Generating the maximum available unique combinations."
                )
                target_count = len(all_unique)

            selected_groups = pick_unique_groups_diverse_length(
                files=files,
                min_k=min_k,
                max_k=max_k,
                target_count=target_count,
                rng=rng,
            )
        else:
            # Original behavior: random sampling with possible duplicate source combinations.
            for _ in range(target_count):
                k = rng.randint(min_k, max_k)
                selected_groups.append(rng.sample(files, k=k))

        for sample_index, chosen in enumerate(selected_groups, start=1):

            concatenated_parts: List[np.ndarray] = []
            base_sr = None
            timeline_start_sec = 0.0
            for src_name in chosen:
                src_path = os.path.join(config.data_dir, src_name)
                part, part_sr = read_audio_float32(src_path, target_sr=base_sr)
                if base_sr is None:
                    base_sr = part_sr
                concatenated_parts.append(part)

                duration_sec = float(len(part) / part_sr) if part_sr else 0.0
                code = extract_emotion_code(src_name)
                timeline_rows.append(
                    {
                        "generated_file": "",  # filled after output filename is known
                        "speaker_id": speaker_id,
                        "segment_order": len([r for r in timeline_rows if r.get("speaker_id") == speaker_id]) + 1,
                        "source_file": src_name,
                        "emotion_code": code,
                        "emotion_name": EMOTION_CODE_TO_NAME.get(code, "unknown"),
                        "start_sec": round(timeline_start_sec, 6),
                        "end_sec": round(timeline_start_sec + duration_sec, 6),
                        "duration_sec": round(duration_sec, 6),
                    }
                )
                timeline_start_sec += duration_sec

            if not concatenated_parts:
                continue

            merged = np.concatenate(concatenated_parts, axis=0)
            out_name = build_output_name(speaker_id, sample_index, chosen)
            out_path = os.path.join(speaker_out_dir, out_name)
            sf.write(out_path, merged, base_sr)

            # Back-fill generated_file + per-file segment order for rows added in this loop.
            seg_order = 1
            for row in reversed(timeline_rows):
                if row["generated_file"] != "":
                    break
                row["generated_file"] = out_name
            for row in timeline_rows:
                if row["generated_file"] == out_name:
                    row["segment_order"] = seg_order
                    seg_order += 1

            generated_rows.append((out_name, chosen))
            total_generated += 1

        log_path = os.path.join(speaker_out_dir, f"speaker_{speaker_id}_generation_log.txt")
        write_speaker_log(log_path, speaker_id, generated_rows)
        timeline_csv_path = os.path.join(speaker_out_dir, f"speaker_{speaker_id}_timeline.csv")
        write_speaker_timeline_csv(timeline_csv_path, timeline_rows)

        print(
            f"Speaker {speaker_id}: generated {len(generated_rows)} files "
            f"(from {n_original} originals). Log: {log_path}; Timeline CSV: {timeline_csv_path}"
        )

    print("=" * 80)
    print(f"Done. Generated {total_generated} combined files in: {config.output_dir}")


def main() -> None:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(curr_dir, "../emo_db"))
    default_output_dir = os.path.abspath(os.path.join(curr_dir, "../emo_db_comb"))

    args = parse_args(default_data_dir, default_output_dir)
    config = GenerationConfig(
        data_dir=os.path.abspath(args.data_dir),
        output_dir=os.path.abspath(args.output_dir),
        min_concat=args.min_concat,
        max_concat=args.max_concat,
        seed=args.seed,
        unique_combinations=args.unique_combinations,
    )

    print("=" * 80)
    print("Dynamic EMO-DB combination generation")
    print(f"data_dir   : {config.data_dir}")
    print(f"output_dir : {config.output_dir}")
    print(f"concat_k   : [{config.min_concat}, {config.max_concat}]")
    print(f"seed       : {config.seed}")
    print(f"unique_combinations: {config.unique_combinations}")
    print("=" * 80)

    generate_dynamic_audio(config)


if __name__ == "__main__":
    main()
