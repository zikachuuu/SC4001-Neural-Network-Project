import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

from utility import EMOTION_CODE_MAP, EMOTION_ENG_MAP, augment_volume, add_matrix_noise, matrix_pitch_shift, freq_mask, matrix_time_stretch, time_mask


np.random.seed(42)

DEFAULT_SPEAKER_ORDER = ["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"]


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for speaker-based train/validation/test split.
        - EMO-DB has 10 speakers (03, 08, 09, 10, 11, 12, 13, 14, 15, 16).
        - The default split is speaker (03, 08, 09, 10, 11, 12, 13) for training, speaker (14) for validation, and speakers (15, 16) for testing.
        - This class allows customizing the split if needed.
        - The get_split_name method returns the split name for a given speaker ID or None if the speaker is not in any split.

    Information about the speakers:
        - 03 - male, 31 years old
        - 08 - female, 34 years old
        - 09 - female, 21 years old
        - 10 - male, 32 years old
        - 11 - male, 26 years old
        - 12 - male, 30 years old
        - 13 - female, 32 years old
        - 14 - female, 35 years old
        - 15 - male, 25 years old
        - 16 - female, 31 years old

    (The default split is chosen by myself for convience sake, paper used speaker-independent Leave-One-Speaker-Out (LOSO) 
    or Leave-One-Speakers-Group-Out (LOSGO) cross-validation strategy)
    """

    train_speakers      : Sequence[str] = field(default_factory=lambda: ["03", "08", "09", "10", "11", "12", "13"])
    validation_speakers : Sequence[str] = field(default_factory=lambda: ["14"])
    test_speakers       : Sequence[str] = field(default_factory=lambda: ["15", "16"])

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
    """
    Configuration for feature extraction parameters.
        - sampling_rate: Target sampling rate for audio loading (default 16 kHz).
        - n_mels: Number of Mel bands for the spectrogram (default 64).
        - window_ms: Window size in milliseconds for STFT (default 25 ms).
        - hop_ms: Hop size in milliseconds for STFT (default 10 ms) - results in overlapping windows.
        - segment_frames: Number of frames per segment for slicing (default 64 frames, which is 655 ms = 10 ms * 63 + 25 ms).
        - frame_shift: Number of frames to shift for the next segment (default 30 frames, which is 300 ms = 10 ms * 30).

        (The default parameters are directly taken from the paper)
    """
    sampling_rate       : int = 16000
    n_mels              : int = 64
    window_ms           : float = 0.025
    hop_ms              : float = 0.010
    segment_frames      : int = 64
    frame_shift         : int = 30


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration for the entire preprocessing pipeline.
        - data_dir: Directory containing the raw EMO-DB wav files.
        - output_dir: Directory to save the processed .npy files.
        - normalize_speaker: Whether to apply speaker-wise normalization.
        - strict_unknown_emotion: Whether to raise an error or skip files with unknown emotion codes.
        - split_config: Configuration for train/validation/test split.
        - feature_config: Configuration for feature extraction parameters.
    """
    data_dir                : str
    output_dir              : str
    normalize_speaker       : bool          = False
    augment_training        : bool          = False
    strict_unknown_emotion  : bool          = True
    split_config            : SplitConfig   = field(default_factory=SplitConfig)
    feature_config          : FeatureConfig = field(default_factory=FeatureConfig)


def build_loso_split_configs(speaker_order: Sequence[str]) -> List[SplitConfig]:
    """Create rotating LOSO-like folds with 1 test speaker, 1 validation speaker, and remaining train speakers."""
    if len(speaker_order) < 3:
        raise ValueError("Need at least 3 speakers to build LOSO folds with train/validation/test.")

    folds: List[SplitConfig] = []
    n = len(speaker_order)
    for i in range(n):
        test_speaker        = speaker_order[i]
        validation_speaker  = speaker_order[(i + 1) % n]
        train_speakers      = [spk for spk in speaker_order if spk not in {test_speaker, validation_speaker}]
        folds.append(
            SplitConfig(
                train_speakers      = train_speakers,
                validation_speakers = [validation_speaker],
                test_speakers       = [test_speaker],
            )
        )
    return folds


def collect_utterances(
        data_dir                : str, 
        strict_unknown_emotion  : bool = True
    ) -> List[Tuple[str, str, int, str]]:
    """
    Parse the EMO-DB directory and collect valid utterances with their metadata.
    
    Return (filename, speaker_id, label, file_path) for valid EMO-DB utterances.
    """
    utterances: List[Tuple[str, str, int, str]] = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".wav"):
            continue

        speaker_id = filename[:2]
        emotion_code = filename[5]
        if emotion_code not in EMOTION_CODE_MAP:
            if strict_unknown_emotion:
                raise ValueError(f"Unknown emotion code '{emotion_code}' in filename '{filename}'")
            print(f"[WARN] Skipping file with unknown emotion code: {filename}")
            continue

        label = EMOTION_CODE_MAP[emotion_code]
        utterances.append((filename, speaker_id, label, os.path.join(data_dir, filename)))
    return utterances


def extract_log_mel(
        file_path   : str,
        cfg         : FeatureConfig
    ) -> np.ndarray:
    """
    Load the audio file and compute the log-Mel spectrogram.

    Return a 2D array of shape (n_mels, time_frames) containing the log-Mel spectrogram.
        - Rows are the Mel scale, which is a non-linear frequency scale (to approximate human auditory perception).
        - Columns are time frames, determined by the window and hop sizes.
        - Values are the log-scaled amplitude (loudness) of the Mel spectrogram (again to approximate human perception of loudness).

    Note that the resulting log-Mel spectrogram is not yet segmented into fixed-size segments or stacked with delta features.

    We later will have to split the time_frames into segments of segment_frames (e.g., 64 frames) 
    with a certain frame_shift (e.g., 30 frames) to create the final input samples for the model.
    """
    y, sr = librosa.load(file_path, sr=cfg.sampling_rate)
    hop_length = int(sr * cfg.hop_ms)
    win_length = int(sr * cfg.window_ms)

    mel_spec = librosa.feature.melspectrogram(
        y           = y,
        sr          = sr,
        n_fft       = win_length,
        hop_length  = hop_length,
        n_mels      = cfg.n_mels,
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def compute_speaker_stats(
        utterance_mels: List[Tuple[str, str, int, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute mean and std for each speaker across all their utterances for speaker-wise normalization.
    Here we are finding the mean and std for each Mel frequency bin across all time frames.
    So we are finding the "average" log amplitude for each Mel bin (frequency) for that speaker.

    Return a dictionary mapping speaker_id to their mean and std arrays, which can be used for z-score normalization later.
    """
    speaker_mels: Dict[str, List[np.ndarray]] = {}
    for _, speaker_id, _, log_mel in utterance_mels:
        speaker_mels.setdefault(speaker_id, []).append(log_mel)

    speaker_stats: Dict[str, Dict[str, np.ndarray]] = {}
    for speaker_id, mels in speaker_mels.items():
        concatenated    = np.concatenate(mels, axis=1)
        mean            = np.mean(concatenated, axis=1, keepdims=True)
        std             = np.std(concatenated, axis=1, keepdims=True) + 1e-8

        speaker_stats[speaker_id] = {"mean": mean, "std": std}
    return speaker_stats


def apply_speaker_normalization(
    log_mel         : np.ndarray,
    speaker_id      : str,
    speaker_stats   : Optional[Dict[str, Dict[str, np.ndarray]]],
    ) -> np.ndarray:
    """
    Apply speaker-wise z-score normalization to the log-Mel spectrogram for a single speaker
     - If speaker_stats is None, return the original log-Mel without normalization.
     - If speaker_stats is provided but does not contain the speaker_id, raise an error.
    """
    if speaker_stats is None:
        return log_mel

    stats = speaker_stats.get(speaker_id)
    if stats is None:
        raise ValueError(f"No speaker statistics found for speaker '{speaker_id}'")
    return (log_mel - stats["mean"]) / stats["std"]


def generate_variants(
        log_mel         : np.ndarray, 
        split_name      : str, 
        augment_training: bool
    ) -> List[Tuple[str, np.ndarray]]:
    """
    Return feature variants for one utterance.

    If augmentation is enabled, generate 3 additional clones ONLY for training split:
        1) identity_tweak: random pitch shift in [-3, +3] bins (excluding 0)
        2) specaugment_tweak: one frequency mask + one time mask
        3) environment_tweak: random volume shift + light gaussian noise
    """
    variants: List[Tuple[str, np.ndarray]] = [("base", log_mel)]

    if not augment_training or split_name != "train":
        return variants

    # Clone 1: identity tweak (pitch only)
    shift_bins = np.random.randint(-3, 4)
    if shift_bins == 0:
        shift_bins = 1
    variants.append(("identity_tweak", matrix_pitch_shift(log_mel, shift_bins=shift_bins)))

    # Clone 2: specaugment tweak (freq mask + time mask)
    specaug_mel = freq_mask(log_mel)
    specaug_mel = time_mask(specaug_mel)
    variants.append(("specaugment_tweak", specaug_mel))

    # Clone 3: environment tweak (volume shift + noise)
    env_mel = augment_volume(log_mel)
    env_mel = add_matrix_noise(env_mel)
    variants.append(("environment_tweak", env_mel))

    return variants


def build_feature_tensor(log_mel: np.ndarray) -> np.ndarray:
    """
    Build a 3D feature tensor by stacking the log-Mel spectrogram with its first and second order deltas.
    """
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    return np.stack([log_mel, delta, delta2], axis=0)


def slice_segments(
        stacked_features: np.ndarray, 
        cfg             : FeatureConfig
    ) -> List[np.ndarray]:
    """
    Slice the stacked feature tensor into overlapping segments of fixed size.
        - stacked_features has shape (3, n_mels, time_frames)
        - We want to slice along the time_frames dimension into segments of segment_frames (e.g., 64 frames) with a shift of frame_shift (e.g., 30 frames).
        - This will create multiple segments for each utterance, which can help increase the number of training samples and capture temporal dynamics.
        - The resulting segments will have shape (3, n_mels, segment_frames) and will be returned as a list.
        - Note that some frames at the end may be discarded if they don't fit into a full segment.
        - The paper does not specify how to handle the last few frames that don't fit into a full segment, so we will simply discard them for simplicity.
    """
    total_frames = stacked_features.shape[2]
    segments: List[np.ndarray] = []

    for start in range(0, total_frames - cfg.segment_frames + 1, cfg.frame_shift):
        end = start + cfg.segment_frames
        segments.append(stacked_features[:, :, start:end])
    return segments


def initialize_dataset_buffers() -> Dict[str, Dict[str, List]]:
    return {
        "train": {"X": [], "y": [], "utterance_ids": []},
        "validation": {"X": [], "y": [], "utterance_ids": []},
        "test": {"X": [], "y": [], "utterance_ids": []},
    }


def preprocess_emodb(config: PreprocessConfig) -> Dict[str, Dict[str, np.ndarray]]:

    # 1. We collect all valid utterances from the EMO-DB directory, extracting their filename, speaker ID, emotion label, and file path.
    utterances = collect_utterances(
        data_dir                = config.data_dir,
        strict_unknown_emotion  = config.strict_unknown_emotion,
    )
    print(f"Found {len(utterances)} valid utterances in {config.data_dir}")

    # 2. For each utterance, we extract the log-Mel spectrogram (2D matrix) and store it along with its metadata in a list.
    utterance_mels: List[Tuple[str, str, int, np.ndarray]] = []
    for filename, speaker_id, label, file_path in utterances:
        log_mel = extract_log_mel(file_path, config.feature_config)
        utterance_mels.append((filename, speaker_id, label, log_mel))

    # 3. If speaker normalization is enabled, we compute the mean and std for each speaker across all their utterances to prepare for z-score normalization.
    speaker_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    if config.normalize_speaker:
        print("Computing speaker-wise statistics...")
        speaker_stats = compute_speaker_stats(utterance_mels)

    # 4. For each utterance, we apply speaker normalization if enabled, 
    #       generate feature variants (currently just the base log-Mel), 
    #       build the stacked feature tensor (log-Mel + deltas), 
    #       and slice it into segments. 
    #       Each segment is then added to the appropriate dataset split based on the speaker ID.
    datasets = initialize_dataset_buffers()
    for filename, speaker_id, label, raw_log_mel in utterance_mels:
        split_name = config.split_config.get_split_name(speaker_id)
        if split_name is None:
            continue

        normalized_mel  = apply_speaker_normalization(raw_log_mel, speaker_id, speaker_stats)
        variants        = generate_variants(
            normalized_mel,
            split_name=split_name,
            augment_training=config.augment_training,
        )

        for variant_name, variant_mel in variants:
            stacked_features    = build_feature_tensor(variant_mel)
            segments            = slice_segments(stacked_features, config.feature_config)

            utterance_tag = filename if variant_name == "base" else f"{filename}|{variant_name}"
            for segment in segments:
                datasets[split_name]["X"].append(segment)
                datasets[split_name]["y"].append(label)
                datasets[split_name]["utterance_ids"].append(utterance_tag)

    # 5. Finally, we convert the lists in each split to numpy arrays for efficient storage and return the processed datasets.
    converted: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in ["train", "validation", "test"]:
        converted[split_name] = {
            "X": np.array(datasets[split_name]["X"]),
            "y": np.array(datasets[split_name]["y"]),
            "utterance_ids": np.array(datasets[split_name]["utterance_ids"]),
        }
    return converted


def save_datasets(
    output_dir  : str, 
    datasets    : Dict[str, Dict[str, np.ndarray]]
    ) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        split = datasets[split_name]
        np.save(os.path.join(output_dir, f"X_{split_name}.npy"), split["X"])
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), split["y"])
        np.save(os.path.join(output_dir, f"utterance_ids_{split_name}.npy"), split["utterance_ids"])
        print(f"Saved {len(split['X'])} segments for {split_name} to {output_dir}")


def parse_args(
        default_data_dir    : str, 
        default_output_dir  : str
    ) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified EMO-DB feature extraction with optional speaker normalization.",
        epilog=(
            "Examples:\n"
            "  python 1_extract_features_emodb.py --normalize-speaker --output-dir ./processed_emodb_speaker_norm\n"
            "  python 1_extract_features_emodb.py --output-dir ./processed_emodb_og\n"
            "  python 1_extract_features_emodb.py --split-mode loso --output-dir ./processed_emodb_og\n"
            "  python 1_extract_features_emodb.py --interactive"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data-dir", 
        default = default_data_dir, 
        help    = "Path to EMO-DB wav folder."
    )

    parser.add_argument(
        "--output-dir", 
        default = default_output_dir, 
        help    = "Directory to save .npy outputs."
    )

    parser.add_argument(
        "--normalize-speaker",
        action  = "store_true",
        help    = "Apply speaker-wise z-score normalization before feature stacking.",
    )

    parser.add_argument(
        "--augment-training",
        action="store_true",
        help="Generate 3 augmented clones for each training file (4x train data including base).",
    )

    parser.add_argument(
        "--interactive",
        action  = "store_true",
        help    = "Prompt for key options interactively.",
    )

    parser.add_argument(
        "--skip-unknown-emotion",
        action  = "store_true",
        help    = "Skip files with unknown emotion code instead of raising an error.",
    )

    parser.add_argument(
        "--split-mode",
        choices=["original", "loso"],
        default="original",
        help="Use original fixed speaker split or generate rotating LOSO folds.",
    )

    return parser.parse_args()


def ask_user_yes_no(
        question    : str, 
        default_yes : bool
    ) -> bool:
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

    args.augment_training = ask_user_yes_no(
        "Enable training augmentation (3 clones per training file)?",
        default_yes=args.augment_training,
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

    return args


def run_pipeline(args: argparse.Namespace) -> None:
    config = PreprocessConfig(
        data_dir                = os.path.abspath(args.data_dir),
        output_dir              = os.path.abspath(args.output_dir),
        normalize_speaker       = args.normalize_speaker,
        augment_training        = args.augment_training,
        strict_unknown_emotion  = not args.skip_unknown_emotion,
    )

    print("=" * 70)
    print("EMO-DB preprocessing configuration")
    print(f"data_dir            : {config.data_dir}")
    print(f"output_dir          : {config.output_dir}")
    print(f"normalize_speaker   : {config.normalize_speaker}")
    print(f"augment_training    : {config.augment_training}")
    print(f"strict_unknown_code : {config.strict_unknown_emotion}")
    print(f"split_mode          : {args.split_mode}")
    print("=" * 70)

    if args.split_mode == "original":
        datasets = preprocess_emodb(config)
        save_datasets(config.output_dir, datasets)
        print("Preprocessing completed.")
        return

    # LOSO mode: create fold1..fold10 under output_dir.
    loso_folds = build_loso_split_configs(DEFAULT_SPEAKER_ORDER)
    print(f"Generating {len(loso_folds)} LOSO folds in: {config.output_dir}")

    for idx, split_cfg in enumerate(loso_folds, start=1):
        fold_dir = os.path.join(config.output_dir, f"fold{idx}")
        fold_config = PreprocessConfig(
            data_dir                = config.data_dir,
            output_dir              = fold_dir,
            normalize_speaker       = config.normalize_speaker,
            augment_training        = config.augment_training,
            strict_unknown_emotion  = config.strict_unknown_emotion,
            split_config            = split_cfg,
            feature_config          = config.feature_config,
        )

        print("-" * 70)
        print(
            f"fold{idx}: test={split_cfg.test_speakers[0]}, "
            f"validation={split_cfg.validation_speakers[0]}, "
            f"train={len(split_cfg.train_speakers)} speakers"
        )
        datasets = preprocess_emodb(fold_config)
        save_datasets(fold_dir, datasets)

    print("LOSO preprocessing completed.")


def main() -> None:
    curr_dir                = os.path.dirname(os.path.abspath(__file__))
    default_data_dir        = os.path.join(curr_dir, "../emo_db/")
    default_output_dir      = os.path.join(curr_dir, "./processed_emodb_og/")

    args = parse_args(default_data_dir, default_output_dir)
    if args.interactive:
        args = prompt_interactive(args)

    run_pipeline(args)


if __name__ == "__main__":
    main()