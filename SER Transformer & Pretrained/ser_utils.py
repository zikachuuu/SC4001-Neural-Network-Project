"""
Shared utilities for SC4001 Speech Emotion Recognition project (4a, 4b).
"""

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# EMO-DB emotion code → integer label  (character at index 5 of filename)
EMOTION_CODE_MAP: Dict[str, int] = {
    "W": 0,  # Wut        → Anger
    "L": 1,  # Langeweile → Boredom
    "A": 2,  # Angst      → Fear
    "F": 3,  # Freude     → Happiness
    "T": 4,  # Trauer     → Sadness
    "E": 5,  # Ekel       → Disgust
    "N": 6,  # Neutral    → Neutral
}

EMOTION_ENG_MAP: Dict[int, str] = {
    0: "Anger",
    1: "Boredom",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Disgust",
    6: "Neutral",
}

NUM_CLASSES = 7

DEFAULT_SPEAKER_ORDER: List[str] = [
    "03", "08", "09", "10", "11", "12", "13", "14", "15", "16",
]


# ──────────────────────────────────────────────────────────────────────
# LOSO split configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SplitConfig:
    """Speaker-based train / validation / test split."""

    train_speakers: Sequence[str] = field(
        default_factory=lambda: ["03", "08", "09", "10", "11", "12", "13"]
    )
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


def build_loso_split_configs(
    speaker_order: Sequence[str] = DEFAULT_SPEAKER_ORDER,
) -> List[SplitConfig]:
    """Create 10 rotating LOSO folds (1 test, 1 val, 8 train)."""
    n = len(speaker_order)
    if n < 3:
        raise ValueError("Need ≥3 speakers for LOSO folds.")
    folds: List[SplitConfig] = []
    for i in range(n):
        test_spk = speaker_order[i]
        val_spk = speaker_order[(i + 1) % n]
        train_spks = [s for s in speaker_order if s not in {test_spk, val_spk}]
        folds.append(
            SplitConfig(
                train_speakers=train_spks,
                validation_speakers=[val_spk],
                test_speakers=[test_spk],
            )
        )
    return folds


# ──────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────

def collect_utterances(
    data_dir: str,
    strict_unknown_emotion: bool = True,
) -> List[Tuple[str, str, int, str]]:
    """
    Parse the EmoDB directory and return a list of
    (filename, speaker_id, label, file_path) for every valid .wav file.
    """
    utterances: List[Tuple[str, str, int, str]] = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".wav"):
            continue
        speaker_id = filename[:2]
        emotion_code = filename[5]
        if emotion_code not in EMOTION_CODE_MAP:
            if strict_unknown_emotion:
                raise ValueError(
                    f"Unknown emotion code '{emotion_code}' in '{filename}'"
                )
            print(f"[WARN] Skipping file with unknown emotion code: {filename}")
            continue
        label = EMOTION_CODE_MAP[emotion_code]
        utterances.append((filename, speaker_id, label, os.path.join(data_dir, filename)))
    return utterances


# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# Majority voting & segment aggregation
# ──────────────────────────────────────────────────────────────────────

def majority_vote(labels) -> int:
    """Return the most common label (ties broken by first-seen)."""
    counts: Dict[int, int] = {}
    for lb in labels:
        k = int(lb)
        counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get)


def aggregate_segment_predictions(
    seg_preds: np.ndarray,
    seg_true: np.ndarray,
    utterance_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group segment-level predictions by utterance ID, apply majority vote.
    Returns (y_true_utterance, y_pred_utterance).
    """
    utt_pred_buckets: Dict[str, List[int]] = {}
    utt_true_label: Dict[str, int] = {}

    for uid, pred, true_label in zip(utterance_ids, seg_preds, seg_true):
        uid_str = str(uid)
        if uid_str not in utt_pred_buckets:
            utt_pred_buckets[uid_str] = []
            utt_true_label[uid_str] = int(true_label)
        utt_pred_buckets[uid_str].append(int(pred))

    y_true, y_pred = [], []
    for uid in sorted(utt_pred_buckets.keys()):
        y_true.append(utt_true_label[uid])
        y_pred.append(majority_vote(utt_pred_buckets[uid]))
    return np.array(y_true), np.array(y_pred)


# ──────────────────────────────────────────────────────────────────────
# Evaluation & reporting
# ──────────────────────────────────────────────────────────────────────

def evaluate_and_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    results_dir: str = "results",
    split_mode: str = "loso",
) -> Tuple[float, str, np.ndarray]:
    """
    Compute utterance-level accuracy, classification report, confusion matrix.
    Prints results and saves to ``results_dir``.
    """
    os.makedirs(results_dir, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    target_names = [EMOTION_ENG_MAP[i] for i in range(NUM_CLASSES)]
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))

    print(f"\n{'=' * 60}")
    print(f"  {model_name}  —  {split_mode.upper()}")
    print(f"  Utterance-level accuracy: {acc * 100:.2f}%")
    print(f"{'=' * 60}")
    print(report)

    # Save text report
    report_path = os.path.join(results_dir, f"report_{model_name}_{split_mode}.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: {split_mode}\n")
        f.write(f"Utterance-level accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    # Save confusion matrix plot
    plot_confusion_matrix(cm, model_name, results_dir, split_mode)

    return acc, report, cm


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    results_dir: str = "results",
    split_mode: str = "loso",
) -> None:
    """Save a labelled confusion-matrix heatmap."""
    target_names = [EMOTION_ENG_MAP[i] for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names, ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix ({split_mode.upper()})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    path = os.path.join(results_dir, f"cm_{model_name}_{split_mode}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {path}")


# ──────────────────────────────────────────────────────────────────────
# Class-weight computation
# ──────────────────────────────────────────────────────────────────────

def compute_class_weights(
    labels: np.ndarray, num_classes: int = NUM_CLASSES
) -> torch.FloatTensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(float)
    total = len(labels)
    weights = total / (num_classes * counts + 1e-8)
    return torch.FloatTensor(weights)
