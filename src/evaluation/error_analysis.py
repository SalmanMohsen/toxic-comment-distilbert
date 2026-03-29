"""
evaluation/error_analysis.py
-----------------------------
Tools for inspecting model failure modes.

Focuses on False Negatives (toxic comments the model missed) because
in content-moderation settings the cost of missing harmful content is
higher than the cost of over-flagging benign content.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.logger import get_logger

logger = get_logger(__name__)


def get_false_negatives(
    texts: List[str],
    true_labels: List[int],
    pred_labels: List[int],
) -> List[Tuple[str, int, int]]:
    """
    Return all (text, true_label, pred_label) triples where the model
    predicted non-toxic (0) but the true label was toxic (1).

    Parameters
    ----------
    texts : List[str]
    true_labels : List[int]
    pred_labels : List[int]

    Returns
    -------
    List[Tuple[str, int, int]]
    """
    fns = [
        (text, true, pred)
        for text, true, pred in zip(texts, true_labels, pred_labels)
        if true == 1 and pred == 0
    ]
    fn_rate = len(fns) / max(1, sum(1 for t in true_labels if t == 1)) * 100
    logger.info(
        "False Negatives: %d / %d toxic samples (FN rate %.1f%%)",
        len(fns), sum(1 for t in true_labels if t == 1), fn_rate,
    )
    return fns


def plot_confusion_matrix(
    true_labels: List[int],
    pred_labels: List[int],
    save_path: str | Path = "confusion_matrix.png",
) -> None:
    """
    Plot and save a labelled confusion matrix heatmap.

    Parameters
    ----------
    true_labels : List[int]
    pred_labels : List[int]
    save_path : str | Path
    """
    cm = confusion_matrix(true_labels, pred_labels)
    logger.info("Confusion matrix:\n%s", cm)
    logger.info(
        "FN (toxic missed): %d | FP (clean flagged): %d",
        cm[1][0], cm[0][1],
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-toxic", "Toxic"],
        yticklabels=["Non-toxic", "Toxic"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_loss_curves(
    history: list,
    save_path: str | Path = "loss_curves.png",
) -> None:
    """
    Plot training / validation loss and validation F1 across epochs.

    Parameters
    ----------
    history : List[EpochRecord]
        List of :class:`training.trainer.EpochRecord` objects.
    save_path : str | Path
    """
    epochs      = [r.epoch      for r in history]
    train_loss  = [r.train_loss for r in history]
    val_loss    = [r.val_loss   for r in history]
    val_f1      = [r.val_f1     for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_loss, "o-",  label="Train Loss")
    ax1.plot(epochs, val_loss,   "s--", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_f1, "D-", color="green", label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curves saved to %s", save_path)
