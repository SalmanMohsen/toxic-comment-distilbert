"""
evaluation/metrics.py
---------------------
Evaluation utilities: per-batch inference loop, F1, AUROC, confusion matrix.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Run inference over *loader* and return aggregate metrics.

    Parameters
    ----------
    model : nn.Module
        Fine-tuned classifier.
    loader : DataLoader
    device : torch.device

    Returns
    -------
    Tuple[float, float, float]
        ``(avg_loss, f1_score, auroc)``
    """
    model.eval()
    all_preds: list = []
    all_labels: list = []
    all_probs: list = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    auroc = roc_auc_score(all_labels, all_probs)
    return avg_loss, f1, auroc


def full_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Produce a full evaluation report including confusion matrix and
    per-class metrics.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device

    Returns
    -------
    dict
        Keys: ``loss``, ``f1``, ``auroc``, ``confusion_matrix``,
        ``classification_report``, ``all_preds``, ``all_labels``, ``all_probs``.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["non-toxic", "toxic"],
        output_dict=True,
    )
    logger.info("\n%s", classification_report(
        all_labels, all_preds, target_names=["non-toxic", "toxic"]
    ))

    return {
        "loss":                   total_loss / len(loader),
        "f1":                     f1_score(all_labels, all_preds),
        "auroc":                  roc_auc_score(all_labels, all_probs),
        "confusion_matrix":       cm,
        "classification_report":  report,
        "all_preds":              all_preds,
        "all_labels":             all_labels,
        "all_probs":              all_probs,
    }
