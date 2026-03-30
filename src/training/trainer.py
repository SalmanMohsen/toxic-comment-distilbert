"""
training/trainer.py
-------------------
Training loop with early stopping, checkpointing, and history tracking.

Overfitting mitigations applied
--------------------------------
1. Early stopping on **val_loss** (not val_F1).
   Val loss rising while F1 creeps up is the classic sign that the model
   is memorising training data. Watching val_loss catches this earlier.

2. Classifier-head **dropout** injected before the linear layer.
   DistilBERT's default pre-classifier dropout is 0.2; we expose it as a
   configurable parameter so it can be tuned without changing the backbone.

3. **Differential learning rates** — the backbone gets a 10x smaller lr
   than the classification head. This prevents the pre-trained
   representations from being overwritten too quickly on small datasets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_loader
from training.scheduler import build_optimizer_and_scheduler
from utils.io import save_checkpoint
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Result containers ─────────────────────────────────────────────────────

@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_f1: float
    val_auroc: float


@dataclass
class TrainingResult:
    history: List[EpochRecord] = field(default_factory=list)
    best_checkpoint: Optional[Path] = None
    best_val_f1: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    elapsed_minutes: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────

def set_classifier_dropout(model: nn.Module, dropout: float) -> None:
    """
    Set the dropout rate on DistilBERT's pre-classifier layer.

    DistilBERT's DistilBertForSequenceClassification has a dedicated
    pre_classifier Linear layer followed by a dropout module.
    Increasing dropout from the default 0.2 toward 0.3-0.4 is the lightest
    regularisation lever available without changing the backbone.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace DistilBertForSequenceClassification instance.
    dropout : float
        Dropout probability to apply (e.g. 0.3).
    """
    if hasattr(model, "dropout"):
        model.dropout.p = dropout
        logger.info("Set classifier dropout to %.2f", dropout)
    else:
        logger.warning(
            "Could not find model.dropout — dropout rate unchanged. "
            "This may happen with non-DistilBERT architectures."
        )


# ── Trainer ───────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates one full training run.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    device : torch.device
    epochs : int
    learning_rate : float
        Learning rate applied to the classification head.
        The backbone receives learning_rate / backbone_lr_factor.
    backbone_lr_factor : float
        Divisor applied to learning_rate for backbone parameters.
        Default 10 means the backbone trains 10x slower than the head,
        preserving pre-trained representations on small datasets.
    weight_decay : float
    warmup_ratio : float
    grad_clip : float
    classifier_dropout : float
        Dropout probability on the pre-classifier layer.
        Increase from 0.2 (default) toward 0.4 to combat overfitting.
    patience : int
        Early stopping patience measured in epochs.
        Triggers when val_loss does not decrease (not val_F1).
    checkpoint_dir : str | Path
    log_every_n_steps : int
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        backbone_lr_factor: float = 10.0,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.10,
        grad_clip: float = 1.0,
        classifier_dropout: float = 0.3,
        patience: int = 2,
        checkpoint_dir: str | Path = "checkpoints",
        log_every_n_steps: int = 50,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every_n_steps = log_every_n_steps

        # Apply dropout regularisation to the classifier head
        set_classifier_dropout(self.model, classifier_dropout)

        # ── Differential learning rates ───────────────────────────────────
        # Backbone (distilbert.*) gets lr / backbone_lr_factor.
        # Classifier head (pre_classifier.*, classifier.*) gets full lr.
        # This prevents catastrophic forgetting of pre-trained features.
        backbone_lr = learning_rate / backbone_lr_factor
        param_groups = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "distilbert" in n
                ],
                "lr": backbone_lr,
                "name": "backbone",
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "distilbert" not in n
                ],
                "lr": learning_rate,
                "name": "head",
            },
        ]
        logger.info(
            "Differential LR | backbone=%.1e | head=%.1e",
            backbone_lr, learning_rate,
        )

        total_steps = len(train_loader) * epochs
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
            param_groups=param_groups,
        )
        logger.info(
            "Trainer ready | epochs=%d | total_steps=%d | dropout=%.2f",
            epochs, total_steps, classifier_dropout,
        )

    # ── public API ────────────────────────────────────────────────────────

    def fit(self) -> TrainingResult:
        """
        Run the full training loop.

        Early stopping criterion: val_loss must decrease.
        This catches the diverging-loss pattern observed in the run where
        val_loss rose (0.129 -> 0.146 -> 0.161) even as val_F1 kept improving
        marginally — a sign of overfitting masked by the metric.

        Returns
        -------
        TrainingResult
        """
        result = TrainingResult()
        patience_counter = 0
        start = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss, val_f1, val_auroc = evaluate_loader(
                self.model, self.val_loader, self.device
            )

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_f1=val_f1,
                val_auroc=val_auroc,
            )
            result.history.append(record)

            # Log the train/val loss gap — a rising gap signals overfitting
            # before it shows up in F1.
            loss_gap = val_loss - train_loss
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f "
                "(gap=+%.4f) | val_F1=%.4f | val_AUROC=%.4f",
                epoch, self.epochs,
                train_loss, val_loss, loss_gap, val_f1, val_auroc,
            )
            if loss_gap > 0.05:
                logger.warning(
                    "val_loss - train_loss = %.4f > 0.05 — overfitting signal.",
                    loss_gap,
                )

            # Always track best F1 for the final report
            if val_f1 > result.best_val_f1:
                result.best_val_f1 = val_f1

            # ── Early stopping & checkpointing on val_loss ────────────────
            # We checkpoint on val_loss (not val_F1) because val_loss rising
            # while F1 inches up is the overfitting pattern we observed:
            # val_loss: 0.129 -> 0.146 -> 0.161 across 3 epochs.
            if val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                result.best_epoch = epoch
                patience_counter = 0
                ckpt = save_checkpoint(
                    self.model, self.checkpoint_dir, epoch, val_f1
                )
                result.best_checkpoint = ckpt
                logger.info(
                    "New best val_loss=%.4f — saved: %s", val_loss, ckpt.name
                )
            else:
                patience_counter += 1
                logger.info(
                    "val_loss did not improve (%d/%d).",
                    patience_counter, self.patience,
                )
                if patience_counter >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d "
                        "(val_loss has not improved for %d epochs).",
                        epoch, self.patience,
                    )
                    break

        result.elapsed_minutes = (time.time() - start) / 60
        logger.info(
            "Training complete in %.1f min | best val_loss=%.4f "
            "| best val_F1=%.4f at epoch %d",
            result.elapsed_minutes,
            result.best_val_loss,
            result.best_val_f1,
            result.best_epoch,
        )
        return result

    # ── private helpers ───────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> float:
        """Run one full pass over train_loader; return mean loss."""
        self.model.train()
        running_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Loss: categorical cross-entropy
            # L = -(1/N) * sum_i sum_c [ y_{i,c} * log(softmax(Wh_i+b)_c) ]
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )

            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()

            if (step + 1) % self.log_every_n_steps == 0:
                logger.info(
                    "  Epoch %d | step %d/%d | loss=%.4f",
                    epoch, step + 1, len(self.train_loader),
                    running_loss / (step + 1),
                )

        return running_loss / len(self.train_loader)