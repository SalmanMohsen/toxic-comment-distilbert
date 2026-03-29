"""
training/trainer.py
-------------------
Training loop with early stopping, checkpointing, and history tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_loader
from training.scheduler import build_optimizer_and_scheduler
from utils.io import save_checkpoint
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Result container ──────────────────────────────────────────────────────

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
    best_epoch: int = 0
    elapsed_minutes: float = 0.0


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
    weight_decay : float
    warmup_ratio : float
    grad_clip : float
        Max gradient norm for clipping (prevents exploding gradients during
        transformer fine-tuning).
    patience : int
        Early stopping: stop if val F1 does not improve for this many epochs.
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
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.10,
        grad_clip: float = 1.0,
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

        total_steps = len(train_loader) * epochs
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
        )
        logger.info(
            "Trainer ready | epochs=%d | lr=%.1e | steps=%d",
            epochs, learning_rate, total_steps,
        )

    # ── public API ────────────────────────────────────────────────────────

    def fit(self) -> TrainingResult:
        """
        Run the full training loop and return a :class:`TrainingResult`.

        Returns
        -------
        TrainingResult
            Contains per-epoch history, best checkpoint path, and timing.
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

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_F1=%.4f | val_AUROC=%.4f",
                epoch, self.epochs, train_loss, val_loss, val_f1, val_auroc,
            )

            # Checkpoint if best
            if val_f1 > result.best_val_f1:
                result.best_val_f1 = val_f1
                result.best_epoch = epoch
                patience_counter = 0
                ckpt = save_checkpoint(
                    self.model, self.checkpoint_dir, epoch, val_f1
                )
                result.best_checkpoint = ckpt
                logger.info("New best — saved checkpoint: %s", ckpt.name)
            else:
                patience_counter += 1
                logger.info(
                    "No improvement (%d/%d).", patience_counter, self.patience
                )
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        result.elapsed_minutes = (time.time() - start) / 60
        logger.info(
            "Training complete in %.1f min | best val_F1=%.4f at epoch %d",
            result.elapsed_minutes, result.best_val_f1, result.best_epoch,
        )
        return result

    # ── private helpers ───────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> float:
        """Run one full pass over *train_loader*; return mean loss."""
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
            # L = -(1/N) * Σ_i Σ_{c} y_{i,c} · log( softmax(Wh_i + b)_c )
            loss = outputs.loss
            loss.backward()

            # Gradient clipping — essential for stable transformer fine-tuning
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )

            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()

            if (step + 1) % self.log_every_n_steps == 0:
                avg = running_loss / (step + 1)
                logger.info(
                    "  Epoch %d | step %d/%d | loss=%.4f",
                    epoch, step + 1, len(self.train_loader), avg,
                )

        return running_loss / len(self.train_loader)
