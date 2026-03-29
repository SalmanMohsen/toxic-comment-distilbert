"""
training/scheduler.py
---------------------
Optimizer and learning-rate scheduler factory.

Centralising these here means hyperparameter changes (lr, weight_decay,
warmup_ratio) only need to be made in the config, not scattered across
training code.
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    total_steps: int,
):
    """
    Build AdamW optimizer and a linear warmup + decay scheduler.

    The learning-rate schedule is:
    - Linear warmup from 0 → *learning_rate* over the first
      ``warmup_ratio * total_steps`` steps.
    - Linear decay from *learning_rate* → 0 over the remaining steps.

    Parameters
    ----------
    model : nn.Module
    learning_rate : float
        Peak learning rate (e.g. 2e-5).
    weight_decay : float
        L2 regularisation coefficient (e.g. 0.01).
    warmup_ratio : float
        Fraction of total steps used for warmup (e.g. 0.10).
    total_steps : int
        Total number of gradient update steps across all epochs.

    Returns
    -------
    Tuple[AdamW, LambdaLR]
    """
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler
