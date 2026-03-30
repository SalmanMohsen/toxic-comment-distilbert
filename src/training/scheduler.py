"""
training/scheduler.py
---------------------
Optimizer and learning-rate scheduler factory.

Supports per-parameter-group learning rates (differential LR) via the
optional param_groups argument, which is used by Trainer to give the
backbone a smaller lr than the classification head.
"""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    total_steps: int,
    param_groups: Optional[List[dict]] = None,
):
    """
    Build AdamW optimizer and a linear warmup + decay scheduler.

    Parameters
    ----------
    model : nn.Module
    learning_rate : float
        Peak learning rate. Used as the default if param_groups is None,
        and as the reference rate for scheduler scaling when it is not.
    weight_decay : float
        L2 regularisation coefficient (e.g. 0.01).
    warmup_ratio : float
        Fraction of total steps used for linear warmup (e.g. 0.10).
    total_steps : int
        Total gradient update steps across all epochs.
    param_groups : List[dict] | None
        Optional list of per-group dicts with 'params' and 'lr' keys,
        used for differential learning rates (backbone vs head).
        When None, all model parameters use learning_rate.

    Returns
    -------
    Tuple[AdamW, LambdaLR]
    """
    if param_groups is not None:
        # Attach weight_decay to each group that does not override it
        for group in param_groups:
            group.setdefault("weight_decay", weight_decay)
        optimizer = AdamW(param_groups)
    else:
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