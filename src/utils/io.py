"""
utils/io.py
-----------
Checkpoint save / load helpers and config loading.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a nested dict.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file (e.g. ``configs/default.yaml``).

    Returns
    -------
    dict
    """
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    epoch: int,
    val_f1: float,
) -> Path:
    """
    Save model state dict to *checkpoint_dir*.

    Filename convention: ``checkpoint_epoch{N}_f1{score:.4f}.pt``

    Parameters
    ----------
    model : nn.Module
    checkpoint_dir : str | Path
    epoch : int
    val_f1 : float

    Returns
    -------
    Path
        Full path of the saved checkpoint.
    """
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"checkpoint_epoch{epoch}_f1{val_f1:.4f}.pt"
    path = directory / filename
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """
    Load a saved state dict into *model* and return it.

    Parameters
    ----------
    model : nn.Module
        Instantiated model (architecture must match the checkpoint).
    checkpoint_path : str | Path
    device : torch.device

    Returns
    -------
    nn.Module
        Model with loaded weights, moved to *device*.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)
