"""
models/classifier.py
--------------------
Thin factory wrapper around HuggingFace AutoModelForSequenceClassification.

Keeping the model construction in one place makes it easy to swap
architectures (e.g. BERT-base → RoBERTa) by changing only the config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


def build_model(model_name: str, num_labels: int = 2) -> nn.Module:
    """
    Instantiate a pre-trained sequence classification model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"distilbert-base-uncased"``).
    num_labels : int
        Number of output classes (2 for binary classification).

    Returns
    -------
    nn.Module
        Model ready for fine-tuning (weights on CPU; call ``.to(device)``
        after this function).
    """
    logger.info("Loading model '%s' with %d labels…", model_name, num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded — %.2fM parameters.", n_params / 1e6)
    return model


def build_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the fast tokenizer corresponding to *model_name*.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    PreTrainedTokenizerFast
    """
    logger.info("Loading tokenizer for '%s'…", model_name)
    return AutoTokenizer.from_pretrained(model_name)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("GPU: %s (VRAM %.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        logger.info("No GPU found — running on CPU.")
    return device
