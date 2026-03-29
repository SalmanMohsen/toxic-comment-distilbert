
"""
data/dataset.py
---------------
PyTorch Dataset that wraps tokenised comment texts.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ToxicDataset(Dataset):
    """
    Map-style dataset for the Jigsaw toxic comment task.

    Each item is a dict with keys:
    - ``input_ids``      : LongTensor of shape (max_len,)
    - ``attention_mask`` : LongTensor of shape (max_len,)
    - ``labels``         : LongTensor scalar (0 or 1)

    Parameters
    ----------
    texts : List[str]
        Raw comment strings.
    labels : List[int]
        Binary labels (0 = non-toxic, 1 = toxic).
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer (e.g. DistilBertTokenizerFast).
    max_len : int
        Maximum token sequence length; longer inputs are truncated,
        shorter ones are padded to this length.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    # ── dunder methods ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }
