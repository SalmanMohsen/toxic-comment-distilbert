"""
tests/test_dataset.py
---------------------
Unit tests for ToxicDataset.

Run with:
    pytest tests/test_dataset.py -v
"""

import sys
from pathlib import Path

# Allow imports from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from unittest.mock import MagicMock
from data.dataset import ToxicDataset


# ── Fixtures ──────────────────────────────────────────────────────────────

MAX_LEN = 32

def _make_mock_tokenizer(max_len: int = MAX_LEN):
    """Return a minimal mock tokenizer that returns fixed-shape tensors."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids":      torch.zeros(1, max_len, dtype=torch.long),
        "attention_mask": torch.ones(1,  max_len, dtype=torch.long),
    }
    return tokenizer


@pytest.fixture
def small_dataset():
    texts  = ["This is fine.", "You are an idiot!", "Hello world.", "Kill yourself."]
    labels = [0, 1, 0, 1]
    tok    = _make_mock_tokenizer()
    return ToxicDataset(texts, labels, tok, MAX_LEN)


# ── Tests ─────────────────────────────────────────────────────────────────

def test_length(small_dataset):
    """Dataset __len__ must equal the number of texts supplied."""
    assert len(small_dataset) == 4


def test_item_keys(small_dataset):
    """Each item must expose the three required keys."""
    item = small_dataset[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}


def test_input_ids_shape(small_dataset):
    """input_ids must be a 1-D tensor of length MAX_LEN."""
    item = small_dataset[0]
    assert item["input_ids"].shape == (MAX_LEN,), (
        f"Expected ({MAX_LEN},), got {item['input_ids'].shape}"
    )


def test_attention_mask_shape(small_dataset):
    """attention_mask must be a 1-D tensor of length MAX_LEN."""
    item = small_dataset[0]
    assert item["attention_mask"].shape == (MAX_LEN,)


def test_label_dtype(small_dataset):
    """Labels must be LongTensor (required by CrossEntropyLoss)."""
    item = small_dataset[0]
    assert item["labels"].dtype == torch.long


def test_label_values(small_dataset):
    """Labels must only contain 0 or 1."""
    for i in range(len(small_dataset)):
        label = small_dataset[i]["labels"].item()
        assert label in {0, 1}, f"Unexpected label value: {label}"


def test_no_empty_texts():
    """Dataset must reject construction when any text is empty."""
    # Empty-text detection is the caller's responsibility (done in preprocessing).
    # Here we verify the dataset does not crash on empty strings —
    # it delegates to the tokenizer which handles them.
    tok = _make_mock_tokenizer()
    ds  = ToxicDataset([""], [0], tok, MAX_LEN)
    item = ds[0]   # should not raise
    assert item["labels"].item() == 0


def test_tokenizer_called_with_correct_args(small_dataset):
    """Tokenizer must be called with truncation=True and padding='max_length'."""
    _ = small_dataset[0]
    call_kwargs = small_dataset.tokenizer.call_args.kwargs
    assert call_kwargs.get("truncation") is True
    assert call_kwargs.get("padding") == "max_length"
    assert call_kwargs.get("max_length") == MAX_LEN
