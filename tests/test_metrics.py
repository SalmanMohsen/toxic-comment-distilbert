"""
tests/test_metrics.py
---------------------
Unit tests for evaluation utilities.

Uses a tiny dummy model and synthetic DataLoader so no GPU or real
checkpoint is required.

Run with:
    pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset
from evaluation.metrics import evaluate_loader, full_report


# ── Helpers ───────────────────────────────────────────────────────────────

class _PerfectModel(nn.Module):
    """
    A mock model that always predicts the correct class by returning
    logits of [-100, 100] for label=1 and [100, -100] for label=0.
    """

    def forward(self, input_ids, attention_mask, labels=None):
        batch = input_ids.shape[0]
        # logits: shape (B, 2)
        logits = torch.where(
            labels.unsqueeze(1) == 1,
            torch.tensor([[-100.0, 100.0]]).expand(batch, -1),
            torch.tensor([[100.0, -100.0]]).expand(batch, -1),
        )
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        from types import SimpleNamespace
        return SimpleNamespace(loss=loss, logits=logits)


class _AllNegativeModel(nn.Module):
    """Always predicts class 0 (non-toxic)."""

    def forward(self, input_ids, attention_mask, labels=None):
        batch  = input_ids.shape[0]
        logits = torch.tensor([[100.0, -100.0]]).expand(batch, -1)
        loss   = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        from types import SimpleNamespace
        return SimpleNamespace(loss=loss, logits=logits)


def _make_loader(n: int = 20, seq_len: int = 8, pos_ratio: float = 0.5) -> DataLoader:
    """Synthetic DataLoader with random token ids and stratified labels."""
    input_ids      = torch.randint(0, 100, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    n_pos  = int(n * pos_ratio)
    labels = torch.tensor([1] * n_pos + [0] * (n - n_pos), dtype=torch.long)
    ds = TensorDataset(input_ids, attention_mask, labels)

    # Wrap so batches look like dicts (matching our Dataset contract)
    class _DictLoader:
        def __init__(self, tensor_loader):
            self._loader = tensor_loader

        def __iter__(self):
            for ids, mask, lbl in self._loader:
                yield {"input_ids": ids, "attention_mask": mask, "labels": lbl}

        def __len__(self):
            return len(self._loader)

    return _DictLoader(DataLoader(ds, batch_size=10, shuffle=False))


DEVICE = torch.device("cpu")


# ── evaluate_loader tests ─────────────────────────────────────────────────

def test_evaluate_loader_returns_three_values():
    model  = _PerfectModel()
    loader = _make_loader()
    result = evaluate_loader(model, loader, DEVICE)
    assert len(result) == 3, "evaluate_loader must return (loss, f1, auroc)"


def test_evaluate_loader_perfect_model_f1_is_1():
    model  = _PerfectModel()
    loader = _make_loader(n=20)
    _, f1, auroc = evaluate_loader(model, loader, DEVICE)
    assert f1    == pytest.approx(1.0, abs=1e-4)
    assert auroc == pytest.approx(1.0, abs=1e-4)


def test_evaluate_loader_all_negative_model_f1_is_0():
    """A model that always predicts 0 should achieve F1=0 for the toxic class."""
    model  = _AllNegativeModel()
    loader = _make_loader(n=20, pos_ratio=0.5)
    _, f1, _ = evaluate_loader(model, loader, DEVICE)
    assert f1 == pytest.approx(0.0, abs=1e-4)


def test_evaluate_loader_loss_is_nonnegative():
    model  = _PerfectModel()
    loader = _make_loader()
    loss, _, _ = evaluate_loader(model, loader, DEVICE)
    assert loss >= 0.0


# ── full_report tests ─────────────────────────────────────────────────────

def test_full_report_keys():
    model  = _PerfectModel()
    loader = _make_loader()
    report = full_report(model, loader, DEVICE)
    required = {
        "loss", "f1", "auroc",
        "confusion_matrix", "classification_report",
        "all_preds", "all_labels", "all_probs",
    }
    assert required.issubset(set(report.keys()))


def test_full_report_confusion_matrix_shape():
    model  = _PerfectModel()
    loader = _make_loader(n=20)
    report = full_report(model, loader, DEVICE)
    cm = report["confusion_matrix"]
    assert cm.shape == (2, 2), f"Expected (2,2) confusion matrix, got {cm.shape}"


def test_full_report_pred_label_lengths_match():
    model  = _PerfectModel()
    loader = _make_loader(n=20)
    report = full_report(model, loader, DEVICE)
    assert len(report["all_preds"])  == 20
    assert len(report["all_labels"]) == 20
    assert len(report["all_probs"])  == 20


def test_full_report_probs_in_unit_interval():
    model  = _PerfectModel()
    loader = _make_loader(n=20)
    report = full_report(model, loader, DEVICE)
    probs = report["all_probs"]
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities must be in [0, 1]"
