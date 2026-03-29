"""
tests/test_preprocessing.py
----------------------------
Unit tests for data cleaning and splitting logic.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest
from data.preprocessing import clean, split


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_raw_df(rows: list) -> pd.DataFrame:
    """
    Build a minimal raw dataframe that mimics the Jigsaw schema.

    Each row is a dict with at least 'comment_text'; toxicity columns
    default to 0 unless supplied.
    """
    tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    records = []
    for r in rows:
        rec = {col: 0 for col in tox_cols}
        rec.update(r)
        records.append(rec)
    return pd.DataFrame(records)


# ── clean() tests ─────────────────────────────────────────────────────────

def test_clean_adds_label_column():
    df = _make_raw_df([{"comment_text": "hello", "toxic": 0}])
    out = clean(df)
    assert "label" in out.columns


def test_clean_toxic_label_is_1_when_any_subtag_set():
    df = _make_raw_df([{"comment_text": "bad", "insult": 1}])
    out = clean(df)
    assert out["label"].iloc[0] == 1


def test_clean_label_is_0_when_all_subtags_zero():
    df = _make_raw_df([{"comment_text": "nice", "toxic": 0}])
    out = clean(df)
    assert out["label"].iloc[0] == 0


def test_clean_drops_null_comments():
    df = _make_raw_df([
        {"comment_text": "hello"},
        {"comment_text": None},
    ])
    out = clean(df)
    assert len(out) == 1


def test_clean_drops_empty_string_comments():
    df = _make_raw_df([
        {"comment_text": "hello"},
        {"comment_text": "   "},   # whitespace-only
        {"comment_text": ""},
    ])
    out = clean(df)
    assert len(out) == 1


def test_clean_drops_exact_duplicates():
    df = _make_raw_df([
        {"comment_text": "hello"},
        {"comment_text": "hello"},   # duplicate
        {"comment_text": "world"},
    ])
    out = clean(df)
    assert len(out) == 2


def test_clean_output_columns():
    df = _make_raw_df([{"comment_text": "hi"}])
    out = clean(df)
    assert list(out.columns) == ["comment_text", "label"]


# ── split() tests ─────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """200-row balanced dataframe (100 toxic, 100 non-toxic)."""
    texts  = [f"comment {i}" for i in range(200)]
    labels = [i % 2 for i in range(200)]          # alternating 0/1
    return pd.DataFrame({"comment_text": texts, "label": labels})


def test_split_total_rows(clean_df):
    train, val, test = split(clean_df, sample_size=None, seed=42)
    assert len(train) + len(val) + len(test) == len(clean_df)


def test_split_ratios_approximate(clean_df):
    train, val, test = split(
        clean_df, sample_size=None,
        train_ratio=0.70, val_ratio=0.15, seed=42
    )
    n = len(clean_df)
    assert abs(len(train) / n - 0.70) < 0.03
    assert abs(len(val)   / n - 0.15) < 0.03
    assert abs(len(test)  / n - 0.15) < 0.03


def test_split_stratification_preserves_class_ratio(clean_df):
    """All splits should have approximately 50% toxic (like the source)."""
    train, val, test = split(clean_df, sample_size=None, seed=42)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        ratio = df["label"].mean()
        assert abs(ratio - 0.5) < 0.06, (
            f"{name} toxic ratio {ratio:.2f} deviates too far from 0.50"
        )


def test_split_no_overlap(clean_df):
    """Texts must not appear in more than one split."""
    train, val, test = split(clean_df, sample_size=None, seed=42)
    train_set = set(train["comment_text"])
    val_set   = set(val["comment_text"])
    test_set  = set(test["comment_text"])
    assert train_set.isdisjoint(val_set),  "Overlap between train and val"
    assert train_set.isdisjoint(test_set), "Overlap between train and test"
    assert val_set.isdisjoint(test_set),   "Overlap between val and test"


def test_split_sample_size_respected():
    """When sample_size < len(df), output total must equal sample_size."""
    texts  = [f"t{i}" for i in range(500)]
    labels = [i % 2 for i in range(500)]
    df = pd.DataFrame({"comment_text": texts, "label": labels})
    train, val, test = split(df, sample_size=100, seed=42)
    assert len(train) + len(val) + len(test) == 100


def test_split_deterministic(clean_df):
    """Same seed must produce identical splits."""
    t1, v1, te1 = split(clean_df, sample_size=None, seed=7)
    t2, v2, te2 = split(clean_df, sample_size=None, seed=7)
    assert list(t1["comment_text"]) == list(t2["comment_text"])
    assert list(v1["comment_text"]) == list(v2["comment_text"])
