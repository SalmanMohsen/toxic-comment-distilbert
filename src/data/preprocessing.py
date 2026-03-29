"""
data/preprocessing.py
---------------------
Dataset loading, cleaning, and stratified splitting.

Loading strategy (tried in order)
----------------------------------
1. Local CSV  — fastest; pass ``csv_path`` to :func:`load_raw_dataframe`.
   Download train.csv from:
   https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
   and place it anywhere, then point the config / CLI at it.

2. HuggingFace — ``tdavidson/hate_speech_offensive``.
   Schema: columns are ``tweet`` (text) and ``class`` (int).
   class values:  0 = hate speech, 1 = offensive, 2 = neither
   We map  0 and 1  →  toxic (label=1),  2  →  non-toxic (label=0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Jigsaw schema (used when loading from local CSV) ─────────────────────
_TOXICITY_COLS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate",
]

# ── tdavidson/hate_speech_offensive schema ────────────────────────────────
# Column names exactly as they appear in the HuggingFace dataset
_DAVIDSON_TEXT_COL  = "tweet"
_DAVIDSON_LABEL_COL = "class"
# class integers: 0 = hate speech, 1 = offensive language, 2 = neither
_DAVIDSON_TOXIC_CLASSES = {0, 1}   # both map to label=1 (toxic)


def load_raw_dataframe(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the dataset and return a raw DataFrame.

    Loading order
    -------------
    1. **Local CSV** (preferred) — if ``csv_path`` is supplied and the file
       exists, read it directly with pandas.  This is the fastest and most
       reliable path.  Expected schema: Jigsaw ``train.csv`` with columns
       ``comment_text``, ``toxic``, ``severe_toxic``, etc.

    2. **tdavidson/hate_speech_offensive** (fallback) — loaded via
       ``load_dataset(..., split="train")`` which returns a single
       ``Dataset`` object (not a ``DatasetDict``), so ``.to_pandas()``
       works directly.  Columns are normalised in :func:`_normalise_columns`.

    Parameters
    ----------
    csv_path : str | None
        Path to a local Kaggle ``train.csv``.  Pass ``None`` to use the
        HuggingFace fallback.

    Returns
    -------
    pd.DataFrame
        Columns ``comment_text`` and ``label`` (binary: 1 = toxic).
    """
    # ── Path 1: local Jigsaw CSV ──────────────────────────────────────────
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"CSV not found at '{csv_path}'.\n"
                "Download train.csv from Kaggle:\n"
                "  https://www.kaggle.com/competitions/"
                "jigsaw-toxic-comment-classification-challenge/data\n"
                "Then pass --csv_path /path/to/train.csv"
            )
        logger.info("Loading local CSV: %s", path)
        df = pd.read_csv(path)
        logger.info("Loaded %d rows from CSV. Columns: %s", len(df), list(df.columns))
        return df   # caller passes this to clean() which handles Jigsaw schema

    # ── Path 2: tdavidson/hate_speech_offensive ───────────────────────────
    logger.info(
        "No csv_path provided — loading 'tdavidson/hate_speech_offensive' "
        "from HuggingFace Hub…"
    )
    try:
        from datasets import load_dataset  # local import — optional dependency

        # Specify split="train" explicitly so load_dataset returns a Dataset
        # (not a DatasetDict). Without this, ds is a DatasetDict and calling
        # ds.to_pandas() raises AttributeError.
        ds = load_dataset("tdavidson/hate_speech_offensive", split="train")

        # ds is now a datasets.Dataset — .to_pandas() is valid
        df = ds.to_pandas()
        logger.info(
            "Loaded %d rows from HuggingFace. Columns: %s",
            len(df), list(df.columns),
        )

        # Map dataset-specific columns to the canonical schema used by clean()
        df = _normalise_columns(df)
        return df

    except Exception as exc:
        raise RuntimeError(
            "Could not load 'tdavidson/hate_speech_offensive' from HuggingFace.\n"
            f"Reason: {exc}\n\n"
            "Alternatives:\n"
            "  1. Check your internet connection and try again.\n"
            "  2. Download Jigsaw train.csv from Kaggle and run:\n"
            "       python scripts/train.py --csv_path /path/to/train.csv"
        ) from exc


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the ``tdavidson/hate_speech_offensive`` schema into the
    canonical schema expected by :func:`clean`.

    tdavidson schema
    ----------------
    - ``tweet``  : raw text (maps to ``comment_text``)
    - ``class``  : int — 0 = hate speech, 1 = offensive, 2 = neither
    - other cols : count columns (hate_speech_count, etc.) — discarded

    Output schema
    -------------
    - ``comment_text`` : str
    - ``toxic``        : int  (1 if class ∈ {0, 1}, else 0)
    - all other ``_TOXICITY_COLS`` set to 0 (clean() will OR-aggregate them)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Contains ``comment_text`` and the six Jigsaw toxicity columns so
        that :func:`clean` works without modification.
    """
    # Validate expected source columns exist
    for col in (_DAVIDSON_TEXT_COL, _DAVIDSON_LABEL_COL):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in dataset.\n"
                f"Available columns: {list(df.columns)}\n"
                "The dataset schema may have changed. "
                "Use --csv_path with the Kaggle train.csv instead."
            )

    out = pd.DataFrame()

    # Map 'tweet' → 'comment_text'
    out["comment_text"] = df[_DAVIDSON_TEXT_COL].astype(str)

    # Map 'class' → binary 'toxic' column
    # class 0 (hate speech) and class 1 (offensive) → toxic=1
    # class 2 (neither)                              → toxic=0
    out["toxic"] = df[_DAVIDSON_LABEL_COL].isin(_DAVIDSON_TOXIC_CLASSES).astype(int)

    # Fill remaining Jigsaw sub-label columns with 0.
    # clean() computes label = OR(all _TOXICITY_COLS), so only 'toxic'
    # needs to carry signal; the rest can be zero.
    for col in _TOXICITY_COLS:
        if col not in out.columns:
            out[col] = 0

    logger.info(
        "Normalised tdavidson schema → %d rows | toxic rate: %.1f%%",
        len(out),
        out["toxic"].mean() * 100,
    )
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data cleaning steps and add a binary ``label`` column.

    Steps
    -----
    1. Collapse the six toxicity sub-labels into one binary flag
       (label = 1 if *any* sub-label is 1).
    2. Drop rows where ``comment_text`` is null or empty after stripping.
    3. Remove exact-duplicate comments to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from :func:`load_raw_dataframe`.
        Must contain ``comment_text`` and the six ``_TOXICITY_COLS``.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with exactly two columns: ``comment_text``, ``label``.
    """
    df = df.copy()

    # Step 1 — binary label via OR-aggregation across all toxicity sub-labels
    df["label"] = (df[_TOXICITY_COLS].sum(axis=1) > 0).astype(int)

    # Step 2 — drop nulls / whitespace-only comments
    before = len(df)
    df = df[df["comment_text"].notna()]
    df["comment_text"] = df["comment_text"].str.strip()
    df = df[df["comment_text"].str.len() > 0]
    logger.info("Dropped %d null/empty rows.", before - len(df))

    # Step 3 — drop exact-duplicate comments (prevents data leakage)
    before = len(df)
    df = df.drop_duplicates(subset="comment_text")
    logger.info("Dropped %d exact-duplicate rows.", before - len(df))

    toxic_rate = df["label"].mean() * 100
    logger.info("Clean dataset: %d rows | toxic rate: %.1f%%", len(df), toxic_rate)

    return df[["comment_text", "label"]]


def split(
    df: pd.DataFrame,
    sample_size: Optional[int] = 10_000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split with optional sub-sampling.

    The test ratio is derived as ``1 - train_ratio - val_ratio``.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (must have ``label`` column).
    sample_size : int | None
        Number of rows to sample before splitting.
        Set to ``None`` to use the full dataset.
        Default 10 000 is safe for 2 GB VRAM (940MX).
    train_ratio : float
    val_ratio : float
    seed : int

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    # Optional stratified sub-sample
    if sample_size and sample_size < len(df):
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df["label"], random_state=seed
        )
        logger.info("Sub-sampled to %d rows.", len(df))

    # Two-stage stratified split: 70 / 15 / 15
    val_test_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df, test_size=val_test_size, stratify=df["label"], random_state=seed
    )
    relative_test = test_ratio / val_test_size
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df["label"], random_state=seed
    )

    logger.info(
        "Split → train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df