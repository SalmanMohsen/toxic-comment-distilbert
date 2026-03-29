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

2. HuggingFace parquet mirror — no Kaggle account needed, no legacy script.
   Uses ``datasets`` >= 2.19 with ``data_files`` pointing at a public parquet
   export, so the deprecated dataset-script path is never triggered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import get_logger

logger = get_logger(__name__)

# Toxicity sub-label columns present in the Jigsaw dataset
_TOXICITY_COLS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate",
]

# Public parquet mirror — does NOT use a dataset script, so it works with
# any modern version of the `datasets` library.
_HF_PARQUET_URL = (
    "https://huggingface.co/datasets/tdavidson/hate_speech_offensive"
    "/resolve/main/data/train.parquet"
)

# Canonical Jigsaw parquet hosted on HF (no script required)
_JIGSAW_PARQUET_URL = (
    "https://huggingface.co/datasets/lewtun/jigsaw-toxic-comments"
    "/resolve/main/data/train-00000-of-00001.parquet"
)


def load_raw_dataframe(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Jigsaw toxic comment dataset and return a raw DataFrame.

    Loading order
    -------------
    1. **Local CSV** (preferred) — if ``csv_path`` is supplied and the file
       exists, read it directly with pandas.  This is the fastest and most
       reliable path.  Download ``train.csv`` from Kaggle:
       https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

    2. **HuggingFace parquet** (fallback) — downloads a parquet snapshot that
       does *not* use a dataset script, so it is compatible with
       ``datasets >= 2.14`` without ``trust_remote_code``.

    Parameters
    ----------
    csv_path : str | None
        Path to a local ``train.csv`` (Kaggle download).  Pass ``None`` to
        use the HuggingFace parquet fallback.

    Returns
    -------
    pd.DataFrame
        Columns include ``comment_text`` plus the six toxicity sub-label
        columns.
    """
    # ── Path 1: local CSV ─────────────────────────────────────────────────
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
        logger.info("Loaded %d rows from CSV.", len(df))
        return df

    # ── Path 2: HuggingFace parquet (no legacy script) ───────────────────
    logger.info(
        "No csv_path provided — attempting HuggingFace parquet download…\n"
        "  Tip: for faster / offline runs, download train.csv from Kaggle\n"
        "  and pass --csv_path /path/to/train.csv"
    )
    try:
        from datasets import load_dataset  # local import — optional dependency

        ds = load_dataset(
            "parquet",
            data_files={"train": _JIGSAW_PARQUET_URL},
            split="train",
        )
        df = ds.to_pandas()
        logger.info("Loaded %d rows from HuggingFace parquet.", len(df))

        # The parquet mirror may use slightly different column names;
        # normalise to the expected schema.
        df = _normalise_columns(df)
        return df

    except Exception as exc:
        raise RuntimeError(
            "Could not load the dataset automatically.\n"
            f"Reason: {exc}\n\n"
            "Please download train.csv manually from Kaggle:\n"
            "  https://www.kaggle.com/competitions/"
            "jigsaw-toxic-comment-classification-challenge/data\n"
            "Then run:\n"
            "  python scripts/train.py --csv_path /path/to/train.csv"
        ) from exc


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns from alternative mirrors to the canonical Jigsaw schema.

    Canonical required columns:
      comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
    """
    rename_map: dict[str, str] = {}

    # Some mirrors use 'text' instead of 'comment_text'
    if "text" in df.columns and "comment_text" not in df.columns:
        rename_map["text"] = "comment_text"

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("Renamed columns: %s", rename_map)

    # Verify all required columns are present
    required = {"comment_text"} | set(_TOXICITY_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Please use the official Kaggle train.csv via --csv_path."
        )

    return df


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

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns ``comment_text`` and ``label``.
    """
    # Step 1 — binary label via OR-aggregation
    df = df.copy()
    df["label"] = (df[_TOXICITY_COLS].sum(axis=1) > 0).astype(int)

    # Step 2 — drop nulls / empty strings
    before = len(df)
    df = df[df["comment_text"].notna()]
    df["comment_text"] = df["comment_text"].str.strip()
    df = df[df["comment_text"].str.len() > 0]
    logger.info("Dropped %d null/empty rows.", before - len(df))

    # Step 3 — drop exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset="comment_text")
    logger.info("Dropped %d exact-duplicate rows.", before - len(df))

    toxic_rate = df["label"].mean() * 100
    logger.info(
        "Clean dataset: %d rows | toxic rate: %.1f%%", len(df), toxic_rate
    )
    return df[["comment_text", "label"]]


def split(
    df: pd.DataFrame,
    sample_size: int = 20_000,
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
    sample_size : int
        Number of rows to sample before splitting (for Colab free tier).
        Set to ``None`` to use the full dataset.
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

    # Optional sub-sample (stratified)
    if sample_size and sample_size < len(df):
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df["label"], random_state=seed
        )
        logger.info("Sub-sampled to %d rows.", len(df))

    # 70 / 30 first split, then 50/50 on the 30% to get val and test
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