"""
data/preprocessing.py
---------------------
Dataset loading, cleaning, and stratified splitting.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from utils.logger import get_logger

logger = get_logger(__name__)

# Toxicity sub-label columns present in the Jigsaw dataset
_TOXICITY_COLS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate",
]


def load_raw_dataframe(dataset_name: str = "jigsaw_toxicity_pred") -> pd.DataFrame:
    """
    Download the Jigsaw dataset via HuggingFace and return a DataFrame.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier.

    Returns
    -------
    pd.DataFrame
        Columns: comment_text + toxicity sub-labels.
    """
    logger.info("Loading dataset '%s' from HuggingFace Hub…", dataset_name)
    raw = load_dataset(dataset_name, split="train", trust_remote_code=True)
    df = raw.to_pandas()
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
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
