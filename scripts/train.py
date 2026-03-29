#!/usr/bin/env python
"""
scripts/train.py
----------------
CLI entry point for training the Toxic Comment Classifier.

Usage examples
--------------
# Recommended: point at the Kaggle CSV directly
python scripts/train.py --csv_path /path/to/train.csv

# No CSV? Falls back to HuggingFace parquet (requires internet)
python scripts/train.py

# Override hyperparameters
python scripts/train.py --csv_path /path/to/train.csv --lr 3e-5 --batch_size 8

Full flag reference: python scripts/train.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from data.dataset       import ToxicDataset
from data.preprocessing import load_raw_dataframe, clean, split
from evaluation         import full_report, get_false_negatives, plot_confusion_matrix, plot_loss_curves
from models.classifier  import build_model, build_tokenizer, get_device
from training.trainer   import Trainer
from utils              import ExperimentLogger, get_logger, load_checkpoint, load_config, set_all_seeds

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on the Jigsaw toxic comment dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default=str(REPO_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help=(
            "Path to the Kaggle train.csv. "
            "Download from: https://www.kaggle.com/competitions/"
            "jigsaw-toxic-comment-classification-challenge/data  "
            "Omit to use the HuggingFace parquet fallback (requires internet)."
        ),
    )
    parser.add_argument("--model_name",     type=str,   default=None)
    parser.add_argument("--max_len",        type=int,   default=None)
    parser.add_argument("--sample_size",    type=int,   default=None,
                        help="Rows to sample. 10000 is safe for 2 GB VRAM.")
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--batch_size",     type=int,   default=None,
                        help="16 for 2 GB VRAM (940MX); 32 for Colab T4.")
    parser.add_argument("--lr",             type=float, default=None, dest="learning_rate")
    parser.add_argument("--weight_decay",   type=float, default=None)
    parser.add_argument("--warmup_ratio",   type=float, default=None)
    parser.add_argument("--grad_clip",      type=float, default=None)
    parser.add_argument("--patience",       type=int,   default=None)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--checkpoint_dir", type=str,   default=None)
    return parser.parse_args()


def merge_config(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI flags override YAML values; YAML overrides built-in defaults."""
    flat = {
        **cfg.get("model",    {}),
        **cfg.get("data",     {}),
        **cfg.get("training", {}),
        **cfg.get("logging",  {}),
    }
    overrides = {
        "csv_path":       args.csv_path,
        "model_name":     args.model_name,
        "max_len":        args.max_len,
        "sample_size":    args.sample_size,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  args.learning_rate,
        "weight_decay":   args.weight_decay,
        "warmup_ratio":   args.warmup_ratio,
        "grad_clip":      args.grad_clip,
        "patience":       args.patience,
        "seed":           args.seed,
        "checkpoint_dir": args.checkpoint_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            flat[key] = value
    return flat


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    c    = merge_config(cfg, args)

    logger.info("=" * 60)
    logger.info("Toxic Comment Classifier — Training Run")
    logger.info("Config  : %s", args.config)
    logger.info("CSV path: %s", c.get("csv_path") or "(HuggingFace parquet fallback)")
    logger.info("=" * 60)

    # 1. Reproducibility
    set_all_seeds(c["seed"])

    # 2. Device
    device = get_device()

    # 3. Data
    raw_df   = load_raw_dataframe(csv_path=c.get("csv_path"))
    clean_df = clean(raw_df)
    train_df, val_df, test_df = split(
        clean_df,
        sample_size=c["sample_size"],
        train_ratio=c["train_ratio"],
        val_ratio=c["val_ratio"],
        seed=c["seed"],
    )

    # 4. Tokeniser & Datasets
    tokenizer = build_tokenizer(c["name"])
    train_ds = ToxicDataset(train_df["comment_text"], train_df["label"], tokenizer, c["max_len"])
    val_ds   = ToxicDataset(val_df["comment_text"],   val_df["label"],   tokenizer, c["max_len"])
    test_ds  = ToxicDataset(test_df["comment_text"],  test_df["label"],  tokenizer, c["max_len"])

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=c["batch_size"], shuffle=False, num_workers=0, pin_memory=pin)

    logger.info("DataLoaders: train=%d | val=%d | test=%d batches",
                len(train_loader), len(val_loader), len(test_loader))

    # 5. Model
    model = build_model(c["name"], num_labels=c["num_labels"])

    # 6. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=c["epochs"],
        learning_rate=c["learning_rate"],
        weight_decay=c["weight_decay"],
        warmup_ratio=c["warmup_ratio"],
        grad_clip=c["grad_clip"],
        patience=c["patience"],
        checkpoint_dir=c["checkpoint_dir"],
        log_every_n_steps=c["log_every_n_steps"],
    )
    result = trainer.fit()

    # 7. Test evaluation
    model = load_checkpoint(model, result.best_checkpoint, device)
    report = full_report(model, test_loader, device)
    logger.info("TEST → loss=%.4f | F1=%.4f | AUROC=%.4f",
                report["loss"], report["f1"], report["auroc"])

    # 8. Error analysis
    fns = get_false_negatives(
        texts=list(test_df["comment_text"]),
        true_labels=report["all_labels"],
        pred_labels=report["all_preds"],
    )
    for text, _, _ in fns[:3]:
        logger.info("  FN sample » %s", text[:120])

    plot_confusion_matrix(report["all_labels"], report["all_preds"], save_path="confusion_matrix.png")
    plot_loss_curves(result.history, save_path="loss_curves.png")

    # 9. Experiment log
    exp_record = {
        "run_id":          f"{c['name']}-lr{c['learning_rate']}-ep{c['epochs']}-seed{c['seed']}",
        "model":           c["name"],
        "lr":              c["learning_rate"],
        "batch_size":      c["batch_size"],
        "max_len":         c["max_len"],
        "epochs_run":      result.best_epoch,
        "best_val_f1":     round(result.best_val_f1, 4),
        "test_f1":         round(report["f1"], 4),
        "test_auroc":      round(report["auroc"], 4),
        "checkpoint":      str(result.best_checkpoint),
        "elapsed_minutes": round(result.elapsed_minutes, 1),
    }
    ExperimentLogger(c["experiment_log"]).log(exp_record)
    logger.info("Experiment record:\n%s", json.dumps(exp_record, indent=2))
    logger.info("Done.")


if __name__ == "__main__":
    main()