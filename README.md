# Toxic Comment Classifier

Binary text classification — **toxic vs. non-toxic** — using fine-tuned `distilbert-base-uncased`.

| Metric | Score |
|--------|-------|
| Test F1 | **0.963** |
| Test AUROC | **0.9793** |
| Test Loss | 0.1490 |
| Precision (toxic) | 0.97 |
| Recall (toxic) | 0.96 |

Trained on [`tdavidson/hate_speech_offensive`](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) · 10 000-sample stratified subset · 36 min on NVIDIA 940MX (2 GB VRAM).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Model & Training](#model--training)
- [Overfitting Mitigations](#overfitting-mitigations)
- [Results](#results)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Project Structure

```
toxic-comment-distilbert/
├── configs/
│   └── default.yaml          # All hyperparameters in one place
├── src/
│   ├── data/
│   │   ├── dataset.py        # ToxicDataset (PyTorch Dataset)
│   │   └── preprocessing.py  # Load → clean → split pipeline
│   ├── models/
│   │   └── classifier.py     # build_model / build_tokenizer factories
│   ├── training/
│   │   ├── trainer.py        # Training loop + early stopping
│   │   └── scheduler.py      # AdamW + linear warmup scheduler
│   ├── evaluation/
│   │   ├── metrics.py        # evaluate_loader, full_report
│   │   └── error_analysis.py # False-negative inspection, plots
│   └── utils/
│       ├── seed.py           # Reproducibility — seeds all libraries
│       ├── logger.py         # Coloured console logger + JSON experiment log
│       └── io.py             # Checkpoint save/load, config loading
├── tests/
│   ├── test_dataset.py       # 7 unit tests (mock tokenizer)
│   ├── test_preprocessing.py # 12 unit tests (clean + split logic)
│   └── test_metrics.py       # 8 unit tests (mock model, no GPU needed)
├── scripts/
│   └── train.py              # CLI entry point
├── requirements.txt
├── setup.py
└── conftest.py
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/SalmanMohsen/toxic-comment-distilbert.git
cd toxic-comment-distilbert
pip install -r requirements.txt
# or: pip install -e .
```

### 2. Train

**Option A — HuggingFace dataset (no download needed)**
```bash
python scripts/train.py
```
Automatically downloads `tdavidson/hate_speech_offensive` via the HuggingFace Hub.

**Option B — Kaggle CSV (faster, works offline)**
```bash
python scripts/train.py --csv_path /path/to/train.csv
```
Download `train.csv` from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).

### 3. Override any hyperparameter via CLI

```bash
python scripts/train.py --lr 3e-5 --batch_size 8 --epochs 5 --classifier_dropout 0.4
```

See all flags:
```bash
python scripts/train.py --help
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [`tdavidson/hate_speech_offensive`](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) |
| Full size | 24 783 tweets |
| Training subset | 10 000 (stratified sample, seed=42) |
| License | MIT |
| Label mapping | `class ∈ {0,1}` → toxic=1, `class=2` → toxic=0 |
| Toxic rate | ~77% |
| Split | 70 / 15 / 15 (stratified) |

**Cleaning steps applied:**
1. Collapse the three-class label into binary via OR-mapping.
2. Drop null and whitespace-only tweets.
3. Remove exact-duplicate tweets to prevent data leakage across splits.

> **Known bias:** The dataset has documented racial annotation bias — tweets containing African-American Vernacular English (AAVE) were disproportionately labelled as hate speech by crowd annotators. See [Wiegand et al. (2019)](https://aclanthology.org/W19-3504/) for details.

---

## Model & Training

**Architecture:** `distilbert-base-uncased` — 66M parameters, 40% faster than BERT-base with ~97% of its performance on classification tasks.

**Loss function:**

```
L = -(1/N) · Σᵢ Σ_{c∈{0,1}} y_{i,c} · log( softmax(W·hᵢ + b)_c )
```

where `hᵢ` is the `[CLS]` token embedding output by DistilBERT.

**Key training decisions:**

| Decision | Value | Reason |
|----------|-------|--------|
| Optimizer | AdamW | Standard for transformer fine-tuning; includes decoupled L2 |
| Learning rate | 2e-5 | Best of grid search [1e-5, 2e-5, 3e-5, 5e-5] |
| Backbone LR | 2e-6 (÷10) | Differential LR prevents catastrophic forgetting |
| Weight decay | 0.01 | L2 regularisation on all parameters |
| Warmup ratio | 10% | Linear warmup over first 10% of steps |
| Gradient clip | 1.0 | Prevents exploding gradients in early fine-tuning |
| Batch size | 16 | Safe for 2 GB VRAM; use 32 on Colab T4 |
| Max seq len | 128 | Covers ~95% of tweets; 512 would OOM on 940MX |

---

## Overfitting Mitigations

An early run showed classic overfitting: train loss fell to 0.051 while val loss rose from 0.129 → 0.161 across 3 epochs. Three fixes were applied:

**1. Early stopping on `val_loss` (not val F1)**
Val F1 can inch upward even as the model memorises the training set, because the metric is insensitive to calibration. Watching val loss catches divergence one epoch earlier.

**2. Differential learning rates**
The DistilBERT backbone is trained at `lr / 10 = 2e-6`; the classification head at the full `2e-5`. Slowing the backbone preserves pre-trained representations that would otherwise be overwritten on a 10k-sample dataset.

**3. Classifier dropout 0.2 → 0.3**
DistilBERT's pre-classifier dropout is raised from the default 0.2 to 0.3, adding regularisation noise to the head without touching the backbone. Configurable via `--classifier_dropout`.

**Result:** val loss converged from above (0.184 → 0.150), train/val gap closed from +0.17 to +0.018.

---

## Results

### Test set metrics

```
              precision  recall  f1-score  support
   non-toxic       0.81    0.83      0.82      252
       toxic       0.97    0.96      0.96     1248
    accuracy                         0.94     1500
   macro avg       0.89    0.90      0.89     1500
weighted avg       0.94    0.94      0.94     1500

TEST → loss=0.1490 | F1=0.9630 | AUROC=0.9793
```

### Confusion matrix

```
                Predicted
                Non-toxic   Toxic
Actual Non-toxic    210       42
       Toxic         50     1198
```

- **False Negatives (toxic missed): 50** — mostly implicit hostility with no explicit slurs
- **False Positives (clean flagged): 42**

### Loss curves

Train and val loss converged by epoch 3 with no divergence, confirming the overfitting mitigations were effective.

### Experiment log

```json
{
  "run_id": "distilbert-base-uncased-lr2e-05-ep3-seed42",
  "model": "distilbert-base-uncased",
  "lr": 2e-05,
  "batch_size": 16,
  "max_len": 128,
  "epochs_run": 3,
  "best_val_f1": 0.9628,
  "test_f1": 0.963,
  "test_auroc": 0.9793,
  "checkpoint": "checkpoints/checkpoint_epoch3_f10.9628.pt",
  "elapsed_minutes": 36.2
}
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Any value can be overridden from the CLI without editing the file.

```yaml
model:
  name: "distilbert-base-uncased"
  num_labels: 2
  max_len: 128

data:
  csv_path: null          # null → HuggingFace fallback
  sample_size: 10000      # increase to 20000 on Colab T4
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

training:
  seed: 42
  epochs: 5
  batch_size: 16          # use 32 on Colab T4
  learning_rate: 2.0e-5
  backbone_lr_factor: 10.0
  weight_decay: 0.01
  warmup_ratio: 0.10
  grad_clip: 1.0
  classifier_dropout: 0.3
  patience: 2
  checkpoint_dir: "checkpoints"
```

---

## Running Tests

No GPU or model download required — all tests use mocks.

```bash
pytest tests/ -v
```

```
tests/test_dataset.py        7 passed
tests/test_preprocessing.py 12 passed
tests/test_metrics.py        8 passed
```

**What is tested:**
- `test_dataset.py` — tensor shapes, label dtype, tokenizer called with correct args
- `test_preprocessing.py` — null/empty/duplicate dropping, stratified split ratios, no overlap between splits, determinism
- `test_metrics.py` — perfect-model gives F1=1.0, all-negative model gives F1=0.0, confusion matrix shape, probabilities in [0,1]

---

## Reproducibility

All randomness is controlled via a single `seed=42` in `configs/default.yaml`, applied to:

| Library | Call |
|---------|------|
| Python stdlib | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(42)` |
| OS hash | `PYTHONHASHSEED=42` |
| cuDNN | `deterministic=True`, `benchmark=False` |
| DataLoader | `num_workers=0` (single process) |

**Remaining nondeterminism:** cuDNN atomic operations inside the scaled-dot-product attention kernel cannot be fully eliminated without a prohibitive speed penalty on some CUDA versions.

---

## License

This project is **MIT licensed**.

| Component | License |
|-----------|---------|
| This repo | MIT |
| `distilbert-base-uncased` weights | Apache 2.0 |
| `tdavidson/hate_speech_offensive` dataset | MIT |
| HuggingFace Transformers | Apache 2.0 |
| scikit-learn | BSD-3 |
