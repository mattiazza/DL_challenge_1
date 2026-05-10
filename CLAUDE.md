# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.12, managed with **uv** (`pyproject.toml` + `uv.lock`)
- Virtual environment at `.venv/`

```bash
uv sync                  # install/update dependencies
uv run jupyter notebook  # launch notebook
```

## Project Overview

Kaggle competition: **unipd-deep-learning-2026-challenge-1**

Multi-label classification of celebrity facial attributes from images. Labels: `No_Beard`, `Young`, `Mouth_Slightly_Open`, `Smiling`, `Male`, `Wavy_Hair`, `Black_Hair`, `Wearing_Hat`, `celebrity_id` (501 classes: 500 celebrities + "other").

## Data Layout

Data lives in `data/` (gitignored, downloaded via `kagglehub`):
```
data/
├── train_data.csv      # id + 9 attribute columns
├── train_images/       # .jpg files
└── test_images/        # .jpg files for submission
```

Kaggle credentials go in `.env` (gitignored). Download snippet (commented out in notebook):
```python
from dotenv import load_dotenv; load_dotenv()
import kagglehub
path = kagglehub.competition_download("unipd-deep-learning-2026-challenge-1")
```

## Main Notebook

**`Challenge1_MattiaPiazza.ipynb`** — single file containing everything.

Global constants: `NUM_BINARY = 8`, `NUM_CLASSES = 501`.

### `CelebDataset` (PyTorch `Dataset`)
- Constructor: `CelebDataset(csv_file: Path, img_dir: Path, transform=None)`
  - Default transform: `ToTensor()` if none provided
- Loads 20,000 samples; split 80/20 via `random_split` → 16K train / 4K val
- Returns `(image_tensor: [3, 60, 48], labels: np.ndarray[9, float32])`
  - `labels[0:8]`: binary attributes (float32); `labels[8]`: celebrity_id (float32, cast to `.long()` for CE loss)

### `MultiTaskCelebNet` (dual-head CNN)
Constructor parameters:
```python
img_shape: tuple = (3, 60, 48)
conv_filters: list[int]       # output channels per conv layer
kernel_sizes: list[tuple]     # (H, W) per conv layer
max_pool_sizes: list[tuple]   # (H, W) per pool; (1,1) skips pooling
act_fs: list[Callable]        # activation per conv layer
fc_out: int = 2048            # FC bottleneck output dimension
fc_act: Callable = F.relu     # FC bottleneck activation
dp_rate: float = 0.5          # dropout rate after FC bottleneck
```
Architecture: shared conv backbone (all `padding='same'`) → flatten → `Linear → fc_act → Dropout` → two heads:
- `binary_head`: `Linear(fc_out, 8)` — raw logits, no sigmoid
- `class_head`: `Linear(fc_out, 501)` — raw logits, no softmax

`forward()` returns `(bin_logits: [B, 8], cls_logits: [B, 501])`.

### `train()`
```python
def train(
    model, dataloader_train, dataloader_val,
    optimizer=None,
    binary_criterion=BCEWithLogitsLoss(),
    class_criterion=CrossEntropyLoss(),
    epochs=30,
    hparam_tuning=False,
    cls_weight_fn=None,
    early_stopping_patience=None,
) -> tuple[list, list, list, list, list, list]
```
- Combined loss: `bin_loss + cls_w * cls_loss`, where `cls_w = cls_weight_fn(epoch, prev_bin_loss, prev_cls_loss)` if provided, else `1.0`
- `early_stopping_patience`: stops after N epochs with no val-loss improvement; restores best weights
- `hparam_tuning=True`: suppresses plots, prints metrics per epoch
- Binary accuracy: per-sample mean of 8 attribute predictions; classification accuracy: top-1 argmax
- Returns 6-tuple of per-epoch lists: `loss_train, bin_acc_train, cls_acc_train, loss_val, bin_acc_val, cls_acc_val`

### `test(model, dataloader)`
Prints combined loss, binary attribute accuracy, and celebrity ID accuracy. No default dataloader — must be passed explicitly.

### `plot_learning_acc_and_loss()`
```python
plot_learning_acc_and_loss(loss_tr, bin_acc_tr, cls_acc_tr, loss_val, bin_acc_val, cls_acc_val)
```
Renders 3 subplots (Loss, Binary Accuracy, Classification Accuracy) from the 6-tuple returned by `train()`.

### Submission cell
Generates `submissions/<name>.csv` using Polars, matching the `train_data.csv` column schema (`id` + attribute columns), with test image IDs (filename without `.jpg`).

## Specs

```
specs/
├── model.md   # MultiTaskCelebNet architecture spec (inputs, outputs, invariants, constructor params)
└── train.md   # train() function spec (pseudocode, loss weighting, early stopping, metrics)
```

These are the authoritative technical specs — read them before modifying the model or training loop.

## Submissions

Output CSVs go in `submissions/`. The format mirrors `train_data.csv` columns (`id` + attribute columns), with test image IDs (filename without `.jpg`).
