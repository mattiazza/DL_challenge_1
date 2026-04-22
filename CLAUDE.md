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
data/competitions/unipd-deep-learning-2026-challenge-1/
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

**`Challenge1_MattiaPiazza.ipynb`** — single file containing everything:

- **`CelebDataset`** (PyTorch `Dataset`): loads images from disk + labels from CSV via Polars. Returns `(image_tensor, labels_float32_array)`.
- **`My_Convolutional_Network`** (configurable CNN): accepts lists of `conv_filters`, `kernel_sizes`, `max_pool_sizes`, `act_fs`. All conv layers use `padding='same'`.
- **`train()`**: training loop with `hiddenlayer` live plotting; set `hparam_tuning=True` to suppress plots and print metrics instead.
- **`test()`**: evaluation on a dataloader.
- Submission cell: generates `submissions/<name>.csv` using Polars, matching the `train_data.csv` column schema.

## Submissions

Output CSVs go in `submissions/`. The format mirrors `train_data.csv` columns (`id` + attribute columns), with test image IDs (filename without `.jpg`).
