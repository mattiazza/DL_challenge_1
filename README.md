# Challenge 1 - Deep Learning Course

This repository contains my solution workspace for the first Deep Learning course challenge. The competition asks for a CNN trained from scratch to predict both facial attributes and celebrity identity from face images.

## Goal

The task is a multi-label, multi-class image classification problem:

- predict 8 binary facial attributes: `No_Beard`, `Young`, `Mouth_Slightly_Open`, `Smiling`, `Male`, `Wavy_Hair`, `Black_Hair`, `Wearing_Hat
- predict the celebrity identity among 500 + 1 classes, where the extra class represents `other`

The official score is the average of two metrics: the mean F1 over the binary attributes and the macro F1 over the celebrity classes. Full challenge rules are in [overview_challenge_1.md](overview_challenge_1.md).

## Constraints

The solution must respect the course rules:

- the model must be an explicitly coded CNN
- training must happen inside the notebook used for submission
- pre-trained models are not allowed
- external datasets are not allowed
- augmentation, dropout, early stopping, and other regularization methods are allowed and encouraged

## Main Files

- [Challenge1_MattiaPiazza.ipynb](Challenge1_MattiaPiazza.ipynb) is the main notebook with data loading, model definition, training, validation, and submission generation
- [overview_challenge_1.md](overview_challenge_1.md) contains the challenge description, rules, evaluation, and delivery instructions
- [specs/model.md](specs/model.md) defines the `MultiTaskCelebNet` architecture contract
- [specs/train.md](specs/train.md) defines the `train()` loop contract and metric semantics
- [specs/submission.md](specs/submission.md) defines the submission-generation contract
- [data/](data/) stores the competition data locally and is ignored by git
- [submissions/](submissions/) stores generated CSV files

## Data Layout

The expected local structure is:

```text
data/
├── train_data.csv
├── train_images/
└── test_images/
```

`train_data.csv` contains the image id plus the 9 label columns. The notebook uses the image ids to load the matching `.jpg` files from `train_images/` and `test_images/`.

## Notebook Workflow

The notebook is meant to be run end to end:

1. load and inspect the training data
2. define the `CelebDataset` dataset
3. build the CNN architecture from scratch
4. train and validate the model
5. evaluate on the validation split or test split
6. generate a Kaggle submission CSV in `submissions/`

## Local Setup

This project uses Python 3.12 with `uv`.

```bash
uv sync
uv run jupyter notebook
```

The Kaggle dataset can be downloaded via `kagglehub.competition_download('unipd-deep-learning-2026-challenge-1')`.

## Submission Format

The final output must be a CSV with the same column order as `train_data.csv`, using test image ids and predicted labels. The notebook writes the file to `submissions/`, and the final notebook version must be shared on Kaggle for evaluation.
