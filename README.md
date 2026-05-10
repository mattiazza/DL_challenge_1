# Challenge 1 - Deep Learning Course

This repository contains my solution workspace for the first Deep Learning course challenge. The project is based on the Kaggle competition described in `overview_challenge_1.md`, where the goal is to train a CNN from scratch to predict both facial attributes and celebrity identity from face images.

## Project Goal

The model must solve a multi-task image classification problem:

- predict 8 binary facial attributes such as `Smiling`, `Young`, `No_Beard` and `Wearing_Hat`
- predict the celebrity identity among 500 + 1 classes, where the extra class represents `other`

## Challenge Constraints

The project follows the rules of the course challenge:

- the network must be explicitly implemented as a CNN
- training must happen inside the notebook used for submission
- pre-trained models are not allowed
- external datasets are not allowed
- data augmentation and regularization techniques are allowed and encouraged

## Repository Structure

- [Challenge1_MattiaPiazza.ipynb](Challenge1_MattiaPiazza.ipynb) - main notebook with data loading, model definition, training, validation, and submission generation
- [overview_challenge_1.md](overview_challenge_1.md) - challenge description, rules, evaluation, and submission requirements
- [specs/model.md](specs/model.md) - model specification for the multi-task CNN
- [plans/model_plan.md](plans/model_plan.md) - implementation plan and debugging notes
- [data/](data/) - competition data directory, downloaded locally and ignored by git
- [submissions/](submissions/) - generated submission files

## Data Layout

The expected data structure is:

```text
data/
├── train_data.csv
├── train_images/
└── test_images/
```

`train_data.csv` contains the image id plus the label columns. The notebook uses the image ids to load the corresponding `.jpg` files from disk.

## Notebook Workflow

The notebook is intended to be executed end to end:

1. load and inspect the training data
2. define the `CelebDataset` class for image and label loading
3. build the CNN architecture from scratch
4. train and validate the model
5. evaluate on the validation or test split
6. generate a Kaggle submission file in `submissions/`

## Local Setup

This project uses Python 3.12 with `uv`.

```bash
uv sync
uv run jupyter notebook
```

The Kaggle dataset can be downloaded via kagglehub.competition_download('unipd-deep-learning-2026-challenge-1')

## Submission Format

The final output must be a CSV file with the same column structure as `train_data.csv`, using the test image ids and the predicted labels for each row. The submission file should be written to the `submissions/` folder.

## Notes

- training and evaluation should be performed on the notebook used for submission
- the final notebook must be shared on Kaggle for verification
