# Emotion Classification Dataset Preparation

## Overview

This project prepares the Emotion dataset from Hugging Face for training a text classification model.

Dataset:
https://huggingface.co/datasets/dair-ai/emotion

The dataset contains English text samples labeled with one of six emotions:

* sadness
* joy
* love
* anger
* fear
* surprise

## Dataset Statistics

| Split      | Samples |
| ---------- | ------- |
| Train      | 16,000  |
| Validation | 2,000   |
| Test       | 2,000   |
| Total      | 20,000  |

## Data Preparation Steps

The preparation script performs the following operations:

1. Download the dataset from Hugging Face.
2. Inspect dataset size, structure, and class distribution.
3. Remove missing values.
4. Remove duplicate records.
5. Convert text to lowercase.
6. Remove punctuation.
7. Remove extra whitespace.
8. Encode labels using an id-to-label mapping.
9. Save cleaned datasets locally.

## Label Mapping

| ID | Label    |
| -- | -------- |
| 0  | sadness  |
| 1  | joy      |
| 2  | love     |
| 3  | anger    |
| 4  | fear     |
| 5  | surprise |

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Data Preparation

```bash
python prepare_data.py
```

## Generated Files

```text
raw/
├── train.csv
├── validation.csv
└── test.csv

processed/
├── train_clean.csv
├── validation_clean.csv
└── test_clean.csv

id2label.json
```

## Repository Structure

```text
emotion-mlops/
│
├── prepare_data.py
├── requirements.txt
├── README.md
├── id2label.json
│
├── raw/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
└── processed/
    ├── train_clean.csv
    ├── validation_clean.csv
    └── test_clean.csv
```

## Git Ignore

Large dataset files are excluded from version control.

```gitignore
raw/
processed/
*.csv
*.parquet
```

## Author

MLOps Pipeline Project – Task 2: Data Preparation & Normalisation

## URL:

Wandb url: https://wandb.ai/g25ait2102-iit/emotion-classification

Kaggle url: https://www.kaggle.com/code/shivkantsawarkar/notebook-project

HuggingFace url: https://huggingface.co/Shubham231/distilbert_v1

# Accuracy: 94.22