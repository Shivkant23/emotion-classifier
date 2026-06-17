import os
import re
import json
from datasets import load_dataset

# ==================================================
# Create folders
# ==================================================

os.makedirs("raw", exist_ok=True)
os.makedirs("processed", exist_ok=True)

# ==================================================
# Download dataset
# ==================================================

print("Loading dair-ai/emotion dataset...")

dataset = load_dataset("dair-ai/emotion")

train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()

# Save raw data
train_df.to_csv("raw/train.csv", index=False)
val_df.to_csv("raw/validation.csv", index=False)
test_df.to_csv("raw/test.csv", index=False)

# ==================================================
# Dataset Inspection
# ==================================================

print("\n===== DATASET INSPECTION =====")

print(f"Train Shape: {train_df.shape}")
print(f"Validation Shape: {val_df.shape}")
print(f"Test Shape: {test_df.shape}")

print("\nColumns:")
print(train_df.columns.tolist())

print("\nMissing Values:")
print(train_df.isnull().sum())

print("\nDuplicate Rows:")
print(train_df.duplicated().sum())

print("\nClass Distribution:")
print(train_df["label"].value_counts().sort_index())

# ==================================================
# Label Mapping
# ==================================================

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

label2id = {v: k for k, v in id2label.items()}

# ==================================================
# Save id2label.json
# ==================================================

with open("id2label.json", "w") as f:
    json.dump(
        {str(k): v for k, v in id2label.items()},
        f,
        indent=4
    )

print("\nid2label.json created")

# ==================================================
# Text Cleaning
# ==================================================



def clean_text(text):
    text = str(text)

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==================================================
# Process Function
# ==================================================



def preprocess(df):

    before_rows = len(df)

    df = df.dropna()

    df = df.drop_duplicates()

    df["text"] = df["text"].apply(clean_text)

    df["label_name"] = df["label"].map(id2label)

    after_rows = len(df)

    print(
        f"Removed {before_rows - after_rows} rows"
    )

    return df



# ==================================================
# Preprocess
# ==================================================



print("\nCleaning train data...")
train_df = preprocess(train_df)

print("Cleaning validation data...")
val_df = preprocess(val_df)

print("Cleaning test data...")
test_df = preprocess(test_df)

# ==================================================
# Save Processed Data
# ==================================================

train_df.to_csv(
    "processed/train_clean.csv",
    index=False
)

val_df.to_csv(
    "processed/validation_clean.csv",
    index=False
)

test_df.to_csv(
    "processed/test_clean.csv",
    index=False
)

print("\nProcessed files saved.")

# ==================================================
# Summary
# ==================================================

print("\n===== FINAL SUMMARY =====")

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

print("\nSample Record:")
print(train_df.head(3))

