MODEL_NAME = "distilbert-base-uncased"

ID2LABEL = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}

NUM_LABELS = len(ID2LABEL)