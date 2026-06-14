from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

label2id = {v: k for k, v in id2label.items()}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

print("Tokenizer and model loaded successfully!")