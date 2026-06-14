import os
from transformers import pipeline

hf_token = os.getenv("HF_TOKEN")
MODEL_NAME = "distilbert-base-uncased"


classifier = pipeline(
    "text-classification",
    model="Shubham231/distilbert_v1",
    token=hf_token,
    tokenizer=MODEL_NAME
)

text = os.getenv("INPUT_TEXT", "I am happy")

print(classifier(text))