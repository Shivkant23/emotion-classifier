import os
from transformers import pipeline

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english"
)

text = os.getenv("INPUT_TEXT")

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME
)

result = classifier(text)

print("\nPrediction:")
print(result)