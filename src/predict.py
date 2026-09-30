import csv
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# --------------------------------------------------
# 1. FIND PROJECT DIRECTORIES
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "bert-intent-classifier"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "test.csv"
)


# --------------------------------------------------
# 2. LOAD TRAINED BERT MODEL
# --------------------------------------------------

print("Loading trained BERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# 3. PREDICT INTENT FOR ONE SENTENCE
# --------------------------------------------------

def predict_intent(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_id = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][predicted_id].item()

    predicted_intent = model.config.id2label[
        predicted_id
    ]

    return predicted_intent, confidence


# --------------------------------------------------
# 4. TEST WITH YOUR OWN SENTENCE
# --------------------------------------------------

print("\n--------------------------------")
print("CUSTOM PREDICTION")
print("--------------------------------")

user_text = input(
    "\nEnter a voice request (or type 'skip' to continue): "
)

if user_text.lower() != "skip":

    intent, confidence = predict_intent(user_text)

    print("\nInput:", user_text)
    print("Predicted intent:", intent)
    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )


# --------------------------------------------------
# 5. LOAD TEST DATASET
# --------------------------------------------------

print("\n--------------------------------")
print("TEST DATASET EVALUATION")
print("--------------------------------")

with open(
    TEST_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    test_rows = list(
        csv.DictReader(file)
    )


# --------------------------------------------------
# 6. PREDICT ALL TEST EXAMPLES
# --------------------------------------------------

true_labels = []
predicted_labels = []

print(
    f"\nTesting {len(test_rows)} unseen examples...\n"
)

for row in test_rows:

    text = row["text"]
    actual_intent = row["intent"]

    predicted_intent, confidence = predict_intent(
        text
    )

    true_labels.append(actual_intent)
    predicted_labels.append(predicted_intent)

    print("Text:", text)
    print("Actual:", actual_intent)
    print(
        "Predicted:",
        predicted_intent,
        f"({confidence * 100:.2f}%)"
    )
    print("-" * 60)


# --------------------------------------------------
# 7. CALCULATE METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )
)


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\n================================")
print("FINAL MODEL EVALUATION")
print("================================")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print("================================")