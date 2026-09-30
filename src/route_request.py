import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Find the project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python's import path
sys.path.append(BASE_DIR)

# Import our SIP router
from sip.router import route_intent


# Location of the trained BERT model
MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "bert-intent-classifier"
)


print("Loading trained BERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

print("BERT model loaded successfully!")


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

    predicted_intent = model.config.id2label[predicted_id]

    return predicted_intent, confidence


def process_request(text):

    # Step 1: BERT predicts the intent
    intent, confidence = predict_intent(text)

    # Step 2: Intent is mapped to a SIP endpoint
    endpoint = route_intent(intent)

    return intent, confidence, endpoint


if __name__ == "__main__":

    print("\n================================")
    print("VOICE REQUEST → SIP ROUTER")
    print("================================")

    user_text = input(
        "\nEnter a voice request: "
    )

    intent, confidence, endpoint = process_request(
        user_text
    )

    print("\n--------------------------------")
    print("ROUTING RESULT")
    print("--------------------------------")

    print("Request:", user_text)

    print(
        "Predicted Intent:",
        intent
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    print(
        "SIP Endpoint:",
        endpoint
    )

    print("--------------------------------")