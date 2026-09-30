import csv
import os

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# 1. Project paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_FILE = os.path.join(DATASET_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATASET_DIR, "validation.csv")

MODEL_NAME = "bert-base-uncased"


# -----------------------------
# 2. Load CSV files
# -----------------------------

def load_csv(filename):
    with open(filename, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


train_rows = load_csv(TRAIN_FILE)
validation_rows = load_csv(VALIDATION_FILE)

print("Training examples:", len(train_rows))
print("Validation examples:", len(validation_rows))


# -----------------------------
# 3. Create intent labels
# -----------------------------

intents = sorted(set(row["intent"] for row in train_rows))

label2id = {label: i for i, label in enumerate(intents)}
id2label = {i: label for label, i in label2id.items()}

print("\nIntent labels:")
for label, number in label2id.items():
    print(number, "=", label)


# -----------------------------
# 4. Convert intent names to numbers
# -----------------------------

for row in train_rows:
    row["label"] = label2id[row["intent"]]

for row in validation_rows:
    row["label"] = label2id[row["intent"]]


# -----------------------------
# 5. Convert to Hugging Face Dataset
# -----------------------------

train_dataset = Dataset.from_list(train_rows)
validation_dataset = Dataset.from_list(validation_rows)


# -----------------------------
# 6. Load BERT tokenizer
# -----------------------------

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -----------------------------
# 7. Tokenize text
# -----------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
validation_dataset = validation_dataset.map(tokenize_function, batched=True)


# -----------------------------
# 8. Load pretrained BERT
# -----------------------------

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(intents),
    id2label=id2label,
    label2id=label2id,
)


# -----------------------------
# 9. Training configuration
# -----------------------------

training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "model"),
    evaluation_strategy="epoch",   # fixed name
    save_strategy="epoch",         # save once per epoch
    save_total_limit=1,             # keep only latest checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=1,
    report_to="none",
)


# -----------------------------
# 10. Create Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)


# -----------------------------
# 11. Fine-tune BERT
# -----------------------------

print("\nStarting BERT fine-tuning...\n")

trainer.train()


# -----------------------------
# 12. Save the trained model
# -----------------------------

FINAL_MODEL_DIR = os.path.join(BASE_DIR, "model", "bert-intent-classifier")

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("\nTraining completed!")
print("Model saved to:", FINAL_MODEL_DIR)
