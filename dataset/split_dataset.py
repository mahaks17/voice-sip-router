import csv
import random
from collections import defaultdict

# Make the results reproducible
random.seed(42)

# Read the original dataset
with open("intents.csv", "r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

# Group examples by intent
groups = defaultdict(list)

for row in rows:
    groups[row["intent"]].append(row)

# Create empty datasets
train_rows = []
validation_rows = []
test_rows = []

# Split each intent separately
for intent, examples in groups.items():

    # Shuffle examples
    random.shuffle(examples)

    total = len(examples)

    # Calculate split sizes
    train_count = max(1, int(total * 0.70))
    validation_count = max(1, int(total * 0.15))

    # Remaining examples go to test
    test_count = total - train_count - validation_count

    # Make sure there is at least one test example
    if test_count < 1:
        test_count = 1
        train_count -= 1

    train_rows.extend(examples[:train_count])
    validation_rows.extend(
        examples[train_count:train_count + validation_count]
    )
    test_rows.extend(examples[train_count + validation_count:])


# Function to save a CSV file
def save_csv(filename, data):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "intent"])
        writer.writeheader()
        writer.writerows(data)


# Save the three datasets
save_csv("train.csv", train_rows)
save_csv("validation.csv", validation_rows)
save_csv("test.csv", test_rows)

print("Dataset split completed!")
print("Training examples:", len(train_rows))
print("Validation examples:", len(validation_rows))
print("Test examples:", len(test_rows))
print("Total examples:", len(train_rows) + len(validation_rows) + len(test_rows))