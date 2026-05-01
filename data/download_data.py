from datasets import load_dataset
import json
import os

print("Downloading MedQA dataset...")

dataset = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)

print("Dataset loaded")
print(dataset)

print("\n--- Sample ---")
sample = dataset['train'][0]
for key, value in sample.items():
    print(f"{key}: {value}")

os.makedirs("data/raw", exist_ok=True)

dataset['train'].to_json("data/raw/medqa_train.json")
dataset['test'].to_json("data/raw/medqa_test.json")

print(f"\nSaved {len(dataset['train'])} train / {len(dataset['test'])} test samples")