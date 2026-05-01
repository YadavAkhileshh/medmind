import json

with open("data/processed/train_instructions.json") as f:
    data = json.load(f)

print(f"Before: {len(data)} samples")

seen_questions = set()
cleaned = []

for item in data:
    if item['question'] in seen_questions:
        continue
    if len(item['text'].split()) > 600:
        continue
    seen_questions.add(item['question'])
    cleaned.append(item)

print(f"After:  {len(cleaned)} samples")
print(f"Removed: {len(data) - len(cleaned)}")

with open("data/processed/train_instructions.json", "w") as f:
    json.dump(cleaned, f, indent=2)

print("Clean data saved")