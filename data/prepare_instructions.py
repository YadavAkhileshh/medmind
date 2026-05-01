import json
import os

print("Loading raw data...")

with open("data/raw/medqa_train.json", "r") as f:
    train_data = [json.loads(line) for line in f]

with open("data/raw/medqa_test.json", "r") as f:
    test_data = [json.loads(line) for line in f]

print(f"Train: {len(train_data)} | Test: {len(test_data)}")


def format_options(options: dict) -> str:
    """{'A': 'Ampicillin', ...} -> 'A: Ampicillin\nB: ...'"""
    return "\n".join([f"{key}: {val}" for key, val in options.items()])


def build_instruction(sample: dict) -> dict:
    """Convert a raw MedQA sample into the Mistral instruction format used for training."""
    question = sample['question']
    answer = sample['answer']
    answer_idx = sample['answer_idx']
    options = sample['options']

    options_text = format_options(options)

    system = (
        "You are MedMind, an expert clinical decision support AI trained on medical knowledge.\n"
        "You help medical professionals with diagnosis, treatment planning, and drug selection.\n"
        "Always reason step by step before giving your final answer.\n"
        "Be accurate, cite your reasoning, and flag any uncertainty."
    )

    user = f"Clinical Question:\n{question}\n\nOptions:\n{options_text}\n\nWhat is the best answer and why?"

    assistant = (
        f"Let me analyze this clinical scenario step by step.\n\n"
        f"Looking at the patient presentation and the available options, the correct answer is:\n\n"
        f"{answer_idx}: {answer}\n\n"
        f"Clinical Reasoning: This answer is correct based on the clinical presentation described. "
        f"The patient's symptoms, history, and examination findings are most consistent with this "
        f"treatment/diagnosis choice given standard medical guidelines."
    )

    full_text = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>"

    return {
        "text": full_text,
        "question": question,
        "answer": answer,
        "answer_idx": answer_idx
    }


print("\nConverting to instruction format...")
train_instructions = []
skipped = 0

for sample in train_data:
    if not sample.get('question') or not sample.get('answer'):
        skipped += 1
        continue
    if len(sample['question']) < 30:
        skipped += 1
        continue

    train_instructions.append(build_instruction(sample))

print(f"Converted: {len(train_instructions)} | Skipped: {skipped}")

os.makedirs("data/processed", exist_ok=True)

with open("data/processed/train_instructions.json", "w") as f:
    json.dump(train_instructions, f, indent=2)

# test set — kept separate, never used during training
test_instructions = []
for sample in test_data:
    if not sample.get('question') or not sample.get('answer'):
        continue
    test_instructions.append({
        "question": sample['question'],
        "answer": sample['answer'],
        "answer_idx": sample['answer_idx'],
        "options": sample['options']
    })

with open("data/processed/test_data.json", "w") as f:
    json.dump(test_instructions, f, indent=2)

print(f"Test samples saved: {len(test_instructions)}")

# quick sanity check
print(f"\n{'='*60}")
print("SAMPLE:")
print("="*60)
print(train_instructions[0]['text'])
print("="*60)

avg_len = sum(len(item['text'].split()) for item in train_instructions) / len(train_instructions)
print(f"\nAvg words per example: {avg_len:.0f}")