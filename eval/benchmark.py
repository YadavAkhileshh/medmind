import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from tqdm import tqdm

print("Loading model...")

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, "Yakhilesh/medmind-opt-medical")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.1,
    do_sample=True
)

with open("data/processed/test_data.json") as f:
    test_data = json.load(f)

print(f"Running on {len(test_data)} questions...")

correct = 0
total = 0
results = []

# 200 samples — full set takes too long on CPU
test_subset = test_data[:200]

for item in tqdm(test_subset, desc="Evaluating"):
    question = item['question']
    correct_idx = item['answer_idx']
    options = item['options']

    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])

    prompt = f"""You are a medical expert. Answer with only the letter (A, B, C, or D).

Question: {question}

Options:
{options_text}

Answer (letter only):"""

    output = pipe(prompt)[0]['generated_text']
    raw_answer = output[len(prompt):].strip()

    predicted_letter = None
    for char in raw_answer.upper():
        if char in ['A', 'B', 'C', 'D']:
            predicted_letter = char
            break

    is_correct = predicted_letter == correct_idx
    if is_correct:
        correct += 1
    total += 1

    results.append({
        "question": question[:100],
        "correct_answer": correct_idx,
        "predicted": predicted_letter,
        "is_correct": is_correct
    })

accuracy = correct / total * 100
random_baseline = 25.0

print(f"\n{'='*50}")
print(f"EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Questions tested:  {total}")
print(f"Correct:           {correct}")
print(f"Accuracy:          {accuracy:.1f}%")
print(f"Random baseline:   {random_baseline}%")
print(f"Over baseline:     +{accuracy - random_baseline:.1f}%")
print(f"{'='*50}")

os.makedirs("eval", exist_ok=True)
summary = {
    "total_tested": total,
    "correct": correct,
    "accuracy": round(accuracy, 2),
    "random_baseline": random_baseline,
    "improvement": round(accuracy - random_baseline, 2)
}

with open("eval/results.json", "w") as f:
    json.dump({"summary": summary, "detailed": results}, f, indent=2)

print(f"\nResults saved to eval/results.json")