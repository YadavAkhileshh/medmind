import json
from collections import Counter

with open("data/processed/train_instructions.json") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# length stats
lengths = [len(item['text'].split()) for item in data]
print(f"\nLength: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
print(f"Over 600 words: {sum(1 for l in lengths if l > 600)}")

# format checks
bad_start = sum(1 for d in data if not d['text'].startswith("<s>[INST]"))
bad_end = sum(1 for d in data if not d['text'].endswith("</s>"))
bad_inst = sum(1 for d in data if "[/INST]" not in d['text'])
print(f"\nFormat issues: start={bad_start}, end={bad_end}, inst={bad_inst}")

# answer distribution
counts = Counter(item['answer_idx'] for item in data)
print("\nAnswer distribution:")
for opt, count in sorted(counts.items()):
    print(f"  {opt}: {count} ({count/len(data)*100:.1f}%)")

# duplicates
questions = [item['question'] for item in data]
dupes = len(questions) - len(set(questions))
print(f"\nDuplicates: {dupes}")

total_issues = bad_start + bad_end + bad_inst + dupes
if total_issues == 0:
    print("\nData looks good — ready for training")
else:
    print(f"\nFound {total_issues} issues")