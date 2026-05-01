"""
Rebuild the ChromaDB knowledge base with clean Q+A pairs
instead of raw MCQ training text.

Run: python rag/rebuild_knowledge.py
"""

import sys
import os
import json
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentence_transformers import SentenceTransformer
import chromadb

chroma_path = "data/embeddings/chroma_db"

# wipe old DB
if os.path.exists(chroma_path):
    shutil.rmtree(chroma_path)
    print("Deleted old ChromaDB")

os.makedirs(chroma_path, exist_ok=True)

print("Loading embedder...")
embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_or_create_collection(
    name="medical_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# load source data
print("Loading data...")
train_path = "data/processed/train_instructions.json"
test_path = "data/processed/test_data.json"

train_data = json.load(open(train_path)) if os.path.exists(train_path) else []
test_data = json.load(open(test_path)) if os.path.exists(test_path) else []

chunks, ids, metadatas = [], [], []
idx = 0

# test set: clean question + answer pairs
for item in test_data[:500]:
    q = item.get('question', '')[:200]
    a = item.get('answer', '')
    if not q or not a or len(a) > 100:
        continue
    chunks.append(f"Clinical question: {q} Correct answer: {a}")
    ids.append(f"test_{idx}")
    metadatas.append({"source": "MedQA-USMLE", "type": "qa_pair"})
    idx += 1

# train set: question + diagnosis/treatment
for item in train_data[:1000]:
    q = item.get('question', '')[:200]
    a = item.get('answer', '')
    if not q or not a or len(a) > 100:
        continue
    chunks.append(f"Clinical question: {q} Correct diagnosis/treatment: {a}")
    ids.append(f"train_{idx}")
    metadatas.append({"source": "MedQA-USMLE", "type": "qa_pair"})
    idx += 1

if not chunks:
    print("No data found — run download_data.py and prepare_instructions.py first")
    sys.exit(1)

print(f"Embedding {len(chunks)} clean medical facts...")

batch_size = 64
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    embeddings = embedder.encode(batch, show_progress_bar=False)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=batch,
        ids=ids[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )
    print(f"  {min(i+batch_size, len(chunks))}/{len(chunks)}")

print(f"\nDone — {len(chunks)} facts stored in ChromaDB")