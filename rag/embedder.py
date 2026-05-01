from sentence_transformers import SentenceTransformer
import chromadb
import json
import os

print("Loading embedding model...")
embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

os.makedirs("data/embeddings", exist_ok=True)
client = chromadb.PersistentClient(path="data/embeddings/chroma_db")
collection = client.get_or_create_collection(
    name="medical_knowledge",
    metadata={"hnsw:space": "cosine"}
)


def chunk_text(text, chunk_size=200, overlap=30):
    """Split text into overlapping chunks for better embedding quality."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


def embed_and_store(documents: list):
    """Embed a list of documents and store them in ChromaDB."""
    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc['text'])
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"doc{doc_idx}_chunk{chunk_idx}")
            all_metadata.append({
                "source": doc.get("source", "unknown"),
                "topic": doc.get("topic", "general")
            })

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedder.encode(all_chunks, batch_size=32, show_progress_bar=True)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadata
    )
    print(f"Stored {len(all_chunks)} chunks in ChromaDB")


# load training answers as knowledge base
print("Loading medical knowledge...")
with open("data/processed/train_instructions.json") as f:
    data = json.load(f)

docs = []
for item in data[:2000]:
    text = item['text']
    if '[/INST]' in text:
        answer_part = text.split('[/INST]')[-1].replace('</s>', '').strip()
    else:
        answer_part = text

    docs.append({
        "text": answer_part,
        "source": "MedQA-USMLE",
        "topic": "clinical"
    })

embed_and_store(docs)
print("Knowledge base ready")