from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
client = chromadb.PersistentClient(path="data/embeddings/chroma_db")
collection = client.get_collection("medical_knowledge")


def retrieve(query: str, top_k: int = 3) -> list:
    """Search the vector DB for chunks most similar to the query."""
    query_embedding = embedder.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        retrieved.append({
            "content": doc,
            "source": meta['source'],
            "relevance": round(1 - dist, 3)
        })

    return retrieved


if __name__ == "__main__":
    results = retrieve("patient with chest pain and shortness of breath")
    for i, r in enumerate(results):
        print(f"\nResult {i+1} (relevance: {r['relevance']})")
        print(f"Source: {r['source']}")
        print(f"Content: {r['content'][:150]}...")