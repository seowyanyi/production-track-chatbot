"""Stage 1 — Retrieve: query -> top-k chunks.
"""

import chromadb
from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from src.embeddings import embed_texts

def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_texts([query])

    query_result = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    if query_result["documents"] is None or query_result["metadatas"] is None or query_result["distances"] is None:
        raise ValueError("Chroma query returned None for requested include fields")

    chunk_texts  = query_result["documents"][0]
    metadatas    = query_result["metadatas"][0]
    distances    = query_result["distances"][0]

    return [
        {"text": chunk_text, "metadata": chunk_metadata, "distance": chunk_distance}
        for chunk_text, chunk_metadata, chunk_distance in zip(chunk_texts, metadatas, distances)
    ]

def get_collection():
    """Open the collection ingest.py built. Given."""
    vectordb_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return vectordb_client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    query = "intention"
    chunks = retrieve(query)
    print(f"Retrieved {len(chunks)} chunks for query: {query}\n\n")
    for c in chunks:
        print(f"[{c['distance']:.3f}] {c['metadata']['doc_name']} / {c['metadata']['section_heading']}")
        print(c['text'][:200])
        print()
