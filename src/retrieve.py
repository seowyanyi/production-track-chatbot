"""Stage 1 — Retrieve: query -> top-k chunks.
"""

import chromadb
from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from src.embeddings import embed_texts

def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_texts([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # results["documents"] == [["chunk text 1", "chunk text 2", ...]]
    #                            ^ outer list is "per query"; we unwrap [0]
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {"text": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(docs, metadatas, distances)
    ]

def get_collection():
    """Open the collection ingest.py built. Given."""
    vectordb_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return vectordb_client.get_collection(COLLECTION_NAME)
