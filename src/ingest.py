"""Stage 1 — Ingest: load -> chunk -> embed -> store.

Run with:  uv run python -m src.ingest
"""

import numpy as np
import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME, DOCS_DIR
from src.embeddings import embed_texts

def load_documents() -> list[dict]:
    """Read every .md file in docs/ and return one dict per document.
    """
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        docs.append(
            {
                "text": body,
                "doc_name": meta.get("doc_name", path.stem),
                "source_url": meta.get("source_url", str(path)),
            }
        )
    return docs


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split a leading '---\\n...\\n---' YAML-ish block from the body."""
    if not raw.startswith("---"):
        return {}, raw
    _, fm, body = raw.split("---", 2)
    meta = {}
    for line in fm.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def chunk_document(doc: dict) -> list[dict]:
    chunks: list[dict] = []
    current_heading = doc["doc_name"]
    current_body = ""
    chunk_index = 0

    for line in doc["text"].splitlines():
        if line.startswith("#"):
            if current_body.strip():
                chunks.append({
                    "text": current_body.strip(),
                    "source_url": doc["source_url"],
                    "doc_name": doc["doc_name"],
                    "section_heading": current_heading,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
                current_body = ""
            current_heading = line.lstrip("#").strip()
            continue
        current_body += line + "\n"

    if current_body.strip():
        chunks.append({
            "text": current_body.strip(),
            "source_url": doc["source_url"],
            "doc_name": doc["doc_name"],
            "section_heading": current_heading,
            "chunk_index": chunk_index,
        })

    return chunks

def main() -> None:
    """load -> chunk -> embed -> store. Orchestration is given; it calls your
    chunk_document()."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Fresh start each run keeps Stage 1 simple — no dedup/upsert logic to reason
    # about. Re-running ingest fully rebuilds the index.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    documents = load_documents()
    chunks: list[dict] = []
    for doc in documents:
        chunks.extend(chunk_document(doc))

    if not chunks:
        print("No chunks produced — did you implement chunk_document()?")
        return

    texts = [c["text"] for c in chunks]
    embeddings = np.array(embed_texts(texts), dtype=np.float32)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_url": c["source_url"],
                "doc_name": c["doc_name"],
                "section_heading": c["section_heading"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )
    print(f"Ingested {len(chunks)} chunks from {len(documents)} document(s).")


if __name__ == "__main__":
    main()
