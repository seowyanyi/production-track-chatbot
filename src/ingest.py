"""Stage 1 — Ingest: load -> chunk -> embed -> store.

Run with:  uv run python -m src.ingest
"""

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME, DOCS_DIR
from src.embeddings import embed_texts

def load_documents() -> list[dict]:
    """Read every .md file in docs/ and return one dict per document.
    """
    documents = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw_text)
        documents.append(
            {
                "text": body,
                "doc_name": meta.get("doc_name", path.stem),
                "source_url": meta.get("source_url", str(path)),
            }
        )
    return documents


def _parse_front_matter(raw_text: str) -> tuple[dict, str]:
    """Split a leading '---\\n...\\n---' YAML-ish block from the body."""
    if not raw_text.startswith("---"):
        return {}, raw_text
    _, front_matter_text, body = raw_text.split("---", 2)
    meta = {}
    for line in front_matter_text.strip().splitlines():
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
    vectordb_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Fresh start each run keeps Stage 1 simple — no dedup/upsert logic to reason
    # about. Re-running ingest fully rebuilds the index.
    if COLLECTION_NAME in [col.name for col in vectordb_client.list_collections()]:
        vectordb_client.delete_collection(COLLECTION_NAME)
    collection = vectordb_client.create_collection(COLLECTION_NAME)

    documents = load_documents()
    chunks: list[dict] = []
    for doc in documents:
        chunks.extend(chunk_document(doc))

    if not chunks:
        print("No chunks produced — did you implement chunk_document()?")
        return

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(chunk_texts)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=[
            {
                "source_url": chunk["source_url"],
                "doc_name": chunk["doc_name"],
                "section_heading": chunk["section_heading"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )
    print(f"Ingested {len(chunks)} chunks from {len(documents)} document(s).")


if __name__ == "__main__":
    main()
