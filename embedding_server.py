"""OpenAI-compatible embedding HTTP server.

Wraps the same embedding logic used by the ingest pipeline so external tools
(e.g. chromadb-admin's Text Query) can fetch query vectors that match the
indexed vectors exactly. Nothing here is hardcoded: the model name comes from
src.config and the embedding logic from src.embeddings, so any change there is
reflected automatically.

    uv run python embedding_server.py
    # POST http://localhost:8001/v1/embeddings
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import EMBED_MODEL_NAME
from src.embeddings import embed_texts

app = FastAPI()


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vectors = embed_texts(texts)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec.tolist()}
            for i, vec in enumerate(vectors)
        ],
        "model": EMBED_MODEL_NAME,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
