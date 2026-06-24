"""Embedding helper, shared by ingest (index side) and retrieve (query side).
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL_NAME


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Loaded once per process and cached. First call downloads the model.
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings into vectors (one list[float] per input)."""
    embeddings = _model().encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
