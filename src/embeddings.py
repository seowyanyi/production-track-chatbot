from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL_NAME


@lru_cache(maxsize=1)
def _embed_model() -> SentenceTransformer:
    # Loaded once per process and cached. First call downloads the model.
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    return np.asarray(_embed_model().encode(texts, normalize_embeddings=True))
