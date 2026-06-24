"""Shared constants. Import from here instead of hardcoding strings elsewhere."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / ".chroma"  # local persistent vector store (gitignored)

# Embedding model. The SAME model must be used to index docs and to embed
# queries — otherwise the vectors live in different spaces and similarity is
# meaningless.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Chroma collection that holds our chunks.
COLLECTION_NAME = "pt_documents"

ANTHROPIC_MODEL = "claude-haiku-4-5"

# How many chunks to retrieve per query.
TOP_K = 6
