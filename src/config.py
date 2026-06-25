"""Shared constants. Import from here instead of hardcoding strings elsewhere."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = ROOT / "docs"

CHROMA_DIR = ROOT / ".chroma"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pt_documents"

ANTHROPIC_MODEL = "claude-haiku-4-5"

TOP_K = 6
