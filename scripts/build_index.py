"""Index the knowledge base into the persistent Chroma vector store.

Usage:
    python scripts/build_index.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenthumb.config import get_settings  # noqa: E402
from greenthumb.knowledge_base import KnowledgeBaseRetriever, build_index  # noqa: E402
from greenthumb.llm import build_embeddings  # noqa: E402


def main() -> None:
    """Build (or refresh) the index and run a sample query."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-index even if nothing changed.")
    args = parser.parse_args()

    settings = get_settings()
    vector_store = build_index(settings, build_embeddings(settings), force=args.force)
    print(f"Chunks in collection: {vector_store._collection.count()}")

    retriever = KnowledgeBaseRetriever(vector_store, settings)
    for chunk in retriever.search("Quando si piantano i bulbi di tulipano?"):
        print(f"{chunk.score:.3f}  {chunk.source}  [{chunk.section}]")


if __name__ == "__main__":
    main()
