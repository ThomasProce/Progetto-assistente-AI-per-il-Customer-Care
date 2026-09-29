"""Application settings, loaded from environment variables (and an optional ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_str(name: str, default: str) -> str:
    """Read a string environment variable with a default."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a default."""
    return int(_env_str(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable (true/false, 1/0, yes/no) with a default."""
    return _env_str(name, str(default)).strip().lower() in {"true", "1", "yes"}


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a default."""
    return float(_env_str(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """All tunable parameters of the agent.

    Every field can be overridden with an environment variable named
    ``GREENTHUMB_<FIELD_NAME_UPPERCASE>`` (see ``.env.example``). API keys are
    never stored here: LiteLLM reads them directly from the environment
    (e.g. ``OPENAI_API_KEY``).
    """

    llm_model: str = "gpt-4o-mini-2024-07-18"
    llm_temperature: float = 0.0
    embedding_model: str = "text-embedding-3-small"
    eval_model: str = "gpt-4o-2024-11-20"

    kb_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    orders_path: Path = PROJECT_ROOT / "data" / "orders.json"
    escalations_path: Path = PROJECT_ROOT / "data" / "escalations.jsonl"
    vector_store_dir: Path = PROJECT_ROOT / "storage" / "chroma"
    collection_name: str = "greenthumb_kb"

    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_fetch_k: int = 12
    retrieval_top_k: int = 4
    max_chunks_per_doc: int = 2
    retrieval_top_docs: int = 3
    hybrid_search: bool = True
    parent_document: bool = True
    min_relevance_score: float = 0.35
    high_confidence_score: float = 0.45

    memory_strategy: str = "summarization"
    memory_max_tokens: int = 1200
    memory_keep_last_messages: int = 4

    max_agent_iterations: int = 6
    current_date: date = field(default_factory=lambda: date(2026, 6, 16))

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, loading ``.env`` if present."""
        load_dotenv(PROJECT_ROOT / ".env")
        defaults = cls()
        return cls(
            llm_model=_env_str("GREENTHUMB_LLM_MODEL", defaults.llm_model),
            llm_temperature=_env_float("GREENTHUMB_LLM_TEMPERATURE", defaults.llm_temperature),
            embedding_model=_env_str("GREENTHUMB_EMBEDDING_MODEL", defaults.embedding_model),
            eval_model=_env_str("GREENTHUMB_EVAL_MODEL", defaults.eval_model),
            kb_dir=Path(_env_str("GREENTHUMB_KB_DIR", str(defaults.kb_dir))),
            orders_path=Path(_env_str("GREENTHUMB_ORDERS_PATH", str(defaults.orders_path))),
            escalations_path=Path(_env_str("GREENTHUMB_ESCALATIONS_PATH", str(defaults.escalations_path))),
            vector_store_dir=Path(_env_str("GREENTHUMB_VECTOR_STORE_DIR", str(defaults.vector_store_dir))),
            collection_name=_env_str("GREENTHUMB_COLLECTION_NAME", defaults.collection_name),
            chunk_size=_env_int("GREENTHUMB_CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_env_int("GREENTHUMB_CHUNK_OVERLAP", defaults.chunk_overlap),
            retrieval_fetch_k=_env_int("GREENTHUMB_RETRIEVAL_FETCH_K", defaults.retrieval_fetch_k),
            retrieval_top_k=_env_int("GREENTHUMB_RETRIEVAL_TOP_K", defaults.retrieval_top_k),
            max_chunks_per_doc=_env_int("GREENTHUMB_MAX_CHUNKS_PER_DOC", defaults.max_chunks_per_doc),
            retrieval_top_docs=_env_int("GREENTHUMB_RETRIEVAL_TOP_DOCS", defaults.retrieval_top_docs),
            hybrid_search=_env_bool("GREENTHUMB_HYBRID_SEARCH", defaults.hybrid_search),
            parent_document=_env_bool("GREENTHUMB_PARENT_DOCUMENT", defaults.parent_document),
            min_relevance_score=_env_float("GREENTHUMB_MIN_RELEVANCE_SCORE", defaults.min_relevance_score),
            high_confidence_score=_env_float("GREENTHUMB_HIGH_CONFIDENCE_SCORE", defaults.high_confidence_score),
            memory_strategy=_env_str("GREENTHUMB_MEMORY_STRATEGY", defaults.memory_strategy),
            memory_max_tokens=_env_int("GREENTHUMB_MEMORY_MAX_TOKENS", defaults.memory_max_tokens),
            memory_keep_last_messages=_env_int(
                "GREENTHUMB_MEMORY_KEEP_LAST_MESSAGES", defaults.memory_keep_last_messages
            ),
            max_agent_iterations=_env_int("GREENTHUMB_MAX_AGENT_ITERATIONS", defaults.max_agent_iterations),
            current_date=date.fromisoformat(
                _env_str("GREENTHUMB_CURRENT_DATE", defaults.current_date.isoformat())
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings.from_env()
