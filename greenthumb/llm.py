"""Factories for the chat model and the embedding model, both routed through LiteLLM."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_litellm import ChatLiteLLM, LiteLLMEmbeddings

from greenthumb.config import Settings


def build_chat_model(settings: Settings, temperature: float | None = None) -> BaseChatModel:
    """Create the LangChain chat model backed by LiteLLM.

    LiteLLM exposes a single OpenAI-compatible interface over 100+ providers, so
    switching model/provider only requires changing ``GREENTHUMB_LLM_MODEL``
    (e.g. ``anthropic/claude-...`` or ``groq/llama-...``).
    """
    return ChatLiteLLM(
        model=settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_retries=3,
        request_timeout=60,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the LangChain embedding model backed by LiteLLM."""
    return LiteLLMEmbeddings(model=settings.embedding_model, max_retries=3)
