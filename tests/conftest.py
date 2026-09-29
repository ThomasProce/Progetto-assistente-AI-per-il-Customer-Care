"""Shared fixtures: a scripted chat model and a keyword retriever, so tests run offline."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenthumb.config import Settings  # noqa: E402
from greenthumb.knowledge_base import RetrievedChunk  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """Chat model that replays pre-defined messages and structured outputs."""

    responses: list[AIMessage] = []
    structured_outputs: list[Any] = []
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        """Identifier required by LangChain."""
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """Return the next scripted message (or a plain text reply when exhausted)."""
        self.calls.append(list(messages))
        message = self.responses.pop(0) if self.responses else AIMessage(content="Riassunto di prova.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        """Tools are ignored: the script decides which tool calls happen."""
        return self

    def with_structured_output(self, schema, **kwargs):
        """Return the next scripted structured output."""
        def _next(_input):
            """Pop the next scripted output, raising it if it is an exception."""
            output = self.structured_outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return output

        return RunnableLambda(_next)


class KeywordRetriever:
    """Returns fixed chunks whose keywords appear in the query."""

    def __init__(self, entries: list[tuple[str, RetrievedChunk]]) -> None:
        """``entries`` maps a keyword to the chunk returned when it matches."""
        self.entries = entries

    def search(self, query: str, categories: list[str] | None = None) -> list[RetrievedChunk]:
        """Return matching chunks filtered by category."""
        return [
            chunk
            for keyword, chunk in self.entries
            if keyword in query.lower() and (not categories or chunk.category in categories)
        ]


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Default settings with the escalation queue redirected to a temp dir."""
    return replace(Settings(), escalations_path=tmp_path / "escalations.jsonl")


@pytest.fixture
def tulip_chunk() -> RetrievedChunk:
    """A realistic retrieved chunk about tulip planting."""
    return RetrievedChunk(
        source="guides/bulbi_autunnali.md",
        title="Guida ai bulbi autunnali",
        category="guide",
        section="Calendario di impianto",
        content="I tulipani vanno piantati in autunno (ottobre-novembre).",
        score=0.71,
    )
