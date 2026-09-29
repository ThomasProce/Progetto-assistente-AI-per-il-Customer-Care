"""Short-term (conversation) memory with two explicit strategies: summarization and trimming.

Only the *dialogue* is stored (customer messages and final assistant answers);
tool calls and observations are not kept, because they are bulky and can be
re-fetched on demand. Two strategies bound the size of the history sent to the
model:

``TrimmingMemory``
    Keeps the most recent messages that fit into ``max_tokens`` (sliding window,
    always starting from a customer message). Deterministic and free, but facts
    mentioned early in the chat (e.g. an order number) are lost.

``SummarizationMemory`` (default)
    When the history exceeds ``max_tokens``, the oldest messages are condensed by
    the LLM into a running summary that explicitly preserves order numbers,
    products, problems and promises made, while the last
    ``keep_last_messages`` messages are kept verbatim. Costs one extra LLM call
    every few turns, but preserves the facts customer care depends on.

Token counts use LangChain's ``count_tokens_approximately`` (≈4 chars/token),
which is provider-agnostic and good enough for a budget threshold.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from greenthumb.prompts import SUMMARY_PROMPT


class ConversationMemory(ABC):
    """Base class: stores the dialogue of one session and renders it for the model."""

    def __init__(self, max_tokens: int) -> None:
        """Set the token budget for the history."""
        self.max_tokens = max_tokens
        self.messages: list[BaseMessage] = []
        self.summary: str = ""

    def add_turn(self, user_message: str, assistant_answer: str) -> None:
        """Append one exchange and enforce the memory budget."""
        self.messages.extend([HumanMessage(user_message), AIMessage(assistant_answer)])
        self._compact()

    def context_messages(self) -> list[BaseMessage]:
        """Messages to prepend to the next model call (summary first, if any)."""
        prefix: list[BaseMessage] = []
        if self.summary:
            prefix.append(SystemMessage(f"Riassunto della conversazione precedente con il cliente:\n{self.summary}"))
        return prefix + list(self.messages)

    def token_count(self) -> int:
        """Approximate token count of the stored history."""
        return count_tokens_approximately(self.context_messages())

    @abstractmethod
    def _compact(self) -> None:
        """Reduce the history when it exceeds the budget."""


class TrimmingMemory(ConversationMemory):
    """Sliding window: keep the most recent messages within the token budget."""

    def _compact(self) -> None:
        """Drop the oldest messages (whole exchanges) beyond ``max_tokens``."""
        self.messages = trim_messages(
            self.messages,
            max_tokens=self.max_tokens,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=False,
        )


class SummarizationMemory(ConversationMemory):
    """Running summary of old messages + the most recent messages verbatim."""

    def __init__(self, max_tokens: int, keep_last_messages: int, llm: BaseChatModel) -> None:
        """Store the budget, how many messages to keep verbatim and the summarizer model."""
        super().__init__(max_tokens)
        self.keep_last_messages = max(2, keep_last_messages - keep_last_messages % 2)
        self._llm = llm
        self.summarizations = 0

    def _compact(self) -> None:
        """Summarize everything except the last ``keep_last_messages`` when over budget."""
        if count_tokens_approximately(self.messages) <= self.max_tokens:
            return
        if len(self.messages) <= self.keep_last_messages:
            return
        old, recent = self.messages[: -self.keep_last_messages], self.messages[-self.keep_last_messages :]
        transcript = "\n".join(
            f"{'Cliente' if isinstance(message, HumanMessage) else 'Assistente'}: {message.content}"
            for message in old
        )
        prompt = SUMMARY_PROMPT.format(previous_summary=self.summary or "(nessuno)", transcript=transcript)
        self.summary = str(self._llm.invoke(prompt).content).strip()
        self.messages = recent
        self.summarizations += 1


class SessionStore:
    """Thread-safe registry of conversation memories, keyed by session id."""

    def __init__(
        self,
        strategy: str,
        max_tokens: int,
        keep_last_messages: int,
        summarizer_llm: BaseChatModel | None = None,
    ) -> None:
        """Configure the memory strategy used for new sessions."""
        if strategy not in ("summarization", "trimming"):
            raise ValueError(f"Unknown memory strategy: {strategy!r}")
        if strategy == "summarization" and summarizer_llm is None:
            raise ValueError("The summarization strategy needs an LLM")
        self.strategy = strategy
        self._max_tokens = max_tokens
        self._keep_last_messages = keep_last_messages
        self._llm = summarizer_llm
        self._sessions: dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()

    def new_session_id(self) -> str:
        """Generate a fresh session id."""
        return uuid.uuid4().hex

    def get(self, session_id: str) -> ConversationMemory:
        """Return the memory for ``session_id``, creating it if needed."""
        with self._lock:
            if session_id not in self._sessions:
                if self.strategy == "summarization":
                    self._sessions[session_id] = SummarizationMemory(
                        self._max_tokens, self._keep_last_messages, self._llm
                    )
                else:
                    self._sessions[session_id] = TrimmingMemory(self._max_tokens)
            return self._sessions[session_id]

    def reset(self, session_id: str) -> bool:
        """Forget a session; return whether it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
