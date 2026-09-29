"""Pydantic models: API contract, LLM structured output and execution trace."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Confidence = Literal["high", "medium", "low"]
RequestScope = Literal["in_scope", "out_of_scope", "needs_clarification"]


class ChatRequest(BaseModel):
    """Body of ``POST /chat``."""

    message: str = Field(..., min_length=1, max_length=2000, description="Messaggio del cliente.")
    session_id: str | None = Field(
        default=None,
        max_length=100,
        description="Id di conversazione per la memoria a breve termine. Se assente ne viene creato uno nuovo.",
    )

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        """Reject messages made only of whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    """Structured answer returned to the client (exact contract required by the brief)."""

    answer: str = Field(..., min_length=1, description="Risposta in linguaggio naturale per il cliente.")
    confidence: Confidence = Field(..., description="Affidabilità della risposta, calcolata con regole deterministiche.")
    sources: list[str] = Field(
        default_factory=list, description="Documenti della knowledge base effettivamente recuperati e usati."
    )
    needs_human: bool = Field(..., description="True se la richiesta è stata passata a un operatore umano.")


class AnswerDraft(BaseModel):
    """Structured output requested from the LLM at the end of the ReAct loop.

    The LLM only writes the answer and classifies the request; ``confidence`` and
    ``needs_human`` are derived by code (see ``greenthumb.policy``) so that they
    are reproducible and not left to the model's self-assessment.
    """

    answer: str = Field(..., description="Risposta finale per il cliente, in italiano.")
    cited_sources: list[str] = Field(
        default_factory=list,
        description="Id dei documenti (es. 'guides/bulbi_autunnali.md') su cui si basa la risposta, "
        "scelti SOLO tra quelli restituiti dagli strumenti.",
    )
    request_scope: RequestScope = Field(
        ...,
        description="in_scope: richiesta pertinente al servizio clienti GreenThumb; "
        "out_of_scope: richiesta estranea (es. meteo, politica, compiti); "
        "needs_clarification: mancano informazioni per rispondere (es. numero d'ordine assente o errato).",
    )
    fully_answered: bool = Field(
        ...,
        description="True se la risposta soddisfa tutte le domande del cliente usando le informazioni "
        "trovate o se la richiesta è stata correttamente passata a un operatore.",
    )


class ToolCallRecord(BaseModel):
    """One Action/Observation step of the ReAct loop."""

    step: int
    thought: str = ""
    tool: str
    arguments: dict[str, Any]
    status: str
    observation: dict[str, Any]


class AgentTrace(BaseModel):
    """Everything that happened while answering one message (used for debugging and evaluation)."""

    session_id: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    retrieved_sources: dict[str, float] = Field(
        default_factory=dict, description="source -> best relevance score among retrieved chunks."
    )
    retrieved_contexts: list[str] = Field(default_factory=list)
    dropped_sources: list[str] = Field(
        default_factory=list, description="Sources cited by the LLM but never retrieved (discarded)."
    )
    draft: AnswerDraft | None = None
    escalation_ticket: str | None = None
    safety_net_escalation: bool = False
    iteration_limit_reached: bool = False
    memory_summary: str = ""


class AgentResult(BaseModel):
    """Public response plus its trace."""

    response: ChatResponse
    trace: AgentTrace
