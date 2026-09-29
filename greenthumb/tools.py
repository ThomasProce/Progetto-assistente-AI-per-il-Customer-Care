"""LangChain tools exposed to the ReAct agent.

Every tool has an explicit input schema (Pydantic), a description written for
the model (when to use it and what it returns) and a JSON output that always
contains a ``status`` field, so the model can handle edge cases explicitly:

==========================  =================================================
Tool                        ``status`` values
==========================  =================================================
``search_catalog``          ``ok`` | ``no_results``
``search_knowledge_base``   ``ok`` | ``no_results``
``get_order_status``        ``ok`` | ``not_found`` | ``invalid_input``
``escalate_to_human``       ``ok``
==========================  =================================================

Tools are built per turn (``build_tools``) around a ``TurnRecorder``, which
collects the evidence (retrieved documents and scores, order look-ups,
tickets) used afterwards to validate ``sources`` and compute ``confidence``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from greenthumb.escalation import EscalationService
from greenthumb.knowledge_base import RetrievedChunk, Retriever
from greenthumb.orders import OrderRepository

KB_CATEGORIES = {"guide": ["guide"], "policy": ["policy"], "faq": ["faq"]}


@dataclass
class TurnRecorder:
    """Collects the evidence produced by tools during a single agent turn."""

    retrieved_scores: dict[str, float] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)
    order_contexts: list[str] = field(default_factory=list)
    escalation_contexts: list[str] = field(default_factory=list)
    successful_order_lookups: int = 0
    escalation_ticket: str | None = None

    def record_chunks(self, chunks: list[RetrievedChunk]) -> None:
        """Remember retrieved chunks, keeping the best score per document."""
        for chunk in chunks:
            self.retrieved_scores[chunk.source] = max(chunk.score, self.retrieved_scores.get(chunk.source, 0.0))
            if chunk.content not in self.retrieved_contexts:
                self.retrieved_contexts.append(chunk.content)


class SearchCatalogInput(BaseModel):
    """Input of ``search_catalog``."""

    query: str = Field(..., description="Cosa cercare nel catalogo, es. 'bulbi di tulipano quando piantare'.")


class SearchKnowledgeBaseInput(BaseModel):
    """Input of ``search_knowledge_base``."""

    query: str = Field(..., description="Domanda o parole chiave, es. 'tempi di rimborso dopo un reso'.")
    topic: Literal["guide", "policy", "faq", "any"] = Field(
        default="any",
        description="Restringe la ricerca: guide = coltivazione e cura; policy = resi, rimborsi, spedizioni, "
        "garanzie, pagamenti, annullamenti, GreenPoints, contatti; faq = domande frequenti; any = tutto.",
    )


class GetOrderStatusInput(BaseModel):
    """Input of ``get_order_status``."""

    order_id: str = Field(..., description="Numero d'ordine di 4 cifre fornito dal cliente, es. '1042'.")


class EscalateToHumanInput(BaseModel):
    """Input of ``escalate_to_human``."""

    reason: Literal[
        "richiesta_operatore",
        "modifica_o_annullamento_ordine",
        "rimborso_o_eccezione_policy",
        "prodotto_danneggiato_o_difettoso",
        "problema_pagamento",
        "ritardo_consegna",
        "reclamo",
        "informazione_non_disponibile",
        "altro",
    ] = Field(..., description="Motivo principale del passaggio a un operatore.")
    summary: str = Field(
        ..., description="Riassunto per l'operatore: cosa chiede il cliente, dati già raccolti, cosa serve fare."
    )
    priority: Literal["high", "medium", "low"] = Field(..., description="Priorità del ticket.")
    order_id: str | None = Field(default=None, description="Numero d'ordine coinvolto, se noto.")


def _to_json(payload: dict) -> str:
    """Serialize a tool observation for the model."""
    return json.dumps(payload, ensure_ascii=False)


def _format_results(chunks: list[RetrievedChunk], empty_hint: str) -> str:
    """Build the observation returned by the two search tools."""
    if not chunks:
        return _to_json({"status": "no_results", "message": empty_hint})
    return _to_json(
        {
            "status": "ok",
            "results": [
                {"source": c.source, "title": c.title, "section": c.section, "score": round(c.score, 3),
                 "content": c.content}
                for c in chunks
            ],
        }
    )


def build_tools(
    retriever: Retriever,
    orders: OrderRepository,
    escalations: EscalationService,
    recorder: TurnRecorder,
    today: date,
    session_id: str,
) -> list[BaseTool]:
    """Create the agent's tools bound to this turn's recorder and session."""

    def search_catalog(query: str) -> str:
        """Search product sheets."""
        chunks = retriever.search(query, categories=["product"])
        recorder.record_chunks(chunks)
        return _format_results(
            chunks,
            "Nessun prodotto pertinente nel catalogo. Non inventare prodotti: dillo al cliente "
            "o chiedi di specificare meglio cosa cerca.",
        )

    def search_knowledge_base(query: str, topic: str = "any") -> str:
        """Search guides, policies and FAQ."""
        categories = KB_CATEGORIES.get(topic, ["guide", "policy", "faq"])
        chunks = retriever.search(query, categories=categories)
        recorder.record_chunks(chunks)
        return _format_results(
            chunks,
            "Nessun documento pertinente nella knowledge base. Prova con parole diverse o un altro topic; "
            "se ancora nulla, non inventare: proponi il passaggio a un operatore.",
        )

    def get_order_status(order_id: str) -> str:
        """Look up an order."""
        result = orders.lookup(order_id, today)
        if result["status"] == "ok":
            recorder.successful_order_lookups += 1
            recorder.order_contexts.append(_to_json(result["order"]))
        return _to_json(result)

    def escalate_to_human(reason: str, summary: str, priority: str, order_id: str | None = None) -> str:
        """Open a ticket for a human operator."""
        result = escalations.create_ticket(reason, summary, priority, session_id=session_id, order_id=order_id)
        recorder.escalation_ticket = result["ticket_id"]
        recorder.escalation_contexts.append(_to_json(result))
        return _to_json(result)

    return [
        StructuredTool.from_function(
            func=search_catalog,
            name="search_catalog",
            description=(
                "Ricerca semantica (RAG) nelle schede prodotto del catalogo GreenThumb: descrizione, prezzo, "
                "caratteristiche tecniche, periodo di semina/impianto, cura e garanzia di un prodotto specifico. "
                "Usalo per domande su un prodotto o per consigliare prodotti. "
                "Restituisce JSON {status: ok|no_results, results: [{source, title, section, score, content}]}."
            ),
            args_schema=SearchCatalogInput,
        ),
        StructuredTool.from_function(
            func=search_knowledge_base,
            name="search_knowledge_base",
            description=(
                "Ricerca semantica (RAG) nella knowledge base aziendale: guide di coltivazione (quando piantare, "
                "potatura, irrigazione, parassiti...), policy (resi, rimborsi, spedizioni, garanzie, pagamenti, "
                "annullamenti, programma fedeltà, contatti) e FAQ. "
                "Restituisce JSON {status: ok|no_results, results: [{source, title, section, score, content}]}."
            ),
            args_schema=SearchKnowledgeBaseInput,
        ),
        StructuredTool.from_function(
            func=get_order_status,
            name="get_order_status",
            description=(
                "Legge lo stato di un ordine dato il numero di 4 cifre: stato, prodotti, corriere, tracking, "
                "data di consegna stimata (days_until_delivery: 1 = domani; delivery_overdue: true se in ritardo), "
                "resi e rimborsi. Restituisce JSON {status: ok|not_found|invalid_input, order|message}. "
                "Usalo solo con un numero fornito dal cliente, mai inventato."
            ),
            args_schema=GetOrderStatusInput,
        ),
        StructuredTool.from_function(
            func=escalate_to_human,
            name="escalate_to_human",
            description=(
                "Passa la richiesta a un operatore umano aprendo un ticket. Usalo solo nei casi previsti dalle "
                "regole di escalation (richiesta esplicita, azioni su ordini/rimborsi/garanzie, danni, pagamenti, "
                "ritardi, reclami, informazioni non disponibili). "
                "Restituisce JSON {status: ok, ticket_id, priority, expected_pickup}."
            ),
            args_schema=EscalateToHumanInput,
        ),
    ]
