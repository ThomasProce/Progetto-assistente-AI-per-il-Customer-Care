"""Unit tests for data, orders, retrieval preprocessing, policy rules and memory."""

from __future__ import annotations

from datetime import date

from greenthumb.config import Settings
from greenthumb.escalation import EscalationService
from greenthumb.knowledge_base import load_documents, split_documents
from greenthumb.memory import SessionStore, SummarizationMemory, TrimmingMemory
from greenthumb.orders import OrderRepository
from greenthumb.policy import TurnEvidence, compute_confidence, decide_needs_human, resolve_sources
from greenthumb.schemas import AnswerDraft

from conftest import ScriptedChatModel

TODAY = date(2026, 6, 16)


def _draft(scope="in_scope", fully=True, cited=None) -> AnswerDraft:
    """Build an AnswerDraft for policy tests."""
    return AnswerDraft(answer="ok", cited_sources=cited or [], request_scope=scope, fully_answered=fully)


def test_knowledge_base_has_enough_documents_with_metadata():
    """The KB must contain at least 30 documents across the expected categories."""
    documents = load_documents(Settings().kb_dir)
    assert len(documents) >= 30
    assert {d.metadata["category"] for d in documents} == {"product", "guide", "policy", "faq"}
    assert any(d.metadata["source"] == "guides/bulbi_autunnali.md" for d in documents)


def test_chunks_carry_title_section_and_ids():
    """Chunks are prefixed with the document title and have unique ids."""
    chunks = split_documents(load_documents(Settings().kb_dir), 900, 150)
    assert all(c.page_content.startswith("Documento: ") for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert max(len(c.page_content) for c in chunks) < 1100


def test_orders_file_covers_all_statuses():
    """At least 20 orders covering every lifecycle status."""
    repository = OrderRepository(Settings().orders_path)
    assert len(repository) >= 20
    statuses = {repository.get(str(i))["status"] for i in range(1030, 1054)}
    assert {"in_preparazione", "spedito", "consegnato", "reso_in_corso", "rimborsato", "annullato"} <= statuses


def test_order_lookup_edge_cases():
    """Found, normalised, not found and invalid ids."""
    repository = OrderRepository(Settings().orders_path)
    found = repository.lookup("#1042", TODAY)
    assert found["status"] == "ok"
    assert found["order"]["days_until_delivery"] == 1
    assert "customer_email" not in found["order"]
    assert repository.lookup("9999", TODAY)["status"] == "not_found"
    assert repository.lookup("abc", TODAY)["status"] == "invalid_input"
    assert repository.lookup("1038", TODAY)["order"]["delivery_overdue"] is True


def test_escalation_ticket_is_written(tmp_path):
    """Tickets are appended to the queue file."""
    service = EscalationService(tmp_path / "queue.jsonl")
    ticket = service.create_ticket("reclamo", "Cliente arrabbiato", "high", session_id="s1")
    assert ticket["ticket_id"].startswith("ESC-")
    assert ticket["expected_pickup"] == "entro 4 ore lavorative"
    assert (tmp_path / "queue.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_sources_never_contain_unretrieved_documents():
    """Invented citations are dropped; path prefixes are normalised."""
    sources, dropped = resolve_sources(
        ["guides/bulbi_autunnali.md", "data/knowledge_base/policies/resi_e_rimborsi.md", "guides/inventata.md"],
        {"guides/bulbi_autunnali.md": 0.7, "policies/resi_e_rimborsi.md": 0.6},
    )
    assert sources == ["guides/bulbi_autunnali.md", "policies/resi_e_rimborsi.md"]
    assert dropped == ["guides/inventata.md"]


def test_confidence_and_needs_human_rules():
    """Each branch of the documented decision table."""
    threshold = 0.5
    strong = TurnEvidence(retrieved_scores={"a.md": 0.7}, tool_calls=1)
    weak = TurnEvidence(retrieved_scores={"a.md": 0.4}, tool_calls=1)
    assert compute_confidence(_draft(cited=["a.md"]), ["a.md"], strong, False, threshold) == "high"
    assert compute_confidence(_draft(cited=["a.md"]), ["a.md"], weak, False, threshold) == "medium"
    assert compute_confidence(_draft(), [], TurnEvidence(tool_calls=1), False, threshold) == "low"
    assert compute_confidence(_draft(), [], TurnEvidence(), False, threshold) == "medium"
    orders_only = TurnEvidence(successful_order_lookups=1, tool_calls=1)
    assert compute_confidence(_draft(), [], orders_only, False, threshold) == "high"
    uncited = TurnEvidence(retrieved_scores={"a.md": 0.7}, successful_order_lookups=1, tool_calls=2)
    assert compute_confidence(_draft(), [], uncited, False, threshold) == "medium"
    assert compute_confidence(_draft("out_of_scope"), [], TurnEvidence(), False, threshold) == "high"
    assert compute_confidence(_draft("needs_clarification"), [], TurnEvidence(), False, threshold) == "medium"
    assert compute_confidence(_draft(), [], strong, True, threshold) == "low"

    assert decide_needs_human(_draft(), TurnEvidence(escalated=True)) is True
    assert decide_needs_human(_draft(fully=False), TurnEvidence()) is True
    assert decide_needs_human(_draft("out_of_scope", fully=False), TurnEvidence()) is False
    assert decide_needs_human(_draft(), TurnEvidence(iteration_limit_reached=True)) is True


def test_trimming_memory_respects_budget():
    """Old exchanges are dropped and the window starts with a customer message."""
    memory = TrimmingMemory(max_tokens=60)
    for i in range(10):
        memory.add_turn(f"domanda numero {i} " * 3, f"risposta numero {i} " * 3)
    assert memory.token_count() <= 60
    assert memory.messages[0].type == "human"
    assert "9" in memory.messages[-1].content


def test_summarization_memory_keeps_recent_and_summarizes_old():
    """When over budget, old messages become a summary and the last ones stay verbatim."""
    llm = ScriptedChatModel(responses=[], structured_outputs=[], calls=[])
    memory = SummarizationMemory(max_tokens=60, keep_last_messages=2, llm=llm)
    for i in range(4):
        memory.add_turn(f"Ordine 1042, domanda {i} " * 2, f"Risposta {i} " * 2)
    assert memory.summarizations >= 1
    assert memory.summary == "Riassunto di prova."
    assert len(memory.messages) < 8  # 4 exchanges = 8 messages, the oldest were summarized
    assert memory.messages[-1].content.startswith("Risposta 3")
    assert memory.context_messages()[0].type == "system"


def test_session_store_isolates_sessions():
    """Different session ids get different memories; reset forgets them."""
    store = SessionStore("trimming", 500, 4)
    store.get("a").add_turn("ciao", "ciao!")
    assert store.get("b").messages == []
    assert store.reset("a") is True
    assert store.get("a").messages == []


def test_tokenizer_normalises_italian_text():
    """Accents, stop-words and inflections are normalised for BM25."""
    from greenthumb.knowledge_base import tokenize

    assert tokenize("Le cesoie") == tokenize("cesoie")
    assert tokenize("piantare")[0] == tokenize("piantina")[0]
    assert tokenize("può è perché") == ["perch"]
