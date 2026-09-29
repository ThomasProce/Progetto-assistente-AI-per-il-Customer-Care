"""End-to-end tests of the ReAct loop and the API with a scripted model (no network)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from greenthumb.agent import GreenThumbAgent
from greenthumb.api import app
from greenthumb.escalation import EscalationService
from greenthumb.memory import SessionStore
from greenthumb.orders import OrderRepository
from greenthumb.schemas import AnswerDraft

from conftest import KeywordRetriever, ScriptedChatModel


def _agent(settings, llm, retriever) -> GreenThumbAgent:
    """Assemble an agent around fakes."""
    return GreenThumbAgent(
        settings=settings,
        llm=llm,
        retriever=retriever,
        orders=OrderRepository(settings.orders_path),
        escalations=EscalationService(settings.escalations_path),
        sessions=SessionStore("trimming", 1000, 4),
    )


def _brief_example_llm() -> ScriptedChatModel:
    """Script for the example of the brief: order status + tulip planting advice."""
    return ScriptedChatModel(
        calls=[],
        responses=[
            AIMessage(
                content="Pensiero: servono lo stato dell'ordine e la guida ai bulbi.",
                tool_calls=[
                    {"name": "get_order_status", "args": {"order_id": "1042"}, "id": "c1"},
                    {"name": "search_knowledge_base", "args": {"query": "quando piantare tulipani",
                                                               "topic": "guide"}, "id": "c2"},
                ],
            ),
            AIMessage(content="L'ordine 1042 arriva domani. I tulipani si piantano a ottobre-novembre."),
        ],
        structured_outputs=[
            AnswerDraft(
                answer="L'ordine 1042 arriva domani. I tulipani vanno piantati in autunno (ottobre-novembre).",
                cited_sources=["guides/bulbi_autunnali.md", "guides/documento_inventato.md"],
                request_scope="in_scope",
                fully_answered=True,
            )
        ],
    )


def test_react_loop_orchestrates_multiple_tools(settings, tulip_chunk):
    """Both tools run in one turn; sources are grounded; confidence is high."""
    agent = _agent(settings, _brief_example_llm(), KeywordRetriever([("tulipani", tulip_chunk)]))
    result = agent.chat("Quando arriva l'ordine 1042? Posso piantare adesso i bulbi di tulipano?")

    assert [call.tool for call in result.trace.tool_calls] == ["get_order_status", "search_knowledge_base"]
    assert all(call.status == "ok" for call in result.trace.tool_calls)
    assert result.response.sources == ["guides/bulbi_autunnali.md"]
    assert result.trace.dropped_sources == ["guides/documento_inventato.md"]
    assert result.response.confidence == "high"
    assert result.response.needs_human is False
    assert len(result.trace.retrieved_contexts) == 2  # KB chunk + order record


def test_unanswered_in_scope_request_triggers_safety_net(settings):
    """If the draft admits it could not answer, a ticket is opened automatically."""
    llm = ScriptedChatModel(
        calls=[],
        responses=[
            AIMessage(content="", tool_calls=[{"name": "search_catalog", "args": {"query": "trattore"}, "id": "c1"}]),
            AIMessage(content="Non ho trovato informazioni."),
        ],
        structured_outputs=[
            AnswerDraft(answer="Non ho informazioni su questo prodotto.", cited_sources=[],
                        request_scope="in_scope", fully_answered=False)
        ],
    )
    result = _agent(settings, llm, KeywordRetriever([])).chat("Vendete trattorini tosaerba?")
    assert result.trace.tool_calls[0].status == "no_results"
    assert result.response.needs_human is True
    assert result.response.confidence == "low"
    assert result.trace.safety_net_escalation is True
    assert "ESC-" in result.response.answer


def test_structured_output_failure_falls_back_safely(settings):
    """A broken structured-output call still yields a valid response routed to a human."""
    llm = ScriptedChatModel(
        calls=[], responses=[AIMessage(content="Risposta")], structured_outputs=[ValueError("bad json")]
    )
    result = _agent(settings, llm, KeywordRetriever([])).chat("Ciao")
    assert result.response.needs_human is True
    assert result.response.answer


def test_memory_is_passed_to_following_turns(settings):
    """The second turn sees the first exchange in its context."""
    llm = ScriptedChatModel(
        calls=[],
        responses=[AIMessage(content="Ciao Laura!"), AIMessage(content="Certo.")],
        structured_outputs=[
            AnswerDraft(answer="Ciao Laura!", request_scope="in_scope", fully_answered=True),
            AnswerDraft(answer="Certo.", request_scope="in_scope", fully_answered=True),
        ],
    )
    agent = _agent(settings, llm, KeywordRetriever([]))
    first = agent.chat("Ciao, sono Laura")
    agent.chat("Ti ricordi il mio nome?", session_id=first.trace.session_id)
    second_turn_prompt = " ".join(str(m.content) for m in llm.calls[1])
    assert "sono Laura" in second_turn_prompt


def test_api_contract(settings, tulip_chunk):
    """POST /chat returns exactly the four required fields and a session header."""
    app.state.agent = _agent(settings, _brief_example_llm(), KeywordRetriever([("tulipani", tulip_chunk)]))
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Quando arriva l'ordine 1042? E i tulipani?"})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"answer", "confidence", "sources", "needs_human"}
        assert body["sources"] == ["guides/bulbi_autunnali.md"]
        assert response.headers["X-Session-Id"]
        assert client.post("/chat", json={"message": "   "}).status_code == 422
        assert client.get("/health").json()["orders_loaded"] == 24
    app.state.agent = None
