"""The GreenThumb customer-care agent: an explicit ReAct loop built on LangChain + LiteLLM.

One call to ``GreenThumbAgent.chat`` runs:

1. **Context**: system prompt (with today's date) + short-term memory of the
   session (summary and/or recent messages) + the new customer message.
2. **ReAct loop** (at most ``max_agent_iterations`` steps): the tool-calling model
   writes a short *Thought* and requests one or more *Actions* (tool calls); the
   tools run and their *Observations* are appended as ``ToolMessage``; the loop
   ends when the model answers without requesting tools.
3. **Structured output**: the model is asked to turn its answer into an
   ``AnswerDraft`` (Pydantic, via function calling).
4. **Policy**: code validates ``sources`` against retrieved documents and
   computes ``confidence`` / ``needs_human`` deterministically; if a human is
   needed but no ticket was opened, a safety-net ticket is created.
5. **Memory update**: the exchange is stored in the session memory.

The loop is written by hand (instead of using a prebuilt agent executor) so
that every step is visible in the trace and the evidence gathered by the tools
can be used to ground ``sources`` and ``confidence``.
"""

from __future__ import annotations

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from greenthumb.config import Settings
from greenthumb.escalation import EscalationService
from greenthumb.knowledge_base import KnowledgeBaseRetriever, Retriever, build_index
from greenthumb.llm import build_chat_model, build_embeddings
from greenthumb.memory import SessionStore
from greenthumb.orders import OrderRepository
from greenthumb.policy import TurnEvidence, compute_confidence, decide_needs_human, resolve_sources
from greenthumb.prompts import FINAL_ANSWER_PROMPT, SYSTEM_PROMPT
from greenthumb.schemas import AgentResult, AgentTrace, AnswerDraft, ChatResponse, ToolCallRecord
from greenthumb.tools import TurnRecorder, build_tools

logger = logging.getLogger(__name__)

WEEKDAYS_IT = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
FALLBACK_ANSWER = (
    "Mi scuso, al momento non riesco a completare la risposta. "
    "Ho inoltrato la tua richiesta a un operatore del servizio clienti."
)


class GreenThumbAgent:
    """ReAct agent with RAG tools, order look-up, escalation and conversation memory."""

    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        retriever: Retriever,
        orders: OrderRepository,
        escalations: EscalationService,
        sessions: SessionStore,
    ) -> None:
        """Wire the agent's dependencies (injected to keep it testable)."""
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.orders = orders
        self.escalations = escalations
        self.sessions = sessions

    @classmethod
    def from_settings(cls, settings: Settings, force_reindex: bool = False) -> "GreenThumbAgent":
        """Build a fully configured agent (indexing the knowledge base if needed)."""
        llm = build_chat_model(settings)
        vector_store = build_index(settings, build_embeddings(settings), force=force_reindex)
        sessions = SessionStore(
            strategy=settings.memory_strategy,
            max_tokens=settings.memory_max_tokens,
            keep_last_messages=settings.memory_keep_last_messages,
            summarizer_llm=llm,
        )
        return cls(
            settings=settings,
            llm=llm,
            retriever=KnowledgeBaseRetriever(vector_store, settings),
            orders=OrderRepository(settings.orders_path),
            escalations=EscalationService(settings.escalations_path),
            sessions=sessions,
        )

    def _system_prompt(self) -> str:
        """Render the system prompt with the current (simulated) date."""
        today = self.settings.current_date
        return SYSTEM_PROMPT.format(today=today.strftime("%d/%m/%Y"), weekday=WEEKDAYS_IT[today.weekday()])

    def _run_react_loop(
        self, messages: list[BaseMessage], tools_by_name: dict, trace: AgentTrace
    ) -> list[BaseMessage]:
        """Thought -> Action -> Observation until the model stops calling tools."""
        model = self.llm.bind_tools(list(tools_by_name.values()))
        for step in range(1, self.settings.max_agent_iterations + 1):
            ai_message = model.invoke(messages)
            messages.append(ai_message)
            if not ai_message.tool_calls:
                return messages
            thought = str(ai_message.content or "").strip()
            for call in ai_message.tool_calls:
                tool = tools_by_name.get(call["name"])
                try:
                    if tool is None:
                        raise ValueError(f"Unknown tool {call['name']!r}")
                    raw_observation = tool.invoke(call["args"])
                    observation = json.loads(raw_observation)
                except Exception as error:  # the model must see tool failures, not crash the turn
                    logger.warning("Tool %s failed: %s", call["name"], error)
                    observation = {"status": "error", "message": f"Errore dello strumento: {error}"}
                    raw_observation = json.dumps(observation, ensure_ascii=False)
                messages.append(ToolMessage(content=raw_observation, tool_call_id=call["id"], name=call["name"]))
                trace.tool_calls.append(
                    ToolCallRecord(
                        step=step,
                        thought=thought,
                        tool=call["name"],
                        arguments=call["args"],
                        status=str(observation.get("status", "unknown")),
                        observation=observation,
                    )
                )
        trace.iteration_limit_reached = True
        return messages

    def _compose_draft(self, messages: list[BaseMessage], recorder: TurnRecorder) -> AnswerDraft:
        """Ask the model for the structured final answer (Pydantic-validated)."""
        retrieved = "\n".join(f"- {source}" for source in recorder.retrieved_scores) or "(nessuno)"
        structured_model = self.llm.with_structured_output(AnswerDraft, method="function_calling")
        prompt = messages + [HumanMessage(FINAL_ANSWER_PROMPT.format(retrieved_sources=retrieved))]
        try:
            return structured_model.invoke(prompt)
        except Exception as error:
            logger.error("Structured output failed: %s", error)
            last_text = str(messages[-1].content or "").strip() if messages else ""
            return AnswerDraft(
                answer=last_text or FALLBACK_ANSWER,
                cited_sources=[],
                request_scope="in_scope",
                fully_answered=False,
            )

    def chat(self, message: str, session_id: str | None = None) -> AgentResult:
        """Answer one customer message and return the structured response with its trace."""
        session_id = session_id or self.sessions.new_session_id()
        memory = self.sessions.get(session_id)
        trace = AgentTrace(session_id=session_id)
        recorder = TurnRecorder()
        tools = build_tools(
            self.retriever, self.orders, self.escalations, recorder, self.settings.current_date, session_id
        )

        messages: list[BaseMessage] = [SystemMessage(self._system_prompt())]
        messages += memory.context_messages()
        messages.append(HumanMessage(message))

        messages = self._run_react_loop(messages, {tool.name: tool for tool in tools}, trace)
        draft = self._compose_draft(messages, recorder)

        evidence = TurnEvidence(
            retrieved_scores=recorder.retrieved_scores,
            successful_order_lookups=recorder.successful_order_lookups,
            tool_calls=len(trace.tool_calls),
            escalated=recorder.escalation_ticket is not None,
            iteration_limit_reached=trace.iteration_limit_reached,
        )
        sources, dropped = resolve_sources(draft.cited_sources, recorder.retrieved_scores)
        needs_human = decide_needs_human(draft, evidence)
        answer = draft.answer

        if needs_human and recorder.escalation_ticket is None:
            ticket = self.escalations.create_ticket(
                reason="informazione_non_disponibile",
                summary=f"Escalation automatica (safety net). Messaggio del cliente: {message}",
                priority="medium",
                session_id=session_id,
            )
            recorder.escalation_ticket = ticket["ticket_id"]
            trace.safety_net_escalation = True
            answer += (
                f"\n\nHo inoltrato la tua richiesta a un operatore (ticket {ticket['ticket_id']}), "
                f"che ti ricontatterà {ticket['expected_pickup']}."
            )

        response = ChatResponse(
            answer=answer,
            confidence=compute_confidence(
                draft, sources, evidence, needs_human, self.settings.high_confidence_score
            ),
            sources=sources,
            needs_human=needs_human,
        )

        try:
            memory.add_turn(message, response.answer)
        except Exception as error:  # a failed summary must not lose the answer
            logger.error("Memory update failed: %s", error)

        trace.retrieved_sources = {source: round(score, 3) for source, score in recorder.retrieved_scores.items()}
        trace.retrieved_contexts = (
            recorder.retrieved_contexts + recorder.order_contexts + recorder.escalation_contexts
        )
        trace.dropped_sources = dropped
        trace.draft = draft
        trace.escalation_ticket = recorder.escalation_ticket
        trace.memory_summary = memory.summary
        return AgentResult(response=response, trace=trace)
