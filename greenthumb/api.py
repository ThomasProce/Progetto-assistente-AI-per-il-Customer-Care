"""FastAPI deployment of the agent.

Run locally with::

    uvicorn greenthumb.api:app --port 8000

Endpoints
---------
``POST /chat``            body ``{"message": ..., "session_id": optional}`` -> ``ChatResponse``
                          (``answer``, ``confidence``, ``sources``, ``needs_human``).
                          The session id is returned in the ``X-Session-Id`` header, so the
                          body matches the required contract exactly; send it back to keep
                          the conversation memory.
``POST /chat/debug``      same input, returns the response plus the full ReAct trace.
``DELETE /sessions/{id}`` forgets a conversation.
``GET /health``           liveness and configuration summary.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from greenthumb.agent import GreenThumbAgent
from greenthumb.config import get_settings
from greenthumb.schemas import AgentResult, ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent once at start-up (unless a test already injected one)."""
    if getattr(app.state, "agent", None) is None:
        app.state.agent = GreenThumbAgent.from_settings(get_settings())
    yield


app = FastAPI(
    title="GreenThumb Customer Care Agent",
    description="Assistente AI ReAct (LangChain + LiteLLM) per il servizio clienti di GreenThumb Marketplace.",
    version="1.0.0",
    lifespan=lifespan,
)


def _agent(request: Request) -> GreenThumbAgent:
    """Return the agent stored in the application state."""
    return request.app.state.agent


def _run(request: Request, body: ChatRequest) -> AgentResult:
    """Run the agent, turning unexpected failures into a 502 error."""
    try:
        return _agent(request).chat(body.message, session_id=body.session_id)
    except Exception as error:
        logger.exception("Agent failure")
        raise HTTPException(status_code=502, detail=f"Errore del modello o dei servizi a valle: {error}") from error


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request, response: Response) -> ChatResponse:
    """Answer a customer message with the structured JSON contract."""
    result = _run(request, body)
    response.headers["X-Session-Id"] = result.trace.session_id
    return result.response


@app.post("/chat/debug", response_model=AgentResult)
def chat_debug(body: ChatRequest, request: Request, response: Response) -> AgentResult:
    """Answer a customer message and include the full ReAct trace."""
    result = _run(request, body)
    response.headers["X-Session-Id"] = result.trace.session_id
    return result


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str, request: Request) -> dict:
    """Forget the short-term memory of a conversation."""
    if not _agent(request).sessions.reset(session_id):
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    return {"status": "deleted", "session_id": session_id}


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness probe with the main configuration values."""
    agent = _agent(request)
    return {
        "status": "ok",
        "llm_model": agent.settings.llm_model,
        "embedding_model": agent.settings.embedding_model,
        "memory_strategy": agent.sessions.strategy,
        "current_date": agent.settings.current_date.isoformat(),
        "orders_loaded": len(agent.orders),
    }
