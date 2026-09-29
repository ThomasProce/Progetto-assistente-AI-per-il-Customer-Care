"""Deterministic rules turning the LLM draft + tool evidence into ``confidence``, ``needs_human`` and ``sources``.

Keeping these decisions in code (instead of asking the LLM "how confident are
you?") makes them reproducible, testable and explainable:

``sources``
    Intersection between the documents cited by the LLM and the documents
    actually returned by the retrieval tools in *this* turn. Anything else is
    discarded (and logged in the trace), so ``sources`` can never contain an
    invented file.

``needs_human``
    True if an escalation ticket was opened (by the agent through the
    ``escalate_to_human`` tool), if the ReAct loop hit its iteration limit, or
    if the request is in scope but the draft admits it could not be fully
    answered (safety net: the agent will then open the ticket itself).

``confidence`` (evaluated in this order)
    1. ``low``    if ``needs_human`` (the automated answer does not resolve the request);
    2. ``high``   if the request is out of scope (declining is always policy-compliant);
    3. ``medium`` if the agent is asking for clarification;
    4. for answered in-scope requests, the *evidence* is the list of relevance
       scores of the cited documents plus 1.0 for each successful order lookup
       (plus 0.0 if documents were retrieved but none is cited, i.e. unattributed knowledge):
         - no tool used at all (small talk)            -> ``medium``
         - tools used but no usable evidence            -> ``low``
         - min(evidence) >= ``high_confidence_score``   -> ``high``
         - otherwise                                    -> ``medium``
"""

from __future__ import annotations

from dataclasses import dataclass, field

from greenthumb.schemas import AnswerDraft, Confidence


@dataclass
class TurnEvidence:
    """What the tools produced during one agent turn."""

    retrieved_scores: dict[str, float] = field(default_factory=dict)
    successful_order_lookups: int = 0
    tool_calls: int = 0
    escalated: bool = False
    iteration_limit_reached: bool = False


def resolve_sources(cited: list[str], retrieved_scores: dict[str, float]) -> tuple[list[str], list[str]]:
    """Keep only cited sources that were really retrieved; return ``(sources, dropped)``."""
    sources: list[str] = []
    dropped: list[str] = []
    for source in cited:
        normalized = source.strip().lstrip("/")
        if normalized.startswith("data/knowledge_base/"):
            normalized = normalized.removeprefix("data/knowledge_base/")
        if normalized in retrieved_scores:
            if normalized not in sources:
                sources.append(normalized)
        else:
            dropped.append(source)
    return sources, dropped


def decide_needs_human(draft: AnswerDraft, evidence: TurnEvidence) -> bool:
    """Apply the escalation rules described in the module docstring."""
    if evidence.escalated or evidence.iteration_limit_reached:
        return True
    return draft.request_scope == "in_scope" and not draft.fully_answered


def compute_confidence(
    draft: AnswerDraft,
    sources: list[str],
    evidence: TurnEvidence,
    needs_human: bool,
    high_confidence_score: float,
) -> Confidence:
    """Apply the confidence rules described in the module docstring."""
    if needs_human:
        return "low"
    if draft.request_scope == "out_of_scope":
        return "high"
    if draft.request_scope == "needs_clarification":
        return "medium"
    scores = [evidence.retrieved_scores[source] for source in sources]
    scores += [1.0] * evidence.successful_order_lookups
    if evidence.retrieved_scores and not sources:
        # Documents were searched but none is cited: the knowledge part of the answer is unattributed.
        scores.append(0.0)
    if evidence.tool_calls == 0:
        return "medium"
    if not scores:
        return "low"
    return "high" if min(scores) >= high_confidence_score else "medium"
