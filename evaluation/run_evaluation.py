"""Evaluate the agent on ``evaluation/eval_dataset.json`` with RAGAS, DeepEval and deterministic checks.

Steps
-----
1. **Run the agent** on every conversation (fresh session each; all turns are
   played so that short-term memory is exercised). The last turn is evaluated.
2. **RAGAS** (LLM-as-judge through LiteLLM):
   - ``faithfulness``: are the claims in the answer supported by the retrieved
     contexts (KB chunks + order records)? -> hallucination check;
   - ``answer_relevancy``: does the answer address the question?
   - ``context_recall``: do the retrieved contexts contain the facts of the
     reference answer? -> retrieval quality (only for KB questions).
3. **DeepEval**:
   - ``GEval`` "Correctness": answer vs. reference answer (facts, dates, amounts,
     escalation/refusal behaviour);
   - ``ToolCorrectnessMetric``: were the expected tools called? (deterministic).
4. **Deterministic checks**: ``needs_human`` accuracy, source hit rate,
   source grounding (sources subset of retrieved docs), schema validity.

Outputs go to ``evaluation/results/``: ``agent_outputs.json``, ``scores.csv``,
``summary.json`` and ``summary.md``.

Usage:
    python evaluation/run_evaluation.py [--limit N] [--skip-ragas] [--skip-deepeval]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import pandas as pd  # noqa: E402

from greenthumb.agent import GreenThumbAgent  # noqa: E402
from greenthumb.config import get_settings  # noqa: E402
from greenthumb.schemas import ChatResponse  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RAGAS_CONCURRENCY = 4

CORRECTNESS_STEPS = [
    "Compare the facts in 'actual output' (order status, dates, amounts, time windows, planting periods, "
    "policy rules, product specs) with 'expected output'. Any contradiction is a severe error.",
    "Check that the key information of 'expected output' needed to answer the customer is present in "
    "'actual output'. Missing key facts lower the score; different wording is fine.",
    "Check that the behaviour matches: if 'expected output' hands the case to a human operator, asks for "
    "clarification or politely refuses an off-topic request, 'actual output' must do the same.",
    "Extra details are acceptable only if they are correct and relevant; do not penalise politeness.",
]


def _user_input(turns: list[str]) -> str:
    """Render the customer side of a conversation (earlier turns give context to the judge)."""
    if len(turns) == 1:
        return turns[0]
    history = "\n".join(f"- {turn}" for turn in turns[:-1])
    return f"Messaggi precedenti del cliente:\n{history}\nDomanda attuale: {turns[-1]}"


def run_agent(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Play every conversation through the agent and collect the last-turn outputs."""
    settings = replace(get_settings(), escalations_path=RESULTS_DIR / "escalations.jsonl")
    agent = GreenThumbAgent.from_settings(settings)
    records = []
    for conversation in conversations:
        session_id = None
        started = time.perf_counter()
        for turn in conversation["turns"]:
            result = agent.chat(turn, session_id=session_id)
            session_id = result.trace.session_id
        elapsed = time.perf_counter() - started
        response = result.response
        tools_called = [call.tool for call in result.trace.tool_calls]
        records.append(
            {
                "id": conversation["id"],
                "category": conversation["category"],
                "difficulty": conversation["difficulty"],
                "user_input": _user_input(conversation["turns"]),
                "reference": conversation["reference"],
                "expected_tools": conversation["expected_tools"],
                "expected_sources": conversation["expected_sources"],
                "expected_needs_human": conversation["expected_needs_human"],
                "response": response.model_dump(),
                "tools_called": tools_called,
                "tool_steps": [call.model_dump() for call in result.trace.tool_calls],
                "retrieved_sources": result.trace.retrieved_sources,
                "retrieved_contexts": result.trace.retrieved_contexts,
                "dropped_sources": result.trace.dropped_sources,
                "request_scope": result.trace.draft.request_scope if result.trace.draft else None,
                "safety_net_escalation": result.trace.safety_net_escalation,
                "latency_s": round(elapsed, 2),
            }
        )
        print(f"[agent] {conversation['id']}: tools={tools_called} "
              f"confidence={response.confidence} needs_human={response.needs_human}")
    return records


def deterministic_scores(record: dict[str, Any]) -> dict[str, Any]:
    """Checks that need no judge model."""
    response = record["response"]
    sources = response["sources"]
    expected_sources = record["expected_sources"]
    return {
        "schema_valid": float(_is_valid_response(response)),
        "needs_human_correct": float(response["needs_human"] == record["expected_needs_human"]),
        "sources_grounded": float(set(sources) <= set(record["retrieved_sources"])),
        "source_hit": float(bool(set(sources) & set(expected_sources))) if expected_sources else math.nan,
        "retrieval_hit": float(bool(set(record["retrieved_sources"]) & set(expected_sources)))
        if expected_sources else math.nan,
    }


def _is_valid_response(response: dict[str, Any]) -> bool:
    """Validate the response against the Pydantic contract."""
    try:
        ChatResponse.model_validate(response)
        return True
    except Exception:
        return False


async def ragas_scores(records: list[dict[str, Any]], eval_model: str, embedding_model: str) -> list[dict]:
    """Compute RAGAS faithfulness, answer relevancy and context recall (NaN when not applicable)."""
    import instructor
    import litellm
    from ragas.embeddings import LiteLLMEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

    llm = llm_factory(eval_model, provider="litellm", client=instructor.from_litellm(litellm.acompletion),
                      adapter="litellm")
    embeddings = LiteLLMEmbeddings(model=embedding_model)
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    recall = ContextRecall(llm=llm)
    semaphore = asyncio.Semaphore(RAGAS_CONCURRENCY)

    async def _safe(coroutine) -> float:
        """Run one metric, returning NaN on judge failures."""
        async with semaphore:
            try:
                return float((await coroutine).value)
            except Exception as error:
                print(f"[ragas] metric failed: {error}")
                return math.nan

    async def _score(record: dict[str, Any]) -> dict[str, float]:
        """Score one record with the applicable metrics."""
        question, answer = record["user_input"], record["response"]["answer"]
        contexts = record["retrieved_contexts"]
        scores = {"ragas_faithfulness": math.nan, "ragas_answer_relevancy": math.nan,
                  "ragas_context_recall": math.nan}
        tasks = {}
        if contexts:
            tasks["ragas_faithfulness"] = faithfulness.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts)
        if record["category"] != "out_of_scope":
            tasks["ragas_answer_relevancy"] = relevancy.ascore(user_input=question, response=answer)
        if record["expected_sources"] and contexts:
            tasks["ragas_context_recall"] = recall.ascore(
                user_input=question, retrieved_contexts=contexts, reference=record["reference"])
        results = await asyncio.gather(*(_safe(task) for task in tasks.values()))
        scores.update(dict(zip(tasks, results)))
        print(f"[ragas] {record['id']}: " + ", ".join(f"{k}={v:.2f}" for k, v in scores.items()))
        return scores

    return await asyncio.gather(*(_score(record) for record in records))


def deepeval_scores(records: list[dict[str, Any]], eval_model: str) -> list[dict]:
    """Compute DeepEval GEval correctness and tool correctness."""
    from deepeval.metrics import GEval, ToolCorrectnessMetric
    from deepeval.models import LiteLLMModel
    from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

    judge = LiteLLMModel(model=eval_model, temperature=0)
    rows = []
    for record in records:
        test_case = LLMTestCase(
            input=record["user_input"],
            actual_output=record["response"]["answer"],
            expected_output=record["reference"],
            retrieval_context=record["retrieved_contexts"] or None,
            tools_called=[ToolCall(name=name) for name in dict.fromkeys(record["tools_called"])],
            expected_tools=[ToolCall(name=name) for name in record["expected_tools"]],
        )
        correctness = GEval(
            name="Correctness",
            evaluation_steps=CORRECTNESS_STEPS,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                               SingleTurnParams.EXPECTED_OUTPUT],
            model=judge,
            threshold=0.6,
            async_mode=False,
        )
        tool_correctness = ToolCorrectnessMetric(threshold=0.99, async_mode=False)
        row = {}
        for key, metric in (("deepeval_correctness", correctness), ("deepeval_tool_correctness", tool_correctness)):
            try:
                metric.measure(test_case)
                row[key] = float(metric.score)
                row[f"{key}_reason"] = metric.reason
            except Exception as error:
                print(f"[deepeval] {record['id']} {key} failed: {error}")
                row[key] = math.nan
                row[f"{key}_reason"] = str(error)
        print(f"[deepeval] {record['id']}: correctness={row['deepeval_correctness']:.2f} "
              f"tools={row['deepeval_tool_correctness']:.2f}")
        rows.append(row)
    return rows


METRIC_COLUMNS = [
    "ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_recall",
    "deepeval_correctness", "deepeval_tool_correctness",
    "needs_human_correct", "source_hit", "retrieval_hit", "sources_grounded", "schema_valid",
]


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate metrics overall and per category (NaN = metric not applicable)."""
    columns = [column for column in METRIC_COLUMNS if column in frame]
    overall = {column: {"mean": round(float(frame[column].mean()), 3), "n": int(frame[column].notna().sum())}
               for column in columns}
    per_category = frame.groupby("category")[columns].mean().round(3)
    return {
        "n_conversations": len(frame),
        "overall": overall,
        "per_category": json.loads(per_category.to_json(orient="index")),
        "confidence_distribution": frame["confidence"].value_counts().to_dict(),
        "mean_latency_s": round(float(frame["latency_s"].mean()), 2),
    }


def write_markdown(summary: dict[str, Any], frame: pd.DataFrame, path: Path) -> None:
    """Write a human-readable report of the scores."""
    lines = ["# Risultati della valutazione", "", f"Conversazioni valutate: {summary['n_conversations']}", "",
             "## Metriche aggregate", "", "| Metrica | Media | N |", "|---|---|---|"]
    for metric, values in summary["overall"].items():
        lines.append(f"| {metric} | {values['mean']:.3f} | {values['n']} |")
    lines += ["", "## Per categoria", "", pd.DataFrame(summary["per_category"]).T.to_markdown(), "",
              "## Per conversazione", ""]
    columns = ["id", "category", "confidence", "needs_human", "expected_needs_human"] + [
        column for column in METRIC_COLUMNS if column in frame]
    lines.append(frame[columns].round(2).to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full evaluation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N conversations.")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--skip-deepeval", action="store_true")
    parser.add_argument("--reuse-outputs", action="store_true",
                        help="Score the saved agent_outputs.json instead of re-running the agent.")
    args = parser.parse_args()

    settings = get_settings()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs_path = RESULTS_DIR / "agent_outputs.json"
    conversations = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["conversations"][: args.limit]

    if args.reuse_outputs:
        records = json.loads(outputs_path.read_text(encoding="utf-8"))
    else:
        records = run_agent(conversations)
        outputs_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [deterministic_scores(record) for record in records]
    if not args.skip_ragas:
        ragas_rows = asyncio.run(ragas_scores(records, settings.eval_model, settings.embedding_model))
        rows = [{**row, **extra} for row, extra in zip(rows, ragas_rows)]
    if not args.skip_deepeval:
        deepeval_rows = deepeval_scores(records, settings.eval_model)
        rows = [{**row, **extra} for row, extra in zip(rows, deepeval_rows)]

    frame = pd.DataFrame(
        [
            {
                "id": record["id"],
                "category": record["category"],
                "difficulty": record["difficulty"],
                "confidence": record["response"]["confidence"],
                "needs_human": record["response"]["needs_human"],
                "expected_needs_human": record["expected_needs_human"],
                "tools_called": ",".join(record["tools_called"]),
                "expected_tools": ",".join(record["expected_tools"]),
                "sources": ",".join(record["response"]["sources"]),
                "latency_s": record["latency_s"],
                **row,
            }
            for record, row in zip(records, rows)
        ]
    )
    frame.to_csv(RESULTS_DIR / "scores.csv", index=False)
    summary = summarize(frame)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, frame, RESULTS_DIR / "summary.md")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
