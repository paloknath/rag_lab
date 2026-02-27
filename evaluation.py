"""
LLM-as-a-Judge evaluation for RAG pipeline quality.

Evaluates retrieval quality (context relevance, context sufficiency) and
generation quality (faithfulness, answer relevance) using the same local LLM.
"""

import json
import re
import time
from dataclasses import dataclass

import tiktoken
from openai import OpenAI

import config


# ── Data Classes ────────────────────────────────────────────────


@dataclass
class MetricScore:
    """A single evaluation metric result."""
    name: str
    score: int | None          # 1-5 or None if not applicable / parse failure
    reason: str
    applicable: bool = True    # False for retrieval metrics in No RAG mode


@dataclass
class EvaluationResult:
    """Complete evaluation output for a single query-response pair."""
    context_relevance: MetricScore
    context_sufficiency: MetricScore
    faithfulness: MetricScore
    answer_relevance: MetricScore
    latency: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to a serializable dict for Streamlit session state."""
        def _metric(m: MetricScore) -> dict:
            return {"score": m.score, "reason": m.reason, "applicable": m.applicable}
        return {
            "context_relevance": _metric(self.context_relevance),
            "context_sufficiency": _metric(self.context_sufficiency),
            "faithfulness": _metric(self.faithfulness),
            "answer_relevance": _metric(self.answer_relevance),
            "latency": self.latency,
            "error": self.error,
        }

    @staticmethod
    def from_dict(d: dict) -> "EvaluationResult":
        """Reconstruct from a serialized dict."""
        def _make(name: str, data: dict) -> MetricScore:
            return MetricScore(
                name=name,
                score=data.get("score"),
                reason=data.get("reason", ""),
                applicable=data.get("applicable", True),
            )
        return EvaluationResult(
            context_relevance=_make("Context Relevance", d["context_relevance"]),
            context_sufficiency=_make("Context Sufficiency", d["context_sufficiency"]),
            faithfulness=_make("Faithfulness", d["faithfulness"]),
            answer_relevance=_make("Answer Relevance", d["answer_relevance"]),
            latency=d.get("latency", 0.0),
            error=d.get("error"),
        )


# ── Judge Prompts ──────────────────────────────────────────────


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial quality judge for a Retrieval-Augmented Generation (RAG) system.\n"
    "You will evaluate the quality of a RAG pipeline's output given a user query, "
    "retrieved context, and generated answer.\n\n"
    "Score each metric on a 1-5 integer scale. Provide a brief justification (1-2 sentences) "
    "for each score.\n\n"
    "Return your evaluation as a JSON object with this exact structure:\n"
    "{\n"
    '  "context_relevance": {"score": <1-5>, "reason": "<brief justification>"},\n'
    '  "context_sufficiency": {"score": <1-5>, "reason": "<brief justification>"},\n'
    '  "faithfulness": {"score": <1-5>, "reason": "<brief justification>"},\n'
    '  "answer_relevance": {"score": <1-5>, "reason": "<brief justification>"}\n'
    "}\n\n"
    "Scoring rubric:\n\n"
    "context_relevance — How relevant are the retrieved passages to the query?\n"
    "  1: Completely irrelevant  2: Mostly irrelevant  3: Partially relevant  "
    "4: Mostly relevant  5: Highly relevant\n\n"
    "context_sufficiency — Do the passages provide enough information to fully answer the query?\n"
    "  1: Severely insufficient  2: Insufficient  3: Partially sufficient  "
    "4: Mostly sufficient  5: Fully sufficient\n\n"
    "faithfulness — Is the answer grounded in the provided context without hallucination?\n"
    "  1: Heavily hallucinated  2: Partially hallucinated  3: Mixed  "
    "4: Mostly faithful  5: Fully faithful\n\n"
    "answer_relevance — Does the answer directly and completely address the user's query?\n"
    "  1: Off-topic  2: Barely addresses query  3: Partially addresses  "
    "4: Mostly addresses  5: Fully addresses\n\n"
    "Return ONLY the JSON object. No other text."
)

JUDGE_USER_PROMPT_WITH_CONTEXT = (
    "Evaluate the following RAG output:\n\n"
    "**User Query:**\n{query}\n\n"
    "**Retrieved Context:**\n{context}\n\n"
    "**Generated Answer:**\n{answer}\n\n"
    "Provide your JSON evaluation:"
)

JUDGE_USER_PROMPT_NO_CONTEXT = (
    "Evaluate the following LLM output (no retrieval was performed):\n\n"
    "**User Query:**\n{query}\n\n"
    "**Generated Answer:**\n{answer}\n\n"
    "Since no retrieval was performed, set context_relevance and context_sufficiency scores to null.\n"
    "For faithfulness, evaluate whether the answer makes claims that seem unreliable or fabricated.\n\n"
    "Provide your JSON evaluation:"
)


# ── Token Truncation ───────────────────────────────────────────


_encoder = tiktoken.get_encoding("cl100k_base")


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens using tiktoken."""
    tokens = _encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoder.decode(tokens[:max_tokens])


# ── Core Evaluation Function ───────────────────────────────────


def evaluate_rag_response(
    query: str,
    context: str,
    answer: str,
    strategy: str,
) -> EvaluationResult:
    """
    Run LLM-as-a-Judge evaluation on a RAG query-response pair.

    Args:
        query: The user's original question.
        context: The retrieved context (empty string for No RAG).
        answer: The generated answer.
        strategy: The retrieval strategy name.

    Returns:
        EvaluationResult with scores for all four metrics.
    """
    start = time.time()
    client = OpenAI(base_url=config.JUDGE_BASE_URL, api_key=config.JUDGE_API_KEY)

    has_context = bool(context and context.strip())

    if has_context:
        truncated_context = _truncate_to_tokens(context, config.JUDGE_MAX_CONTEXT_TOKENS)
        user_msg = JUDGE_USER_PROMPT_WITH_CONTEXT.format(
            query=query,
            context=truncated_context,
            answer=answer,
        )
    else:
        user_msg = JUDGE_USER_PROMPT_NO_CONTEXT.format(
            query=query,
            answer=answer,
        )

    try:
        response = client.chat.completions.create(
            model=config.JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
        )
        raw_output = response.choices[0].message.content.strip()
        result = _parse_judge_output(raw_output, has_context)
        result.latency = time.time() - start
        return result

    except Exception as e:
        return _error_result(str(e), time.time() - start)


# ── Parsing Logic ──────────────────────────────────────────────


def _parse_judge_output(raw: str, has_context: bool) -> EvaluationResult:
    """Parse the JSON output from the judge LLM with 3-layer fallback."""
    json_str = raw

    # Layer 1: Handle markdown code fences
    if "```" in json_str:
        parts = json_str.split("```")
        if len(parts) >= 3:
            json_str = parts[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

    # Layer 2: Try direct JSON parse
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Layer 3: Regex extraction
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return _error_result(
                    f"Failed to parse judge output as JSON: {raw[:200]}", 0.0
                )
        else:
            return _error_result(
                f"No JSON object found in judge output: {raw[:200]}", 0.0
            )

    def _extract_metric(key: str, applicable: bool) -> MetricScore:
        display_name = key.replace("_", " ").title()
        metric_data = data.get(key)
        if not applicable or metric_data is None:
            return MetricScore(
                name=display_name,
                score=None,
                reason="Not applicable (no retrieval performed)",
                applicable=False,
            )
        score = metric_data.get("score") if isinstance(metric_data, dict) else None
        reason = (
            metric_data.get("reason", "No justification provided")
            if isinstance(metric_data, dict)
            else "No justification provided"
        )
        if isinstance(score, (int, float)) and score is not None:
            score = max(1, min(5, int(score)))
        else:
            score = None
            reason = f"Invalid score value in judge output"
        return MetricScore(name=display_name, score=score, reason=reason)

    return EvaluationResult(
        context_relevance=_extract_metric("context_relevance", has_context),
        context_sufficiency=_extract_metric("context_sufficiency", has_context),
        faithfulness=_extract_metric("faithfulness", True),
        answer_relevance=_extract_metric("answer_relevance", True),
    )


def _error_result(error_msg: str, latency: float) -> EvaluationResult:
    """Create an EvaluationResult representing a failure."""
    def _na(name: str) -> MetricScore:
        return MetricScore(name=name, score=None, reason="Evaluation failed")

    return EvaluationResult(
        context_relevance=_na("Context Relevance"),
        context_sufficiency=_na("Context Sufficiency"),
        faithfulness=_na("Faithfulness"),
        answer_relevance=_na("Answer Relevance"),
        latency=latency,
        error=error_msg,
    )
