"""
Rule-based retrieval optimization recommendations.

Converts an EvaluationResult and retrieval metadata into actionable
tuning suggestions. No LLM calls — purely score-driven rules.
"""

from __future__ import annotations

from evaluation import EvaluationResult


def generate_retrieval_recommendations(
    evaluation: EvaluationResult,
    metadata: dict,
) -> list[str]:
    """
    Generate actionable retrieval tuning suggestions from judge scores
    and retrieval metadata.

    Args:
        evaluation: EvaluationResult from evaluate_rag_response().
        metadata:   The metrics dict stored in session state
                    (keys: strategy, num_chunks, metadata).

    Returns:
        Ordered, deduplicated list of human-readable suggestion strings.
        Empty list when evaluation has an error.
    """
    if evaluation.error:
        return []

    # ── Score extraction (None-safe) ────────────────────────────

    def _score(metric) -> int | None:
        return metric.score if metric.applicable and metric.score is not None else None

    rel   = _score(evaluation.context_relevance)
    suf   = _score(evaluation.context_sufficiency)
    faith = _score(evaluation.faithfulness)
    ans   = _score(evaluation.answer_relevance)

    # ── Metadata extraction ──────────────────────────────────────

    strategy: str        = metadata.get("strategy", "")
    num_chunks: int      = metadata.get("num_chunks", 0)
    inner: dict          = metadata.get("metadata", {})
    reranked: bool       = bool(inner.get("reranked", False))
    hybrid_alpha         = inner.get("hybrid_alpha")          # float | None
    matched_entities     = inner.get("matched_entities", [])  # list
    termination_reason   = inner.get("termination_reason", "")

    suggestions: list[str] = []

    # ── No retrieval (No RAG) ────────────────────────────────────
    if rel is None and suf is None:
        if ans is not None and ans <= 2:
            suggestions.append(
                "No RAG mode has no document context — the LLM answered from "
                "parametric memory and the answer quality is low. Switch to a "
                "retrieval mode (Vector RAG or Vector + Graph RAG) to ground "
                "responses in your documents."
            )
        elif faith is not None and faith <= 3:
            suggestions.append(
                "Without retrieved context the LLM may hallucinate. Enable any "
                "RAG mode to provide factual grounding."
            )
        else:
            suggestions.append(
                "No RAG relies entirely on the LLM's parametric knowledge. "
                "Switch to a retrieval mode for document-grounded answers."
            )
        return suggestions

    # ── All scores high — confirm configuration ──────────────────
    all_scores = [s for s in (rel, suf, faith, ans) if s is not None]
    if all_scores and min(all_scores) >= 4:
        suggestions.append(
            "Retrieval quality looks strong — current configuration is "
            "well-suited to this query type. No changes needed."
        )
        return suggestions

    # ── Rule 1: Low relevance + high sufficiency → over-retrieval ─
    if rel is not None and suf is not None and rel <= 2 and suf >= 4:
        suggestions.append(
            "Low relevance with high sufficiency indicates over-retrieval: "
            "many chunks were fetched but few are on-topic. Consider pruning "
            "the candidate set."
        )
        if not reranked:
            suggestions.append(
                "Enable cross-encoder reranking (Reranker toggle) to score and "
                "filter candidates before generation."
            )
        else:
            suggestions.append(
                "Reranking is already on — try reducing TOP_K_RETRIEVAL in "
                "config.py to limit the initial candidate pool."
            )
        if hybrid_alpha is not None and hybrid_alpha < 0.4:
            suggestions.append(
                "Hybrid Alpha is weighted toward BM25 keyword search. Shifting "
                "it to 0.6–0.8 may improve semantic precision."
            )

    # ── Rule 2: Low sufficiency → deeper retrieval needed ────────
    elif suf is not None and suf <= 2:
        suggestions.append(
            "Retrieved context is insufficient to fully answer the query."
        )
        if "Graph" not in strategy and "Agentic" not in strategy and "MACER" not in strategy:
            suggestions.append(
                "Try Vector + Graph RAG or MACER RAG to broaden coverage across "
                "both semantic and relational knowledge."
            )
        if "Graph" in strategy and not matched_entities:
            suggestions.append(
                "Graph RAG matched no entities from the query. Switch to "
                "Vector + Graph RAG so vector search fills the coverage gap."
            )
        if "MACER" in strategy and termination_reason == "max_iterations":
            suggestions.append(
                "MACER reached its iteration limit before achieving sufficient "
                "context — increase Max Iterations in the sidebar."
            )
        if num_chunks < 3:
            suggestions.append(
                f"Only {num_chunks} chunk(s) were retrieved. Increase "
                "TOP_K_RETRIEVAL in config.py for a larger candidate set."
            )

    # ── Rule 3: Moderate relevance → tuning opportunity ──────────
    if rel is not None and rel == 3:
        if not reranked:
            suggestions.append(
                "Relevance is moderate — enabling cross-encoder reranking could "
                "improve precision by re-scoring candidates with a stronger model."
            )
        if hybrid_alpha is not None and (hybrid_alpha <= 0.1 or hybrid_alpha >= 0.9):
            suggestions.append(
                "A balanced Hybrid Alpha (0.4–0.6) often improves relevance by "
                "blending vector similarity with keyword matching."
            )

    # ── Rule 4: Low faithfulness → grounding issues ──────────────
    if faith is not None and faith <= 2:
        suggestions.append(
            "Low faithfulness suggests the answer contains content not grounded "
            "in the retrieved context."
        )
        if not reranked:
            suggestions.append(
                "Enable cross-encoder reranking to surface the most relevant "
                "chunks and reduce hallucination risk."
            )
        suggestions.append(
            "Consider lowering LLM_TEMPERATURE in config.py for more "
            "deterministic, context-faithful generation."
        )

    # ── Rule 5: Low answer relevance ─────────────────────────────
    if ans is not None and ans <= 2:
        suggestions.append(
            "The answer poorly addresses the query. Verify the retrieval mode "
            "is appropriate for this query type."
        )
        if "Agentic" in strategy:
            suggestions.append(
                "In Agentic RAG, check that the agent is invoking the right "
                "tools (search_vector / search_graph) for this query."
            )

    # ── Rule 6: Graph entity matching failure ────────────────────
    if "Graph" in strategy and rel is not None and rel <= 2 and not matched_entities:
        suggestions.append(
            "Graph RAG matched no entities — query terms may not appear in the "
            "knowledge graph. Ingest more documents or switch to "
            "Vector + Graph RAG."
        )

    # ── Rule 7: Both relevance and sufficiency low → strategy upgrade
    if rel is not None and suf is not None and rel <= 2 and suf <= 2:
        if strategy not in ("FABLE RAG", "MACER RAG", "Agentic RAG"):
            suggestions.append(
                "Both relevance and sufficiency are low. FABLE RAG (hierarchical "
                "bi-path) or MACER RAG (iterative refinement) may recover context "
                "that a single-pass retrieval misses."
            )

    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    unique: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique
