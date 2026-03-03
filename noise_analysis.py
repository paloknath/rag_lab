"""
Context Noise Analyzer — evaluates relevance of each retrieved chunk
relative to the query using a single LLM call at temperature=0.
"""

import json
import random
from dataclasses import dataclass, field

from openai import OpenAI

import config


@dataclass
class ChunkRelevance:
    chunk_id: str
    relevance: str  # relevant | partial | irrelevant
    reason: str


@dataclass
class NoiseAnalysisResult:
    noise_ratio: float
    relevant_count: int
    partial_count: int
    irrelevant_count: int
    details: list[ChunkRelevance] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "noise_ratio": self.noise_ratio,
            "relevant_count": self.relevant_count,
            "partial_count": self.partial_count,
            "irrelevant_count": self.irrelevant_count,
            "details": [
                {
                    "chunk_id": d.chunk_id,
                    "relevance": d.relevance,
                    "reason": d.reason,
                }
                for d in self.details
            ],
        }


def analyze_context_noise(
    query: str,
    retrieved_chunks: list[str],
) -> NoiseAnalysisResult:
    """
    Evaluate relevance of each retrieved chunk relative to the query.

    Makes a single LLM call returning structured JSON.  Each chunk is
    labelled as 'relevant', 'partial', or 'irrelevant' with a short reason.
    Returns a NoiseAnalysisResult with per-chunk detail and aggregate counts.
    """
    if not retrieved_chunks:
        return NoiseAnalysisResult(
            noise_ratio=0.0,
            relevant_count=0,
            partial_count=0,
            irrelevant_count=0,
            details=[],
        )

    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    # Build numbered chunk summaries (truncated to keep prompt manageable)
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        preview = chunk[:400].replace("\n", " ").strip()
        chunks_text += f'\n[chunk_{i}]: "{preview}"\n'

    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved chunks:\n{chunks_text}\n"
        "For each chunk, classify its relevance to the query as exactly one of:\n"
        '  "relevant"   — directly answers or strongly supports the query\n'
        '  "partial"    — tangentially related or only partially useful\n'
        '  "irrelevant" — off-topic or unhelpful for answering the query\n\n'
        "Respond ONLY with valid JSON — no markdown, no extra text:\n"
        '{"chunks": [{"chunk_id": "chunk_0", "relevance": "relevant", "reason": "<one sentence>"}, ...]}'
    )

    response = client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a retrieval quality evaluator. "
                    "Assess each chunk's relevance strictly and return ONLY valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
        seed=random.randint(0, 2**31),
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON; strip markdown fences if the model wraps its output
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        chunk_data: list[dict] = data.get("chunks", [])
    except (json.JSONDecodeError, KeyError, IndexError):
        # On parse failure, mark all chunks as partial
        chunk_data = [
            {
                "chunk_id": f"chunk_{i}",
                "relevance": "partial",
                "reason": "Could not parse LLM response",
            }
            for i in range(len(retrieved_chunks))
        ]

    # Build a lookup by chunk_id
    chunk_map: dict[str, ChunkRelevance] = {}
    for item in chunk_data:
        cid = item.get("chunk_id", "")
        rel = item.get("relevance", "partial").lower().strip()
        if rel not in ("relevant", "partial", "irrelevant"):
            rel = "partial"
        chunk_map[cid] = ChunkRelevance(
            chunk_id=cid,
            relevance=rel,
            reason=item.get("reason", ""),
        )

    # Ensure every input chunk has an entry (fill gaps with "partial")
    details: list[ChunkRelevance] = []
    for i in range(len(retrieved_chunks)):
        cid = f"chunk_{i}"
        details.append(
            chunk_map.get(
                cid,
                ChunkRelevance(chunk_id=cid, relevance="partial", reason="Not evaluated"),
            )
        )

    relevant_count = sum(1 for d in details if d.relevance == "relevant")
    partial_count = sum(1 for d in details if d.relevance == "partial")
    irrelevant_count = sum(1 for d in details if d.relevance == "irrelevant")
    total = len(details)
    noise_ratio = irrelevant_count / total if total > 0 else 0.0

    return NoiseAnalysisResult(
        noise_ratio=noise_ratio,
        relevant_count=relevant_count,
        partial_count=partial_count,
        irrelevant_count=irrelevant_count,
        details=details,
    )
