"""
Four retrieval strategies using the Strategy Pattern.

BaseRetriever (ABC)
├── NoRAGRetriever        — Direct LLM call, no context
├── VectorRAGRetriever    — Hybrid vector+BM25 search with optional reranking
├── GraphRAGRetriever     — Entity extraction + KG 1-hop traversal
└── AgenticRAGRetriever   — LangChain ReAct agent with vector/graph/verify tools
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from ingestion import load_graph, vector_search

# ── Result Data Class ────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """Standardized output from any retriever."""
    context: str                          # Formatted text for LLM prompt
    chunks: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    num_chunks: int = 0
    latency: float = 0.0
    strategy: str = ""
    trace: list[str] | None = None       # Agent thinking steps (AgenticRAG only)


# ── Abstract Base ────────────────────────────────────────────────


class BaseRetriever(ABC):
    """Abstract base for all retrieval strategies."""

    def __init__(self):
        self.llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """Retrieve relevant context for the query."""
        ...

    def generate_answer(self, query: str, context: str) -> str:
        """Call the LLM with the query and retrieved context."""
        if context:
            system_msg = (
                "You are a helpful assistant. Answer the user's question based on "
                "the provided context. If the context doesn't contain enough information, "
                "say so and provide your best answer.\n\n"
                f"Context:\n{context}"
            )
        else:
            system_msg = "You are a helpful assistant. Answer the user's question."

        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content


# ── 1. No RAG ───────────────────────────────────────────────────


class NoRAGRetriever(BaseRetriever):
    """Direct LLM call with no retrieval."""

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        start = time.time()
        return RetrievalResult(
            context="",
            chunks=[],
            num_chunks=0,
            latency=time.time() - start,
            strategy="No RAG",
        )


# ── 2. Vector RAG (Hybrid + Reranker) ───────────────────────────


class VectorRAGRetriever(BaseRetriever):
    """
    Hybrid search combining numpy vector similarity and BM25 keyword scores.
    Optional cross-encoder reranking on the merged candidate set.
    Returns parent chunks (not child chunks) for richer LLM context.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder: CrossEncoder | None = None,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder

        # Load parent store
        self.parent_store = self._load_json(config.PARENT_STORE_PATH, {})

        # Load child texts and build BM25 index
        self.child_docs = self._load_json(config.CHILD_TEXTS_PATH, [])
        if self.child_docs:
            tokenized = [doc["text"].lower().split() for doc in self.child_docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def retrieve(
        self,
        query: str,
        alpha: float = config.HYBRID_ALPHA_DEFAULT,
        use_reranker: bool = True,
        **kwargs,
    ) -> RetrievalResult:
        start = time.time()
        top_k = config.TOP_K_RETRIEVAL

        if not self.child_docs:
            return RetrievalResult(
                context="", strategy="Vector RAG",
                metadata={"error": "No documents ingested"},
                latency=time.time() - start,
            )

        # --- Vector search via numpy cosine similarity ---
        query_embedding = self.embedding_model.encode(query)
        vector_hits = vector_search(query_embedding, top_k=top_k)

        # Build score dicts keyed by child_id
        vector_scores: dict[str, float] = {}
        vector_docs: dict[str, str] = {}
        vector_meta: dict[str, dict] = {}
        for idx, sim in vector_hits:
            if idx < len(self.child_docs):
                doc = self.child_docs[idx]
                cid = doc["child_id"]
                vector_scores[cid] = sim
                vector_docs[cid] = doc["text"]
                vector_meta[cid] = {
                    "parent_id": doc["parent_id"],
                    "source_file": doc["source_file"],
                }

        # --- BM25 keyword search ---
        bm25_scores: dict[str, float] = {}
        if self.bm25 and self.child_docs:
            tokenized_query = query.lower().split()
            raw_scores = self.bm25.get_scores(tokenized_query)
            max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
            for idx, score in enumerate(raw_scores):
                cid = self.child_docs[idx]["child_id"]
                bm25_scores[cid] = score / max_score
                if cid not in vector_docs:
                    vector_docs[cid] = self.child_docs[idx]["text"]
                    vector_meta[cid] = {
                        "parent_id": self.child_docs[idx]["parent_id"],
                        "source_file": self.child_docs[idx]["source_file"],
                    }

        # --- Hybrid merge ---
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        if vector_scores:
            v_max = max(vector_scores.values()) or 1.0
            v_min = min(vector_scores.values())
            v_range = v_max - v_min if v_max != v_min else 1.0
        else:
            v_min, v_range = 0.0, 1.0

        hybrid: list[tuple[str, float]] = []
        for cid in all_ids:
            v_score = (vector_scores.get(cid, 0.0) - v_min) / v_range if cid in vector_scores else 0.0
            b_score = bm25_scores.get(cid, 0.0)
            combined = alpha * v_score + (1.0 - alpha) * b_score
            hybrid.append((cid, combined))

        hybrid.sort(key=lambda x: x[1], reverse=True)
        candidates = hybrid[:top_k]

        # --- Optional cross-encoder reranking ---
        if use_reranker and self.cross_encoder and candidates:
            pairs = [(query, vector_docs[cid]) for cid, _ in candidates]
            rerank_scores = self.cross_encoder.predict(pairs)
            reranked = sorted(
                zip(candidates, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            candidates = [item[0] for item in reranked[:config.TOP_K_RERANK]]

        # --- Map child → parent chunks (deduplicated) ---
        seen_parents: set[str] = set()
        parent_texts: list[str] = []
        sources: list[str] = []

        for cid, _ in candidates:
            meta = vector_meta.get(cid, {})
            pid = meta.get("parent_id", "")
            if pid and pid not in seen_parents and pid in self.parent_store:
                seen_parents.add(pid)
                parent_texts.append(self.parent_store[pid])
                src = meta.get("source_file", "unknown")
                if src not in sources:
                    sources.append(src)

        context = "\n\n---\n\n".join(parent_texts)

        return RetrievalResult(
            context=context,
            chunks=parent_texts,
            num_chunks=len(parent_texts),
            latency=time.time() - start,
            strategy="Vector RAG" + (" + Reranker" if use_reranker and self.cross_encoder else ""),
            metadata={
                "sources": sources,
                "hybrid_alpha": alpha,
                "reranked": use_reranker and self.cross_encoder is not None,
                "candidates_before_rerank": len(hybrid[:top_k]),
            },
        )

    @staticmethod
    def _load_json(path: str, default=None):
        if not os.path.exists(path):
            return default if default is not None else {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ── 3. Graph RAG ────────────────────────────────────────────────


class GraphRAGRetriever(BaseRetriever):
    """
    Extract entities from the query, traverse the Knowledge Graph for
    1-hop neighbors, and retrieve the corresponding text chunks.
    """

    def __init__(self):
        super().__init__()
        self.graph = load_graph()
        self.parent_store = self._load_json(config.PARENT_STORE_PATH, {})

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        start = time.time()

        if self.graph.number_of_nodes() == 0:
            return RetrievalResult(
                context="", strategy="Graph RAG",
                metadata={"error": "Knowledge graph is empty. Please ingest documents first."},
                latency=time.time() - start,
            )

        # --- Entity extraction via n-gram matching against graph nodes ---
        matched_entities = self._extract_entities(query)

        if not matched_entities:
            return RetrievalResult(
                context="",
                strategy="Graph RAG",
                metadata={"error": "No matching entities found in the knowledge graph."},
                latency=time.time() - start,
            )

        # --- 1-hop graph traversal ---
        triplets: list[str] = []
        chunk_ids: set[str] = set()

        for entity in matched_entities:
            # Outgoing edges
            for _, target, data in self.graph.out_edges(entity, data=True):
                rel = data.get("relation", "related_to")
                triplets.append(f"{entity} -[{rel}]-> {target}")
                if "source_chunk" in data:
                    chunk_ids.add(data["source_chunk"])
            # Incoming edges
            for source, _, data in self.graph.in_edges(entity, data=True):
                rel = data.get("relation", "related_to")
                triplets.append(f"{source} -[{rel}]-> {entity}")
                if "source_chunk" in data:
                    chunk_ids.add(data["source_chunk"])

        # --- Retrieve parent texts for graph-connected chunks ---
        parent_texts = [
            self.parent_store[cid]
            for cid in chunk_ids
            if cid in self.parent_store
        ]

        # Format context: triplets first, then supporting text
        triplet_section = "Knowledge Graph Relationships:\n" + "\n".join(triplets)
        text_section = "\n\n---\n\n".join(parent_texts) if parent_texts else ""
        context = triplet_section
        if text_section:
            context += "\n\nSupporting Text:\n" + text_section

        return RetrievalResult(
            context=context,
            chunks=parent_texts,
            num_chunks=len(parent_texts),
            latency=time.time() - start,
            strategy="Graph RAG",
            metadata={
                "matched_entities": matched_entities,
                "triplets": triplets,
                "num_triplets": len(triplets),
            },
        )

    def _extract_entities(self, query: str) -> list[str]:
        """Match query tokens and n-grams against graph node names."""
        query_lower = query.lower()
        words = query_lower.split()
        graph_nodes = set(self.graph.nodes())
        matched: set[str] = set()

        # Try n-grams from longest to shortest (3-gram, 2-gram, 1-gram)
        for n in range(min(3, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                if ngram in graph_nodes:
                    matched.add(ngram)

        return list(matched)

    @staticmethod
    def _load_json(path: str, default=None):
        if not os.path.exists(path):
            return default if default is not None else {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ── 4. Agentic RAG ──────────────────────────────────────────────


class AgenticRAGRetriever(BaseRetriever):
    """
    LangChain ReAct agent that autonomously decides which tools to call
    (search_vector, search_graph, verify_info) before producing a final answer.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder: CrossEncoder | None = None,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        start = time.time()
        trace: list[str] = []
        all_chunks: list[str] = []
        all_context_parts: list[str] = []

        try:
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain.tools import Tool
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import ChatOpenAI

            # Build the internal retrievers for tool use
            vector_retriever = VectorRAGRetriever(self.embedding_model, self.cross_encoder)
            graph_retriever = GraphRAGRetriever()

            # Define agent tools
            def search_vector_fn(q: str) -> str:
                trace.append(f"[Tool] search_vector('{q}')")
                result = vector_retriever.retrieve(q, alpha=kwargs.get("alpha", 0.5))
                if result.chunks:
                    all_chunks.extend(result.chunks)
                    all_context_parts.append(result.context)
                    trace.append(f"  → Found {result.num_chunks} chunks")
                    return result.context[:2000]
                trace.append("  → No results found")
                return "No relevant documents found."

            def search_graph_fn(q: str) -> str:
                trace.append(f"[Tool] search_graph('{q}')")
                result = graph_retriever.retrieve(q)
                if result.chunks:
                    all_chunks.extend(result.chunks)
                    all_context_parts.append(result.context)
                    trace.append(f"  → Found {result.num_chunks} chunks, "
                                 f"{result.metadata.get('num_triplets', 0)} triplets")
                    return result.context[:2000]
                trace.append("  → No graph results found")
                return "No matching entities in the knowledge graph."

            def verify_info_fn(claim: str) -> str:
                trace.append(f"[Tool] verify_info('{claim[:80]}...')")
                ctx = "\n".join(all_context_parts[:3]) if all_context_parts else "No context available."
                response = self.llm_client.chat.completions.create(
                    model=config.LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": (
                            "You are a fact-checker. Given the context below, verify "
                            "whether the claim is supported, contradicted, or unverifiable.\n\n"
                            f"Context:\n{ctx[:3000]}"
                        )},
                        {"role": "user", "content": f"Claim: {claim}"},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                verdict = response.choices[0].message.content
                trace.append(f"  → Verdict: {verdict[:100]}")
                return verdict

            tools = [
                Tool(name="search_vector", func=search_vector_fn,
                     description="Search the vector database for documents relevant to a query. Use this for general semantic search."),
                Tool(name="search_graph", func=search_graph_fn,
                     description="Search the knowledge graph for entity relationships. Use this for questions about relationships, connections, or structured facts."),
                Tool(name="verify_info", func=verify_info_fn,
                     description="Verify a factual claim against retrieved context. Use this to double-check important facts before answering."),
            ]

            # Create ReAct agent
            llm = ChatOpenAI(
                base_url=config.LLM_BASE_URL,
                api_key=config.LLM_API_KEY,
                model=config.LLM_MODEL_NAME,
                temperature=config.LLM_TEMPERATURE,
            )

            react_prompt = PromptTemplate.from_template(
                """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
            )

            agent = create_react_agent(llm, tools, react_prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                max_iterations=5,
                handle_parsing_errors=True,
            )

            trace.append(f"[Agent] Starting with query: '{query}'")
            result = executor.invoke({"input": query})
            agent_answer = result.get("output", "")
            trace.append(f"[Agent] Final answer generated")

            context = "\n\n---\n\n".join(all_context_parts) if all_context_parts else ""

            return RetrievalResult(
                context=context,
                chunks=all_chunks,
                num_chunks=len(all_chunks),
                latency=time.time() - start,
                strategy="Agentic RAG",
                trace=trace,
                metadata={"agent_answer": agent_answer},
            )

        except Exception as e:
            trace.append(f"[Error] Agent failed: {str(e)}")
            # Fallback: use both retrievers directly
            trace.append("[Fallback] Running vector + graph search directly")

            vector_r = VectorRAGRetriever(self.embedding_model, self.cross_encoder)
            graph_r = GraphRAGRetriever()

            v_result = vector_r.retrieve(query, alpha=kwargs.get("alpha", 0.5))
            g_result = graph_r.retrieve(query)

            combined_chunks = v_result.chunks + g_result.chunks
            combined_context = v_result.context
            if g_result.context:
                combined_context += "\n\n---\n\n" + g_result.context

            return RetrievalResult(
                context=combined_context,
                chunks=combined_chunks,
                num_chunks=len(combined_chunks),
                latency=time.time() - start,
                strategy="Agentic RAG (Fallback)",
                trace=trace,
                metadata={"fallback": True, "error": str(e)},
            )


# ── Factory ──────────────────────────────────────────────────────


def get_retriever(
    mode: str,
    embedding_model: SentenceTransformer | None = None,
    cross_encoder: CrossEncoder | None = None,
) -> BaseRetriever:
    """Factory function to get the appropriate retriever by mode name."""
    if mode == "No RAG":
        return NoRAGRetriever()
    elif mode == "Vector RAG":
        return VectorRAGRetriever(embedding_model, cross_encoder)
    elif mode == "Graph RAG":
        return GraphRAGRetriever()
    elif mode == "Agentic RAG":
        return AgenticRAGRetriever(embedding_model, cross_encoder)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")
