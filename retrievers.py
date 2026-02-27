"""
Seven retrieval strategies using the Strategy Pattern.

BaseRetriever (ABC)
├── NoRAGRetriever        — Direct LLM call, no context
├── VectorRAGRetriever    — Hybrid vector+BM25 search with optional reranking
├── GraphRAGRetriever     — Entity extraction + KG N-hop traversal
├── HybridRAGRetriever    — Combined Vector + Graph RAG
├── AgenticRAGRetriever   — LangGraph ReAct agent with vector/graph/verify tools
├── FABLERetriever        — Hierarchical bi-path retrieval (top-down + bottom-up)
└── MACERRetriever        — Multi-agent iterative context evolution
"""

import json
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import chromadb
import networkx as nx
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from ingestion import load_fable_hierarchy, load_graph

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
            seed=random.randint(0, 2**31),
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
    Hybrid search combining ChromaDB vector similarity and BM25 keyword scores.
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

        # Load ChromaDB collection
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

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

        if self.collection.count() == 0:
            return RetrievalResult(
                context="", strategy="Vector RAG",
                metadata={"error": "No documents ingested"},
                latency=time.time() - start,
            )

        # --- Vector search via ChromaDB ---
        query_embedding = self.embedding_model.encode(query).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Build score dict: child_id -> vector_score (cosine similarity)
        vector_scores: dict[str, float] = {}
        vector_docs: dict[str, str] = {}
        vector_meta: dict[str, dict] = {}
        for i, cid in enumerate(vector_results["ids"][0]):
            # ChromaDB returns cosine distance; convert to similarity
            sim = 1.0 - vector_results["distances"][0][i]
            vector_scores[cid] = sim
            vector_docs[cid] = vector_results["documents"][0][i]
            vector_meta[cid] = vector_results["metadatas"][0][i]

        # --- BM25 keyword search ---
        bm25_scores: dict[str, float] = {}
        if self.bm25 and self.child_docs:
            tokenized_query = query.lower().split()
            raw_scores = self.bm25.get_scores(tokenized_query)
            # Normalize to [0, 1]
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
        # Normalize vector scores to [0, 1]
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
    N-hop neighbors, and retrieve the corresponding text chunks.
    """

    def __init__(self):
        super().__init__()
        self.graph = load_graph()
        self.parent_store = self._load_json(config.PARENT_STORE_PATH, {})

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        start = time.time()
        hops = kwargs.get("hops", config.GRAPH_HOPS_DEFAULT)

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

        # --- N-hop graph traversal (BFS) ---
        triplets: list[str] = []
        chunk_ids: set[str] = set()
        visited_edges: set[tuple[str, str, str]] = set()

        # Start BFS from matched entities
        current_nodes: set[str] = set(matched_entities)
        visited_nodes: set[str] = set(matched_entities)

        for hop in range(hops):
            next_nodes: set[str] = set()
            for node in current_nodes:
                # Outgoing edges
                for _, target, data in self.graph.out_edges(node, data=True):
                    rel = data.get("relation", "related_to")
                    edge_key = (node, rel, target)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        triplets.append(f"{node} -[{rel}]-> {target}")
                        if "source_chunk" in data:
                            chunk_ids.add(data["source_chunk"])
                    if target not in visited_nodes:
                        next_nodes.add(target)
                # Incoming edges
                for source, _, data in self.graph.in_edges(node, data=True):
                    rel = data.get("relation", "related_to")
                    edge_key = (source, rel, node)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        triplets.append(f"{source} -[{rel}]-> {node}")
                        if "source_chunk" in data:
                            chunk_ids.add(data["source_chunk"])
                    if source not in visited_nodes:
                        next_nodes.add(source)

            visited_nodes.update(next_nodes)
            current_nodes = next_nodes
            if not current_nodes:
                break

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
            strategy=f"Graph RAG ({hops}-hop)",
            metadata={
                "matched_entities": matched_entities,
                "triplets": triplets,
                "num_triplets": len(triplets),
                "hops": hops,
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


# ── 4. Hybrid RAG (Vector + Graph) ──────────────────────────────


class HybridRAGRetriever(BaseRetriever):
    """
    Combines Vector RAG and Graph RAG results into a single context.
    Vector search provides semantic matches; Graph search provides
    relational/structural knowledge. Deduplicates parent chunks.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder: CrossEncoder | None = None,
    ):
        super().__init__()
        self.vector_retriever = VectorRAGRetriever(embedding_model, cross_encoder)
        self.graph_retriever = GraphRAGRetriever()

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        start = time.time()

        # Run both retrievers
        v_result = self.vector_retriever.retrieve(
            query,
            alpha=kwargs.get("alpha", config.HYBRID_ALPHA_DEFAULT),
            use_reranker=kwargs.get("use_reranker", True),
        )
        g_result = self.graph_retriever.retrieve(
            query, hops=kwargs.get("hops", config.GRAPH_HOPS_DEFAULT),
        )

        # Merge and deduplicate chunks
        seen = set()
        merged_chunks: list[str] = []
        for chunk in v_result.chunks + g_result.chunks:
            chunk_key = chunk[:200]  # use prefix as dedup key
            if chunk_key not in seen:
                seen.add(chunk_key)
                merged_chunks.append(chunk)

        # Build combined context: graph triplets first, then all chunks
        context_parts: list[str] = []
        if g_result.metadata.get("triplets"):
            triplet_section = "Knowledge Graph Relationships:\n" + "\n".join(
                g_result.metadata["triplets"]
            )
            context_parts.append(triplet_section)
        if merged_chunks:
            context_parts.append(
                "Retrieved Documents:\n" + "\n\n---\n\n".join(merged_chunks)
            )

        context = "\n\n".join(context_parts)

        # Merge sources
        sources = list(dict.fromkeys(
            v_result.metadata.get("sources", [])
            + [s for s in g_result.metadata.get("matched_entities", [])]
        ))

        return RetrievalResult(
            context=context,
            chunks=merged_chunks,
            num_chunks=len(merged_chunks),
            latency=time.time() - start,
            strategy="Vector + Graph RAG",
            metadata={
                "sources": v_result.metadata.get("sources", []),
                "vector_chunks": v_result.num_chunks,
                "graph_chunks": g_result.num_chunks,
                "matched_entities": g_result.metadata.get("matched_entities", []),
                "triplets": g_result.metadata.get("triplets", []),
                "num_triplets": g_result.metadata.get("num_triplets", 0),
                "hybrid_alpha": kwargs.get("alpha", config.HYBRID_ALPHA_DEFAULT),
                "reranked": v_result.metadata.get("reranked", False),
            },
        )


# ── 5. Agentic RAG ──────────────────────────────────────────────


class AgenticRAGRetriever(BaseRetriever):
    """
    LangGraph ReAct agent that autonomously decides which tools to call
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

        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        # Build the internal retrievers for tool use
        vector_retriever = VectorRAGRetriever(self.embedding_model, self.cross_encoder)
        graph_retriever = GraphRAGRetriever()
        llm_client = self.llm_client  # capture for closure

        @tool
        def search_vector(q: str) -> str:
            """Search the vector database for documents relevant to a query. Use this for general semantic search."""
            trace.append(f"[Tool] search_vector('{q}')")
            result = vector_retriever.retrieve(q, alpha=kwargs.get("alpha", 0.5))
            if result.chunks:
                all_chunks.extend(result.chunks)
                all_context_parts.append(result.context)
                trace.append(f"  -> Found {result.num_chunks} chunks")
                return result.context[:2000]
            trace.append("  -> No results found")
            return "No relevant documents found."

        @tool
        def search_graph(q: str) -> str:
            """Search the knowledge graph for entity relationships. Use this for questions about relationships, connections, or structured facts."""
            trace.append(f"[Tool] search_graph('{q}')")
            result = graph_retriever.retrieve(q)
            if result.chunks:
                all_chunks.extend(result.chunks)
                all_context_parts.append(result.context)
                trace.append(f"  -> Found {result.num_chunks} chunks, "
                             f"{result.metadata.get('num_triplets', 0)} triplets")
                return result.context[:2000]
            trace.append("  -> No graph results found")
            return "No matching entities in the knowledge graph."

        @tool
        def verify_info(claim: str) -> str:
            """Verify a factual claim against retrieved context. Use this to double-check important facts before answering."""
            trace.append(f"[Tool] verify_info('{claim[:80]}...')")
            ctx = "\n".join(all_context_parts[:3]) if all_context_parts else "No context available."
            response = llm_client.chat.completions.create(
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
                seed=random.randint(0, 2**31),
            )
            verdict = response.choices[0].message.content
            trace.append(f"  -> Verdict: {verdict[:100]}")
            return verdict

        tools = [search_vector, search_graph, verify_info]

        # Create ReAct agent via LangGraph
        llm = ChatOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            model_kwargs={"seed": random.randint(0, 2**31)},
        )

        agent = create_react_agent(llm, tools)

        trace.append(f"[Agent] Starting with query: '{query}'")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
        )

        # Extract the final answer from agent messages
        agent_answer = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                agent_answer = msg.content
                break

        trace.append("[Agent] Final answer generated")

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


# ── 6. FABLE RAG (Hierarchical Bi-Path) ──────────────────────────


class FABLERetriever(BaseRetriever):
    """
    Forest-Based Adaptive Bi-Path Retrieval.
    Two simultaneous retrieval paths:
      Path 1 (Top-down): Navigate hierarchy from root summaries to leaf chunks
      Path 2 (Bottom-up): Standard vector search, then gather parent cluster summaries
    Merges results from both paths for comprehensive context.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder: CrossEncoder | None = None,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder
        self.hierarchy = load_fable_hierarchy()
        self.parent_store = self._load_json(config.PARENT_STORE_PATH, {})
        self.child_docs = self._load_json(config.CHILD_TEXTS_PATH, [])
        # Build child_id -> parent_id lookup
        self._child_to_parent: dict[str, str] = {
            doc["child_id"]: doc["parent_id"] for doc in self.child_docs
        }

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        start = time.time()
        trace: list[str] = []
        alpha = kwargs.get("alpha", config.HYBRID_ALPHA_DEFAULT)
        use_reranker = kwargs.get("use_reranker", True)
        top_k_branches = kwargs.get("top_k_branches", config.FABLE_TOP_K_BRANCHES)

        if not self.hierarchy or not self.hierarchy.get("clusters"):
            return RetrievalResult(
                context="", strategy="FABLE RAG",
                metadata={"error": "FABLE hierarchy not built. Please re-ingest documents."},
                latency=time.time() - start,
            )

        clusters = self.hierarchy["clusters"]
        root = self.hierarchy["root"]

        trace.append(f"[FABLE] Starting bi-path retrieval for: '{query[:80]}...'")
        trace.append(f"[FABLE] Hierarchy: {len(clusters)} clusters, "
                     f"{self.hierarchy.get('levels', 1)} levels")

        # ── Path 1: Top-down semantic navigation ──
        trace.append("\n--- Path 1: Top-Down Semantic Navigation ---")
        query_embedding = self.embedding_model.encode(query)
        topdown_child_ids = self._topdown_traverse(
            query_embedding, root, clusters, top_k_branches, trace,
        )
        trace.append(f"[Path 1] Collected {len(topdown_child_ids)} leaf child IDs")

        # ── Path 2: Bottom-up structural search ──
        trace.append("\n--- Path 2: Bottom-Up Vector Search ---")
        vector_retriever = VectorRAGRetriever(self.embedding_model, self.cross_encoder)
        v_result = vector_retriever.retrieve(query, alpha=alpha, use_reranker=use_reranker)

        # Gather child IDs from vector results via parent mapping
        bottomup_child_ids: list[str] = []
        for doc in self.child_docs:
            if doc["parent_id"] in {
                meta.get("parent_id", "")
                for cid, _ in []  # placeholder
            }:
                pass
        # Simpler: get parent IDs from vector result, find their child IDs
        bottomup_parent_ids = set()
        for chunk in v_result.chunks:
            for pid, ptext in self.parent_store.items():
                if ptext == chunk:
                    bottomup_parent_ids.add(pid)
                    break

        # Gather cluster summaries for bottom-up path
        child_to_cluster = self.hierarchy.get("child_to_cluster", {})
        bottomup_cluster_summaries: list[str] = []
        seen_clusters: set[str] = set()
        for doc in self.child_docs:
            if doc["parent_id"] in bottomup_parent_ids:
                bottomup_child_ids.append(doc["child_id"])
                cid = child_to_cluster.get(doc["child_id"], "")
                if cid and cid not in seen_clusters and cid in clusters:
                    seen_clusters.add(cid)
                    bottomup_cluster_summaries.append(clusters[cid]["summary"])

        trace.append(f"[Path 2] Vector search found {v_result.num_chunks} parent chunks")
        trace.append(f"[Path 2] Mapped to {len(bottomup_child_ids)} child IDs, "
                     f"{len(bottomup_cluster_summaries)} cluster summaries")

        # ── Merge both paths ──
        trace.append("\n--- Merging Results ---")

        # Collect all unique parent IDs from both paths
        all_parent_ids: set[str] = set()

        # From top-down: map child IDs to parent IDs
        for cid in topdown_child_ids:
            pid = self._child_to_parent.get(cid, "")
            if pid:
                all_parent_ids.add(pid)

        # From bottom-up: already have parent IDs
        all_parent_ids.update(bottomup_parent_ids)

        # Retrieve parent texts
        parent_texts = [
            self.parent_store[pid]
            for pid in all_parent_ids
            if pid in self.parent_store
        ]

        # Also include top-down cluster summaries that were traversed
        topdown_cluster_summaries: list[str] = []
        for cid in topdown_child_ids:
            cluster_id = child_to_cluster.get(cid, "")
            if cluster_id and cluster_id in clusters and cluster_id not in seen_clusters:
                seen_clusters.add(cluster_id)
                topdown_cluster_summaries.append(clusters[cluster_id]["summary"])

        all_cluster_summaries = list(dict.fromkeys(
            bottomup_cluster_summaries + topdown_cluster_summaries
        ))

        # Build context: cluster summaries first, then parent chunks
        context_parts: list[str] = []
        if all_cluster_summaries:
            context_parts.append(
                "Hierarchical Summaries:\n" + "\n\n".join(
                    f"[Cluster] {s}" for s in all_cluster_summaries
                )
            )
        if parent_texts:
            context_parts.append(
                "Source Documents:\n" + "\n\n---\n\n".join(parent_texts)
            )
        context = "\n\n".join(context_parts)

        trace.append(f"[Merged] {len(parent_texts)} unique parent chunks, "
                     f"{len(all_cluster_summaries)} cluster summaries")

        return RetrievalResult(
            context=context,
            chunks=parent_texts,
            num_chunks=len(parent_texts),
            latency=time.time() - start,
            strategy="FABLE RAG",
            trace=trace,
            metadata={
                "topdown_leaves": len(topdown_child_ids),
                "bottomup_leaves": len(bottomup_child_ids),
                "merged_parents": len(parent_texts),
                "cluster_summaries_used": len(all_cluster_summaries),
                "branches_explored": top_k_branches,
                "hierarchy_levels": self.hierarchy.get("levels", 1),
            },
        )

    def _topdown_traverse(
        self,
        query_embedding: np.ndarray,
        root: dict,
        clusters: dict,
        top_k_branches: int,
        trace: list[str],
    ) -> list[str]:
        """Navigate the hierarchy top-down, scoring clusters by cosine similarity."""
        collected_child_ids: list[str] = []
        current_candidates = root.get("children", [])

        if not current_candidates:
            return collected_child_ids

        level = 0
        while current_candidates:
            level += 1
            # Score each candidate cluster
            scored: list[tuple[str, float]] = []
            for cluster_id in current_candidates:
                cluster_info = clusters.get(cluster_id, {})
                emb = cluster_info.get("summary_embedding")
                if emb:
                    emb_arr = np.array(emb)
                    norm_q = np.linalg.norm(query_embedding)
                    norm_e = np.linalg.norm(emb_arr)
                    if norm_q > 0 and norm_e > 0:
                        sim = float(np.dot(query_embedding, emb_arr) / (norm_q * norm_e))
                    else:
                        sim = 0.0
                    scored.append((cluster_id, sim))
                else:
                    scored.append((cluster_id, 0.0))

            scored.sort(key=lambda x: x[1], reverse=True)
            selected = scored[:top_k_branches]

            trace.append(
                f"[Path 1] Level {level}: scored {len(scored)} clusters, "
                f"selected top-{len(selected)}: "
                + ", ".join(f"{cid}({sim:.3f})" for cid, sim in selected)
            )

            next_candidates: list[str] = []
            for cluster_id, _ in selected:
                cluster_info = clusters.get(cluster_id, {})
                sub_children = cluster_info.get("children", [])
                if sub_children:
                    next_candidates.extend(sub_children)
                else:
                    leaf_ids = cluster_info.get("leaf_child_ids", [])
                    collected_child_ids.extend(leaf_ids)

            current_candidates = next_candidates

        return collected_child_ids

    @staticmethod
    def _load_json(path: str, default=None):
        if not os.path.exists(path):
            return default if default is not None else {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ── 7. MACER RAG (Multi-Agent Context Evolution) ────────────────


class MACERRetriever(BaseRetriever):
    """
    Multi-Agent Context Evolution and Retrieval.
    Four-agent iterative loop:
      1. Retriever: finds chunks via vector + graph search
      2. Constructor: extracts key facts and relationships from new chunks
      3. Reflector: evaluates context quality, identifies gaps, refines query
      4. Response: synthesizes final answer when context is sufficient
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
        max_iterations = kwargs.get("max_iterations", config.MACER_MAX_ITERATIONS)
        alpha = kwargs.get("alpha", config.HYBRID_ALPHA_DEFAULT)
        use_reranker = kwargs.get("use_reranker", True)
        hops = kwargs.get("hops", config.GRAPH_HOPS_DEFAULT)

        # Build internal retrievers
        vector_retriever = VectorRAGRetriever(self.embedding_model, self.cross_encoder)
        graph_retriever = GraphRAGRetriever()

        # Accumulated state across iterations
        all_facts: list[str] = []
        all_chunks: list[str] = []
        seen_chunk_keys: set[str] = set()
        current_query = query
        termination_reason = "max_iterations"
        total_llm_calls = 0
        iteration = 0

        trace.append(f"[MACER] Starting with query: '{query}'")
        trace.append(f"[MACER] Max iterations: {max_iterations}")

        for iteration in range(1, max_iterations + 1):
            trace.append(f"\n--- Iteration {iteration}/{max_iterations} ---")

            # ── Agent 1: RETRIEVER ──
            trace.append(f"[Iter {iteration}] Retriever: searching with query='{current_query[:80]}...'")

            v_result = vector_retriever.retrieve(
                current_query, alpha=alpha, use_reranker=use_reranker,
            )
            g_result = graph_retriever.retrieve(current_query, hops=hops)

            # Collect new chunks (deduped)
            new_chunks: list[str] = []
            for chunk in v_result.chunks + g_result.chunks:
                chunk_key = chunk[:200]
                if chunk_key not in seen_chunk_keys:
                    seen_chunk_keys.add(chunk_key)
                    new_chunks.append(chunk)
                    all_chunks.append(chunk)

            new_triplets = g_result.metadata.get("triplets", [])
            trace.append(
                f"[Iter {iteration}] Retriever: found {len(new_chunks)} new chunks, "
                f"{len(new_triplets)} graph triplets"
            )

            if not new_chunks and not new_triplets:
                trace.append(f"[Iter {iteration}] Retriever: no new information found, ending loop")
                termination_reason = "no_new_info"
                break

            # ── Agent 2: CONSTRUCTOR ──
            trace.append(f"[Iter {iteration}] Constructor: extracting key facts...")
            constructor_input = "\n\n---\n\n".join(new_chunks)[:3000]
            if new_triplets:
                constructor_input += "\n\nGraph relationships:\n" + "\n".join(new_triplets[:20])

            constructor_response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": (
                        "You are a fact extraction assistant. Given retrieved passages and graph "
                        "relationships, extract the key facts and relationships as a numbered list. "
                        "Be concise. Each fact should be a single sentence."
                    )},
                    {"role": "user", "content": (
                        f"Original question: {query}\n\n"
                        f"Retrieved content:\n{constructor_input}\n\n"
                        "Extract key facts (numbered list):"
                    )},
                ],
                temperature=0.0,
                max_tokens=512,
                seed=random.randint(0, 2**31),
            )
            total_llm_calls += 1

            new_facts_text = constructor_response.choices[0].message.content.strip()
            new_facts = [
                line.strip().lstrip("0123456789.-) ")
                for line in new_facts_text.split("\n")
                if line.strip() and any(c.isalpha() for c in line)
            ]
            all_facts.extend(new_facts)
            trace.append(f"[Iter {iteration}] Constructor: extracted {len(new_facts)} facts")

            # ── Agent 3: REFLECTOR ──
            trace.append(f"[Iter {iteration}] Reflector: evaluating context sufficiency...")

            facts_summary = "\n".join(f"- {f}" for f in all_facts[-20:])
            reflector_response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": (
                        "You are a context quality evaluator. Given the original question and "
                        "accumulated facts, determine:\n"
                        "1. Is the context SUFFICIENT to answer the question? (yes/no)\n"
                        "2. What information gaps remain?\n"
                        "3. If not sufficient, provide a refined follow-up query to fill the gaps.\n\n"
                        "Respond in this exact format:\n"
                        "SUFFICIENT: yes/no\n"
                        "GAPS: <description of missing info, or 'none'>\n"
                        "REFINED_QUERY: <new query to fill gaps, or 'none'>"
                    )},
                    {"role": "user", "content": (
                        f"Original question: {query}\n\n"
                        f"Accumulated facts:\n{facts_summary}\n\n"
                        "Evaluate:"
                    )},
                ],
                temperature=0.0,
                max_tokens=300,
                seed=random.randint(0, 2**31),
            )
            total_llm_calls += 1

            reflector_text = reflector_response.choices[0].message.content.strip()
            trace.append(f"[Iter {iteration}] Reflector response:\n{reflector_text}")

            # Parse reflector output
            is_sufficient = "sufficient: yes" in reflector_text.lower()
            refined_query = current_query

            for line in reflector_text.split("\n"):
                line_stripped = line.strip()
                if line_stripped.lower().startswith("refined_query:"):
                    rq = line_stripped[len("refined_query:"):].strip()
                    if rq and rq.lower() != "none":
                        refined_query = rq

            if is_sufficient:
                trace.append(f"[Iter {iteration}] Reflector: context is SUFFICIENT, ending loop")
                termination_reason = "sufficient"
                break
            else:
                trace.append(f"[Iter {iteration}] Reflector: context insufficient, refining query")
                trace.append(f"[Iter {iteration}] New query: '{refined_query[:80]}...'")
                current_query = refined_query

        # ── Build final context ──
        context_parts: list[str] = []
        if all_facts:
            context_parts.append("Key Facts:\n" + "\n".join(f"- {f}" for f in all_facts))
        if all_chunks:
            context_parts.append("Supporting Documents:\n" + "\n\n---\n\n".join(all_chunks))
        context = "\n\n".join(context_parts)

        trace.append(f"\n[MACER] Complete: {iteration} iterations, {len(all_facts)} facts, "
                     f"{len(all_chunks)} chunks, {total_llm_calls} LLM calls")

        return RetrievalResult(
            context=context,
            chunks=all_chunks,
            num_chunks=len(all_chunks),
            latency=time.time() - start,
            strategy="MACER RAG",
            trace=trace,
            metadata={
                "iterations_completed": iteration,
                "max_iterations": max_iterations,
                "termination_reason": termination_reason,
                "total_facts": len(all_facts),
                "total_llm_calls": total_llm_calls,
                "facts": all_facts,
            },
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
    elif mode == "Vector + Graph RAG":
        return HybridRAGRetriever(embedding_model, cross_encoder)
    elif mode == "Agentic RAG":
        return AgenticRAGRetriever(embedding_model, cross_encoder)
    elif mode == "FABLE RAG":
        return FABLERetriever(embedding_model, cross_encoder)
    elif mode == "MACER RAG":
        return MACERRetriever(embedding_model, cross_encoder)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")
