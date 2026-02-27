"""
RAG Playground — Streamlit entry point.
Compare seven retrieval strategies side-by-side.
"""

import os
import tempfile

import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from ingestion import clear_all_data, ingest_documents, load_graph
from retrievers import get_retriever


# ── Page Config ──────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Playground",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helper Functions ─────────────────────────────────────────────

def _check_ingested() -> bool:
    """Check if documents have been ingested (ChromaDB + parent store exist)."""
    return (
        os.path.exists(config.CHROMA_DB_PATH)
        and os.path.exists(config.PARENT_STORE_PATH)
    )


def _display_metrics(metrics: dict, trace: list[str] | None = None):
    """Render retrieval metrics in an expander below the assistant message."""
    with st.expander("📊 Retrieval Metrics", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            st.metric("Latency", f"{metrics.get('latency', 0):.2f}s")
        with cols[1]:
            st.metric("Chunks Retrieved", metrics.get("num_chunks", 0))
        with cols[2]:
            st.metric("Strategy", metrics.get("strategy", "N/A"))

        meta = metrics.get("metadata", {})

        # Show sources
        sources = meta.get("sources", [])
        if sources:
            st.caption(f"📄 Sources: {', '.join(sources)}")

        # Vector + Graph RAG breakdown
        if "vector_chunks" in meta and "graph_chunks" in meta:
            st.caption(
                f"🔎 Vector: {meta['vector_chunks']} chunks | "
                f"🕸️ Graph: {meta['graph_chunks']} chunks"
            )

        # Graph RAG details
        if "hops" in meta:
            st.caption(f"🔗 Graph hops: {meta['hops']}")
        if "matched_entities" in meta:
            st.caption(f"🔍 Matched entities: {', '.join(meta['matched_entities'])}")
        if "triplets" in meta:
            with st.expander("🕸️ Knowledge Graph Triplets", expanded=False):
                for t in meta["triplets"]:
                    st.text(t)

        # Hybrid search details
        if "hybrid_alpha" in meta:
            st.caption(
                f"⚖️ Alpha: {meta['hybrid_alpha']:.2f} | "
                f"Reranked: {meta.get('reranked', False)}"
            )

        # FABLE details
        if "topdown_leaves" in meta:
            st.caption(
                f"🌲 Top-down: {meta['topdown_leaves']} leaves | "
                f"Bottom-up: {meta['bottomup_leaves']} leaves | "
                f"Merged: {meta['merged_parents']} parents"
            )
            if meta.get("cluster_summaries_used", 0) > 0:
                st.caption(f"📋 Cluster summaries: {meta['cluster_summaries_used']}")
            st.caption(
                f"🔀 Branches: {meta.get('branches_explored', '?')} | "
                f"Levels: {meta.get('hierarchy_levels', '?')}"
            )

        # MACER details
        if "iterations_completed" in meta:
            st.caption(
                f"🔄 Iterations: {meta['iterations_completed']}/{meta['max_iterations']} | "
                f"Termination: {meta.get('termination_reason', 'unknown')}"
            )
            st.caption(
                f"📝 Facts: {meta.get('total_facts', 0)} | "
                f"LLM calls: {meta.get('total_llm_calls', 0)}"
            )
            if meta.get("facts"):
                with st.expander("📋 Extracted Facts", expanded=False):
                    for fact in meta["facts"]:
                        st.text(f"• {fact}")

        # Errors
        if "error" in meta:
            st.warning(meta["error"])

    # Trace display (context-aware label)
    if trace:
        strategy = metrics.get("strategy", "")
        if "FABLE" in strategy:
            trace_label = "🌲 Hierarchy Navigation Trace"
        elif "MACER" in strategy:
            trace_label = "🔄 Iteration Trace"
        else:
            trace_label = "🤖 Agent Trace"
        with st.expander(trace_label, expanded=False):
            for step in trace:
                st.text(step)


def _display_evaluation(eval_data: dict):
    """Render LLM-as-a-Judge evaluation scores in an expander."""
    from evaluation import EvaluationResult

    eval_result = EvaluationResult.from_dict(eval_data)

    with st.expander("📝 LLM-as-a-Judge Evaluation", expanded=False):
        if eval_result.error:
            st.warning(f"Evaluation error: {eval_result.error}")
            return

        metrics = [
            eval_result.context_relevance,
            eval_result.context_sufficiency,
            eval_result.faithfulness,
            eval_result.answer_relevance,
        ]
        applicable = [m for m in metrics if m.applicable]

        cols = st.columns(len(applicable))
        for col, metric in zip(cols, applicable):
            with col:
                if metric.score is not None:
                    emoji = "🟢" if metric.score >= 4 else "🟡" if metric.score >= 3 else "🔴"
                    st.metric(metric.name, f"{emoji} {metric.score}/5")
                else:
                    st.metric(metric.name, "N/A")

        for metric in applicable:
            if metric.score is not None:
                st.caption(f"**{metric.name}**: {metric.reason}")

        not_applicable = [m for m in metrics if not m.applicable]
        if not_applicable:
            names = ", ".join(m.name for m in not_applicable)
            st.caption(f"_{names}: Not applicable (no retrieval performed)_")

        st.caption(f"Judge latency: {eval_result.latency:.2f}s")


# ── Cached Resources ─────────────────────────────────────────────

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer embedding model (cached across reruns)."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


@st.cache_resource
def load_cross_encoder() -> CrossEncoder:
    """Load the cross-encoder reranking model (cached across reruns)."""
    return CrossEncoder(config.RERANKER_MODEL)


# ── Session State Init ───────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = _check_ingested()
if "stats" not in st.session_state:
    st.session_state.stats = {}


# ── Sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 RAG Playground")
    st.caption("Compare retrieval strategies side-by-side")

    st.divider()

    # --- Document Upload ---
    st.subheader("📁 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload PDF or TXT files to build the knowledge base.",
    )

    col1, col2 = st.columns(2)
    with col1:
        ingest_btn = st.button("Ingest", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear Data", use_container_width=True)

    if ingest_btn and uploaded_files:
        with st.status("Ingesting documents...", expanded=True) as status:
            # Save uploaded files to a temp directory
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            for uf in uploaded_files:
                path = os.path.join(temp_dir, uf.name)
                with open(path, "wb") as f:
                    f.write(uf.getbuffer())
                file_paths.append(path)

            progress_text = st.empty()

            def on_progress(msg: str):
                progress_text.text(msg)

            embedding_model = load_embedding_model()
            stats = ingest_documents(file_paths, embedding_model, on_progress)

            st.session_state.ingested = True
            st.session_state.stats = stats
            status.update(label="Ingestion complete!", state="complete")

        st.success(
            f"✅ {stats['num_parents']} parents, {stats['num_children']} children, "
            f"{stats['num_triplets']} triplets, {stats['num_nodes']} graph nodes, "
            f"{stats.get('fable_clusters', 0)} FABLE clusters"
        )

    if clear_btn:
        clear_all_data()
        st.session_state.ingested = False
        st.session_state.stats = {}
        st.session_state.messages = []
        st.cache_resource.clear()
        st.rerun()

    # Ingestion status indicator
    if st.session_state.ingested:
        st.success("Knowledge base loaded")
        if st.session_state.stats:
            s = st.session_state.stats
            st.caption(
                f"📊 {s.get('num_parents', '?')} parents · "
                f"{s.get('num_children', '?')} children · "
                f"{s.get('num_triplets', '?')} triplets"
            )
        G = load_graph()
        if G.number_of_nodes() > 0:
            st.caption(f"🕸️ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        st.warning("No documents ingested yet")

    st.divider()

    # --- RAG Settings ---
    st.subheader("⚙️ RAG Settings")

    rag_mode = st.selectbox(
        "Retrieval Mode",
        options=[
            "No RAG", "Vector RAG", "Graph RAG", "Vector + Graph RAG",
            "Agentic RAG", "FABLE RAG", "MACER RAG",
        ],
        index=1,
        help=(
            "**No RAG**: Direct LLM call\n\n"
            "**Vector RAG**: Hybrid vector+BM25 search with optional reranking\n\n"
            "**Graph RAG**: Knowledge graph entity traversal\n\n"
            "**Vector + Graph RAG**: Combined vector search + graph traversal\n\n"
            "**Agentic RAG**: Autonomous agent with multiple tools\n\n"
            "**FABLE RAG**: Hierarchical bi-path retrieval (top-down + bottom-up)\n\n"
            "**MACER RAG**: Multi-agent iterative context evolution loop"
        ),
    )

    # Conditional settings for modes that use vector search
    use_reranker = False
    hybrid_alpha = config.HYBRID_ALPHA_DEFAULT
    graph_hops = config.GRAPH_HOPS_DEFAULT
    fable_branches = config.FABLE_TOP_K_BRANCHES
    macer_iterations = config.MACER_MAX_ITERATIONS

    if rag_mode in ("Vector RAG", "Vector + Graph RAG", "FABLE RAG", "MACER RAG"):
        use_reranker = st.toggle(
            "Cross-Encoder Reranking", value=True,
            help="Rerank top results using a cross-encoder model for better precision.",
        )
        hybrid_alpha = st.slider(
            "Hybrid Search Alpha",
            min_value=0.0, max_value=1.0,
            value=config.HYBRID_ALPHA_DEFAULT, step=0.05,
            help="1.0 = pure vector search, 0.0 = pure BM25 keyword search",
        )

    if rag_mode in ("Graph RAG", "Vector + Graph RAG", "MACER RAG"):
        graph_hops = st.slider(
            "Graph Traversal Hops",
            min_value=1, max_value=3,
            value=config.GRAPH_HOPS_DEFAULT, step=1,
            help="Number of hops from matched entities. More hops = broader context but may include less relevant info.",
        )

    if rag_mode == "FABLE RAG":
        fable_branches = st.slider(
            "Top-Down Branches",
            min_value=1, max_value=5,
            value=config.FABLE_TOP_K_BRANCHES, step=1,
            help="Number of hierarchy branches to explore in the top-down semantic path.",
        )
        st.info("Bi-path: semantic top-down + structural bottom-up hierarchy navigation", icon="🌲")

    if rag_mode == "MACER RAG":
        macer_iterations = st.slider(
            "Max Iterations",
            min_value=1, max_value=5,
            value=config.MACER_MAX_ITERATIONS, step=1,
            help="Maximum retriever-constructor-reflector loops before forcing a response.",
        )
        st.info("Four-agent loop: Retriever -> Constructor -> Reflector -> Response", icon="🔄")

    if rag_mode == "Agentic RAG":
        st.info("The agent will autonomously choose which tools to use.", icon="🤖")

    st.divider()

    # --- Evaluation Settings ---
    st.subheader("📝 Evaluation")
    enable_evaluation = st.toggle(
        "LLM-as-a-Judge",
        value=False,
        help="Run automatic quality evaluation after each query using the LLM as judge. Adds ~3-6s latency.",
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ───────────────────────────────────────────────

st.header("💬 Chat")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metrics" in msg:
            _display_metrics(msg["metrics"], msg.get("trace"))
        if msg["role"] == "assistant" and "evaluation" in msg:
            _display_evaluation(msg["evaluation"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        needs_data = rag_mode in (
            "Vector RAG", "Graph RAG", "Vector + Graph RAG",
            "Agentic RAG", "FABLE RAG", "MACER RAG",
        )

        if needs_data and not st.session_state.ingested:
            st.warning("⚠️ Please ingest documents first for this retrieval mode.")
            answer = "Please upload and ingest documents using the sidebar before using this retrieval mode."
            metrics = None
            trace = None
        else:
            # Load models
            embedding_model = load_embedding_model()
            cross_encoder = load_cross_encoder() if use_reranker else None

            retriever = get_retriever(rag_mode, embedding_model, cross_encoder)

            if rag_mode == "Agentic RAG":
                with st.status("🤖 Agent thinking...", expanded=True) as agent_status:
                    result = retriever.retrieve(prompt, alpha=hybrid_alpha)
                    if result.trace:
                        for step in result.trace:
                            st.text(step)
                    agent_status.update(
                        label=f"Agent complete ({result.latency:.2f}s)",
                        state="complete",
                    )
                answer = retriever.generate_answer(prompt, result.context)

            elif rag_mode == "FABLE RAG":
                if not os.path.exists(config.FABLE_HIERARCHY_PATH):
                    st.warning("FABLE hierarchy not found. Please re-ingest documents.")
                    answer = "Please re-ingest documents to build the FABLE hierarchy."
                    result = None
                else:
                    with st.status("🌲 FABLE: navigating hierarchy...", expanded=True) as fable_status:
                        result = retriever.retrieve(
                            prompt, alpha=hybrid_alpha, use_reranker=use_reranker,
                            top_k_branches=fable_branches,
                        )
                        if result.trace:
                            for step in result.trace:
                                st.text(step)
                        fable_status.update(
                            label=f"FABLE complete ({result.latency:.2f}s)",
                            state="complete",
                        )
                    answer = retriever.generate_answer(prompt, result.context)

            elif rag_mode == "MACER RAG":
                with st.status("🔄 MACER: iterating...", expanded=True) as macer_status:
                    result = retriever.retrieve(
                        prompt, alpha=hybrid_alpha, use_reranker=use_reranker,
                        hops=graph_hops, max_iterations=macer_iterations,
                    )
                    if result.trace:
                        for step in result.trace:
                            st.text(step)
                    macer_status.update(
                        label=(
                            f"MACER complete ({result.latency:.2f}s, "
                            f"{result.metadata.get('iterations_completed', '?')} iterations)"
                        ),
                        state="complete",
                    )
                answer = retriever.generate_answer(prompt, result.context)

            else:
                result = retriever.retrieve(
                    prompt, alpha=hybrid_alpha, use_reranker=use_reranker,
                    hops=graph_hops,
                )
                answer = retriever.generate_answer(prompt, result.context)

            if result:
                metrics = {
                    "latency": result.latency,
                    "num_chunks": result.num_chunks,
                    "strategy": result.strategy,
                    "metadata": result.metadata,
                }
                trace = result.trace
            else:
                metrics = None
                trace = None

        st.markdown(answer)

        if metrics:
            _display_metrics(metrics, trace)

        # Run LLM-as-a-Judge evaluation if enabled
        eval_result = None
        if enable_evaluation and answer:
            with st.spinner("Evaluating response quality..."):
                from evaluation import evaluate_rag_response
                eval_result = evaluate_rag_response(
                    query=prompt,
                    context=result.context if result else "",
                    answer=answer,
                    strategy=rag_mode,
                )
            _display_evaluation(eval_result.to_dict())

        # Persist to session state
        msg_data = {"role": "assistant", "content": answer}
        if metrics:
            msg_data["metrics"] = metrics
        if trace:
            msg_data["trace"] = trace
        if eval_result:
            msg_data["evaluation"] = eval_result.to_dict()
        st.session_state.messages.append(msg_data)
