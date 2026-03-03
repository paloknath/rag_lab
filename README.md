# RAG Playground

A modular Streamlit app to compare seven Retrieval-Augmented Generation strategies side-by-side:

| Mode | What it does |
|---|---|
| **No RAG** | Direct LLM call — no retrieval |
| **Vector RAG** | Hybrid vector + BM25 search with optional cross-encoder reranking |
| **Graph RAG** | Knowledge graph entity traversal (configurable 1–3 hops) |
| **Vector + Graph RAG** | Combined vector search + graph traversal for broader context |
| **Agentic RAG** | LangGraph ReAct agent that autonomously picks tools |
| **FABLE RAG** | Hierarchical bi-path retrieval — top-down semantic + bottom-up vector |
| **MACER RAG** | Multi-agent iterative context evolution (Retriever → Constructor → Reflector loop) |

## Why?

This is a **demonstration tool** to make it easy to see *when and why* different RAG strategies shine. Graph RAG excels at relational queries ("Who founded X?"), Vector RAG at semantic similarity, Agentic RAG at multi-step reasoning, and MACER at progressively refining context when one pass isn't enough.

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Any OpenAI-compatible API (default: LM Studio at `localhost:1234`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector Store**: ChromaDB (persistent, cosine similarity)
- **Knowledge Graph**: NetworkX (persisted to JSON)
- **Agent**: LangGraph ReAct with `search_vector`, `search_graph`, `verify_info` tools
- **Clustering**: scikit-learn k-means (FABLE hierarchy)
- **Python**: 3.13 required (3.14 breaks Pydantic v1 used by ChromaDB/LangChain)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/paloknath/rag_lab.git
cd rag_lab

# 2. Create venv with Python 3.13
python3.13 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Start your LLM server (e.g. LM Studio on port 1234)

# 5. Run
streamlit run app.py --server.headless true
```

Open **http://localhost:8501** in your browser.

## Usage

1. **Upload** PDF or TXT files via the sidebar and click **Ingest**
2. **Select** a retrieval mode from the dropdown
3. **Chat** — ask questions about your documents
4. **Compare** — switch modes and ask the same question to see the difference

### Vector RAG / Vector + Graph RAG / FABLE RAG / MACER RAG Controls
- **Hybrid Alpha** slider: `1.0` = pure vector, `0.0` = pure BM25
- **Reranking** toggle: enable/disable cross-encoder reranking

### Graph RAG / Vector + Graph RAG / MACER RAG Controls
- **Graph Traversal Hops** slider: 1–3 hops from matched entities (more hops = broader but potentially noisier context)

### FABLE RAG
The **Top-Down Branches** slider controls how many cluster branches are explored during semantic hierarchy navigation. FABLE runs two retrieval paths simultaneously — a top-down semantic traversal from cluster summaries to leaf chunks, and a bottom-up vector search that gathers cluster context for matched documents.

The bi-path trace is shown in the **Hierarchy Navigation Trace** panel.

### MACER RAG
The **Max Iterations** slider caps the retriever-constructor-reflector loop. The loop exits early when the Reflector agent judges the accumulated context sufficient. The full iteration trace is shown in the **Iteration Trace** panel.

### Agentic RAG
The agent's thinking process is shown in a live **Agent Trace** panel — you can see which tools it calls and why.

## Evaluation

### LLM-as-a-Judge
Enable the **LLM-as-a-Judge** toggle in the sidebar to automatically score each response on four metrics after every query:

| Metric | What it measures |
|---|---|
| **Context Relevance** | How relevant are the retrieved passages to the query? (1–5) |
| **Context Sufficiency** | Do the passages provide enough information to fully answer? (1–5) |
| **Faithfulness** | Is the answer grounded in the context without hallucination? (1–5) |
| **Answer Relevance** | Does the answer directly address the query? (1–5) |

Scores are colour-coded: 🟢 ≥ 4, 🟡 ≥ 3, 🔴 < 3. Adds ~3–6s latency.

### Context Noise Analysis
Enable the **Context Noise Analysis** toggle to classify every retrieved chunk as **relevant**, **partial**, or **irrelevant** relative to the query. Results are shown after the answer is rendered (non-blocking) with colour-coded per-chunk highlights:

- ✅ **Relevant** — directly supports the query
- ⚠️ **Partial** — tangentially related or partially useful
- ❌ **Irrelevant** — off-topic noise

The **Noise Ratio** (irrelevant chunks / total chunks) gives an at-a-glance signal of retrieval precision. Adds ~2–4s latency.

## Architecture

```
app.py              → Streamlit UI (sidebar, chat, metrics, evaluation, noise analysis)
config.py           → All configuration constants
ingestion.py        → Document loading, chunking, embedding, KG + FABLE hierarchy
retrievers.py       → Strategy pattern: BaseRetriever + 7 implementations
evaluation.py       → LLM-as-a-Judge: 4-metric quality scoring
noise_analysis.py   → Context Noise Analyzer: per-chunk relevance classification
```

### Key Design

- **Strategy Pattern** — all retrievers share a common `retrieve(query) -> RetrievalResult` interface, making it easy to add new strategies
- **Parent-Document Retrieval** — child chunks (300 tokens) are indexed for search, but parent chunks (800 tokens) are returned to the LLM for richer context
- **Knowledge Graph** — the LLM extracts `(Subject, Predicate, Object)` triplets during ingestion, stored in NetworkX and persisted to disk
- **FABLE Hierarchy** — k-means cluster tree with LLM-generated summaries built at ingestion time and queried via cosine similarity at retrieval time
- **MACER Loop** — iterative context refinement: each cycle retrieves new chunks, extracts facts, and evaluates sufficiency before deciding to continue or stop

## Configuration

Edit `config.py` to change:

```python
LLM_BASE_URL = "http://localhost:1234/v1"   # Any OpenAI-compatible endpoint
LLM_MODEL_NAME = "your-model-name"           # Model loaded in your LLM server
CHILD_CHUNK_SIZE = 300                        # Tokens for search index
PARENT_CHUNK_SIZE = 800                       # Tokens for LLM context
TOP_K_RETRIEVAL = 10                          # Initial candidates
TOP_K_RERANK = 5                              # After reranking
GRAPH_HOPS_DEFAULT = 1                        # Graph traversal depth (1–3)
FABLE_NUM_LEVELS = 2                          # Hierarchy depth
FABLE_TOP_K_BRANCHES = 3                      # Branches explored top-down
MACER_MAX_ITERATIONS = 3                      # Max iterative refinement loops
```

## License

MIT
