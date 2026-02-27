# RAG Playground

A modular Streamlit app to compare five Retrieval-Augmented Generation strategies side-by-side:

| Mode | What it does |
|---|---|
| **No RAG** | Direct LLM call — no retrieval |
| **Vector RAG** | Hybrid vector + BM25 search with optional cross-encoder reranking |
| **Graph RAG** | Knowledge graph entity traversal (configurable 1–3 hops) |
| **Vector + Graph RAG** | Combined vector search + graph traversal for broader context |
| **Agentic RAG** | LangGraph ReAct agent that autonomously picks tools |

## Why?

This is a **demonstration tool** to make it easy to see *when and why* different RAG strategies shine. Graph RAG excels at relational queries ("Who founded X?"), Vector RAG at semantic similarity, and Agentic RAG at multi-step reasoning.

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Any OpenAI-compatible API (default: LM Studio at `localhost:1234`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector Store**: ChromaDB (persistent, cosine similarity)
- **Knowledge Graph**: NetworkX (persisted to JSON)
- **Agent**: LangGraph ReAct with `search_vector`, `search_graph`, `verify_info` tools
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

### Vector RAG / Vector + Graph RAG Controls
- **Hybrid Alpha** slider: `1.0` = pure vector, `0.0` = pure BM25
- **Reranking** toggle: enable/disable cross-encoder reranking

### Graph RAG / Vector + Graph RAG Controls
- **Graph Traversal Hops** slider: 1–3 hops from matched entities (more hops = broader but potentially noisier context)

### Agentic RAG
The agent's thinking process is shown in a live **Agent Trace** panel — you can see which tools it calls and why.

## Architecture

```
app.py            → Streamlit UI (sidebar, chat, metrics)
config.py         → All configuration constants
ingestion.py      → Document loading, chunking, embedding, KG extraction
retrievers.py     → Strategy pattern: BaseRetriever + 5 implementations
```

### Key Design

- **Strategy Pattern** — all retrievers share a common `retrieve(query) -> RetrievalResult` interface, making it easy to add new strategies
- **Parent-Document Retrieval** — child chunks (300 tokens) are indexed for search, but parent chunks (800 tokens) are returned to the LLM for richer context
- **Knowledge Graph** — the LLM extracts `(Subject, Predicate, Object)` triplets during ingestion, stored in NetworkX and persisted to disk

## Configuration

Edit `config.py` to change:

```python
LLM_BASE_URL = "http://localhost:1234/v1"   # Your LLM endpoint
LLM_MODEL_NAME = "qwen2.5-7b-instruct"      # Your model name
CHILD_CHUNK_SIZE = 300                        # Tokens for search index
PARENT_CHUNK_SIZE = 800                       # Tokens for LLM context
TOP_K_RETRIEVAL = 10                          # Initial candidates
TOP_K_RERANK = 5                              # After reranking
GRAPH_HOPS_DEFAULT = 1                        # Graph traversal depth (1–3)
```

## License

MIT
