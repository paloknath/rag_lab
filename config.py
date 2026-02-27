"""All configuration constants for the RAG Playground."""

# ── LLM ──────────────────────────────────────────────────────────
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"            # LM Studio ignores the key but the SDK requires one
LLM_MODEL_NAME = "qwen2.5-7b-instruct"

# ── Embeddings ───────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Reranker ─────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Storage Paths ────────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
GRAPH_STORE_PATH = "./graph_store/kg.json"
PARENT_STORE_PATH = "./graph_store/parents.json"
CHILD_TEXTS_PATH = "./graph_store/child_texts.json"

# ── Chunking ─────────────────────────────────────────────────────
CHILD_CHUNK_SIZE = 300               # Tokens — indexed for vector search
PARENT_CHUNK_SIZE = 800              # Tokens — returned as LLM context
CHUNK_OVERLAP = 50                   # Token overlap between chunks

# ── Retrieval ────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 10                 # Initial candidates from vector/BM25
TOP_K_RERANK = 5                     # Final results after cross-encoder reranking
HYBRID_ALPHA_DEFAULT = 0.5           # 1.0 = pure vector, 0.0 = pure BM25
GRAPH_HOPS_DEFAULT = 1               # Number of hops for graph traversal (1–3)

# ── FABLE (Hierarchical Retrieval) ──────────────────────────────
FABLE_HIERARCHY_PATH = "./graph_store/fable_hierarchy.json"
FABLE_MIN_CLUSTER_SIZE = 2           # Minimum children per cluster
FABLE_MAX_CLUSTERS = 10              # Upper bound on clusters per level
FABLE_NUM_LEVELS = 2                 # 1 = flat clusters, 2 = clusters-of-clusters
FABLE_TOP_K_BRANCHES = 3            # Branches explored in top-down path
FABLE_SUMMARY_MAX_TOKENS = 200       # Max tokens for cluster summaries

# ── MACER (Multi-Agent Iterative Retrieval) ─────────────────────
MACER_MAX_ITERATIONS = 3             # Max retriever-constructor-reflector loops

# ── LLM Generation ──────────────────────────────────────────────
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# ── ChromaDB ─────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_playground"
