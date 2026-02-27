# Add FABLE RAG & MACER RAG to RAG Playground

## Context
The RAG Playground currently has 5 retrieval modes. The user wants to add 2 more advanced strategies as new dropdown options:

1. **FABLE RAG** (Forest-Based Adaptive Bi-Path Retrieval) — hierarchical clustering of chunks with dual-path retrieval: top-down LLM-guided navigation through cluster summaries + bottom-up vector search at leaf level. Based on arXiv:2601.18116.

2. **MACER RAG** (Multi-Agent Context Evolution and Retrieval) — iterative 4-agent loop (Retriever → Constructor → Reflector → Response) where the query and context co-evolve across iterations until context is sufficient. Based on arXiv:2509.21710.

---

## Files to Modify

| File | Changes |
|---|---|
| `config.py` | Add ~10 new constants (FABLE hierarchy params, MACER iteration params) |
| `requirements.txt` | Add `scipy>=1.11.0` |
| `ingestion.py` | Add `build_fable_hierarchy()` + `_llm_summarize()`, hook into pipeline |
| `retrievers.py` | Add `FABLERetriever` + `MACERRetriever` classes, update factory + docstring |
| `app.py` | Add 2 dropdown options, sidebar controls, `st.status` blocks, metrics display |

---

## Step 1: `config.py` — New Constants

Add after `GRAPH_HOPS_DEFAULT`:

```python
# ── FABLE (Hierarchical Retrieval) ──────────────────────────────
FABLE_HIERARCHY_PATH = "./graph_store/fable_hierarchy.json"
FABLE_MIN_CLUSTER_SIZE = 2
FABLE_MAX_CLUSTERS = 10
FABLE_NUM_LEVELS = 2                 # 1 = flat clusters, 2 = clusters-of-clusters
FABLE_TOP_K_BRANCHES = 3            # Branches explored in top-down path
FABLE_SUMMARY_MAX_TOKENS = 200

# ── MACER (Multi-Agent Iterative Retrieval) ─────────────────────
MACER_MAX_ITERATIONS = 3
```

---

## Step 2: `requirements.txt` — Add scipy

Add `scipy>=1.11.0` for agglomerative clustering in FABLE hierarchy building.

---

## Step 3: `ingestion.py` — FABLE Hierarchy Building

### New imports
```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
```

### New functions

**`_llm_summarize(text, llm_client) -> str`** — Concise LLM summarization helper.

**`build_fable_hierarchy(children, embedding_model, llm_client, progress_callback) -> dict`**:
1. Encode all child chunks with the embedding model
2. Agglomerative clustering using cosine distance (`scipy.cluster.hierarchy`)
3. For each cluster: concatenate child texts, LLM-summarize, embed the summary
4. If `FABLE_NUM_LEVELS >= 2` and enough clusters: cluster the cluster summaries into super-clusters, summarize again
5. Build root node summarizing all top-level clusters
6. Persist hierarchy to `FABLE_HIERARCHY_PATH` as JSON

**Hierarchy JSON schema:**
```json
{
  "levels": 2,
  "root": {"summary": "...", "summary_embedding": [...], "children": ["cluster_1_1", ...]},
  "clusters": {
    "cluster_0_1": {
      "level": 0, "summary": "...", "summary_embedding": [...],
      "children": [], "leaf_child_ids": ["uuid1", "uuid2"]
    },
    "cluster_1_1": {
      "level": 1, "summary": "...", "summary_embedding": [...],
      "children": ["cluster_0_1", "cluster_0_2"], "leaf_child_ids": [...]
    }
  },
  "child_to_cluster": {"uuid1": "cluster_0_1", ...}
}
```

### Modify `ingest_documents()`
- Add Phase 4 after KG building: call `build_fable_hierarchy()`
- Add `"fable_clusters"` to returned stats dict

No changes to `clear_all_data()` — hierarchy file is under `graph_store/` which is already cleaned.

---

## Step 4: `retrievers.py` — Two New Retriever Classes

### 4a. `FABLERetriever` (Hierarchical Bi-Path)

**Constructor**: loads hierarchy JSON, parent store, composes `VectorRAGRetriever` internally.

**`retrieve(query, alpha, use_reranker, top_k_branches)`**:
- **Path 1 — Top-down (semantic)**: Embed query → cosine similarity with root children embeddings → select top-K branches → drill into sub-clusters → repeat until leaf level → collect leaf child_ids
- **Path 2 — Bottom-up (structural)**: Use `VectorRAGRetriever.retrieve()` for standard hybrid search → get child_ids from results → look up parent clusters via `child_to_cluster` → gather cluster summaries
- **Merge**: Deduplicate child_ids from both paths → map to parent chunks → include cluster summaries as additional context
- **Trace**: Records which branches explored, how many leaves from each path

**Metadata**: `topdown_leaves`, `bottomup_leaves`, `merged_parents`, `cluster_summaries_used`, `branches_explored`, `hierarchy_levels`

### 4b. `MACERRetriever` (Multi-Agent Iterative)

**Constructor**: stores `embedding_model` and `cross_encoder`.

**`retrieve(query, alpha, use_reranker, hops, max_iterations)`**:
- Composes `VectorRAGRetriever` + `GraphRAGRetriever` internally
- Iterative loop (1 to `max_iterations`):
  1. **Retriever agent**: vector + graph search with `current_query` → collect new chunks (deduplicated)
  2. **Constructor agent**: LLM call to extract key facts as numbered list from new chunks
  3. **Reflector agent**: LLM call with accumulated facts → outputs `SUFFICIENT: yes/no`, `GAPS: ...`, `REFINED_QUERY: ...`
  4. If sufficient or no new info found → break early; else update `current_query`
- Build final context: "Key Facts" section + "Supporting Documents" section
- **Trace**: Detailed per-iteration log of all 3 agents' actions

**Metadata**: `iterations_completed`, `max_iterations`, `termination_reason`, `total_facts`, `total_llm_calls`, `facts` list

### 4c. Update `get_retriever()` factory

Add `"FABLE RAG"` → `FABLERetriever` and `"MACER RAG"` → `MACERRetriever`.

### 4d. Update module docstring to "Seven retrieval strategies"

---

## Step 5: `app.py` — UI Integration

### 5a. Dropdown — Add 2 new options
```python
options=["No RAG", "Vector RAG", "Graph RAG", "Vector + Graph RAG",
         "Agentic RAG", "FABLE RAG", "MACER RAG"]
```

### 5b. Sidebar controls

- **Reranker + Alpha**: shown for `("Vector RAG", "Vector + Graph RAG", "FABLE RAG", "MACER RAG")`
- **Graph Hops**: shown for `("Graph RAG", "Vector + Graph RAG", "MACER RAG")`
- **FABLE-specific**: "Top-Down Branches" slider (1–5, default 3) + info box
- **MACER-specific**: "Max Iterations" slider (1–5, default 3) + info box

### 5c. `needs_data` check — add both new modes

### 5d. Chat response block
- FABLE RAG: wrap in `st.status("FABLE: navigating hierarchy...")` with trace display
- MACER RAG: wrap in `st.status("MACER: iterating...")` with trace display, show iteration count in label
- FABLE-specific check: if hierarchy file doesn't exist, show re-ingest warning

### 5e. `_display_metrics()` — add new metadata sections
- **FABLE**: top-down/bottom-up leaf counts, merged parents, cluster summaries used, branches explored
- **MACER**: iterations completed, termination reason, facts extracted, LLM calls count, expandable facts list

### 5f. Trace expander — rename from "Agent Trace" to context-aware label ("Hierarchy Navigation Trace" for FABLE, "Iteration Trace" for MACER)

### 5g. Ingestion success message — include FABLE cluster count

---

## Implementation Order

1. `config.py` + `requirements.txt` (constants + dependency)
2. `ingestion.py` (FABLE hierarchy building)
3. `retrievers.py` (FABLERetriever + MACERRetriever + factory)
4. `app.py` (UI wiring)
5. Install scipy, re-ingest, test all 7 modes

---

## Verification

1. `pip install scipy` in the venv
2. Re-ingest test documents → verify FABLE hierarchy is built (check `graph_store/fable_hierarchy.json`)
3. Test **FABLE RAG**: query should show hierarchy trace (branches explored, top-down + bottom-up merge)
4. Test **MACER RAG**: query should show iteration trace (retriever → constructor → reflector loop)
5. Test MACER with `max_iterations=1` → single pass, no refinement
6. Test all existing 5 modes still work correctly
7. Verify sidebar controls appear/disappear correctly for each mode
