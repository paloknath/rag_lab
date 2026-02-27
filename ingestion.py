"""
Data ingestion pipeline: document loading, parent-child chunking,
ChromaDB storage, BM25 index building, and Knowledge Graph extraction.
"""

import json
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chromadb
import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import tiktoken
from openai import OpenAI
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer

import config


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class ChildChunk:
    """A small chunk indexed for vector search, linked to its parent."""
    text: str
    parent_id: str
    source_file: str
    chunk_index: int
    child_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ── Document Loading ─────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_text(path: str) -> str:
    """Read a plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(path: str) -> str:
    """Load a document based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".txt":
        return load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Token-Based Chunking ────────────────────────────────────────

class TokenChunker:
    """Splits text into chunks based on token count using tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(encoding_name)

    def _split_tokens(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks of `chunk_size` tokens."""
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
            start += chunk_size - overlap
        return chunks

    def chunk_into_parents(self, text: str) -> list[str]:
        """Split text into parent-sized chunks."""
        return self._split_tokens(text, config.PARENT_CHUNK_SIZE, config.CHUNK_OVERLAP)

    def chunk_into_children(self, text: str) -> list[str]:
        """Split text into child-sized chunks."""
        return self._split_tokens(text, config.CHILD_CHUNK_SIZE, config.CHUNK_OVERLAP)


def chunk_document(
    text: str, source_file: str, chunker: TokenChunker
) -> tuple[dict[str, str], list[ChildChunk]]:
    """
    Split document into parent and child chunks.

    Returns:
        parent_dict: {parent_id: parent_text}
        children: list of ChildChunk objects linked to their parents
    """
    parent_dict: dict[str, str] = {}
    children: list[ChildChunk] = []
    child_index = 0

    parent_texts = chunker.chunk_into_parents(text)
    for parent_text in parent_texts:
        parent_id = str(uuid.uuid4())
        parent_dict[parent_id] = parent_text

        child_texts = chunker.chunk_into_children(parent_text)
        for child_text in child_texts:
            children.append(ChildChunk(
                text=child_text,
                parent_id=parent_id,
                source_file=source_file,
                chunk_index=child_index,
            ))
            child_index += 1

    return parent_dict, children


# ── ChromaDB Storage ─────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the persistent ChromaDB client."""
    return chromadb.PersistentClient(path=config.CHROMA_DB_PATH)


def store_chunks(
    children: list[ChildChunk],
    parent_dict: dict[str, str],
    embedding_model: SentenceTransformer,
) -> None:
    """Embed child chunks and store in ChromaDB. Save parent dict and child texts."""
    if not children:
        return

    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed all child texts
    texts = [c.text for c in children]
    embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()

    # Upsert into ChromaDB
    collection.upsert(
        ids=[c.child_id for c in children],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "parent_id": c.parent_id,
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
            }
            for c in children
        ],
    )

    # Persist parent chunks to JSON
    _ensure_dir(config.PARENT_STORE_PATH)
    existing_parents = _load_json(config.PARENT_STORE_PATH, default={})
    existing_parents.update(parent_dict)
    _save_json(config.PARENT_STORE_PATH, existing_parents)

    # Persist child texts for BM25 index rebuilding
    existing_children = _load_json(config.CHILD_TEXTS_PATH, default=[])
    existing_children.extend([
        {"child_id": c.child_id, "text": c.text, "parent_id": c.parent_id,
         "source_file": c.source_file}
        for c in children
    ])
    _save_json(config.CHILD_TEXTS_PATH, existing_children)


# ── Knowledge Graph Extraction ───────────────────────────────────

TRIPLET_EXTRACTION_PROMPT = """Extract knowledge graph triplets from the following text.
Return ONLY a JSON array of objects with keys "subject", "predicate", "object".
Each triplet should represent a factual relationship found in the text.
Normalize entity names to lowercase.

Example output:
[{{"subject": "albert einstein", "predicate": "born in", "object": "ulm"}},
 {{"subject": "albert einstein", "predicate": "developed", "object": "theory of relativity"}}]

Text:
{text}

JSON array:"""


def extract_triplets(text: str, llm_client: OpenAI) -> list[tuple[str, str, str]]:
    """Use the LLM to extract (Subject, Predicate, Object) triplets from text."""
    try:
        response = llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a knowledge extraction assistant. Return only valid JSON."},
                {"role": "user", "content": TRIPLET_EXTRACTION_PROMPT.format(text=text[:2000])},
            ],
            temperature=0.0,
            max_tokens=1024,
            seed=random.randint(0, 2**31),
        )
        content = response.choices[0].message.content.strip()
        # Try to extract JSON from the response
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        triplets_raw = json.loads(content)
        return [
            (t["subject"].lower().strip(), t["predicate"].lower().strip(), t["object"].lower().strip())
            for t in triplets_raw
            if all(k in t for k in ("subject", "predicate", "object"))
        ]
    except (json.JSONDecodeError, KeyError, IndexError):
        return []


def build_knowledge_graph(
    parent_dict: dict[str, str],
    llm_client: OpenAI,
    progress_callback: Callable[[str], None] | None = None,
) -> nx.DiGraph:
    """Extract triplets from all parent chunks and build a NetworkX graph."""
    G = load_graph()

    total = len(parent_dict)
    for i, (parent_id, text) in enumerate(parent_dict.items()):
        if progress_callback:
            progress_callback(f"Extracting triplets from chunk {i + 1}/{total}...")

        triplets = extract_triplets(text, llm_client)
        for subj, pred, obj in triplets:
            G.add_edge(subj, obj, relation=pred, source_chunk=parent_id)
            for node in (subj, obj):
                if "chunk_ids" not in G.nodes[node]:
                    G.nodes[node]["chunk_ids"] = []
                if parent_id not in G.nodes[node]["chunk_ids"]:
                    G.nodes[node]["chunk_ids"].append(parent_id)

    save_graph(G)
    return G


def save_graph(G: nx.DiGraph) -> None:
    """Serialize NetworkX graph to JSON."""
    _ensure_dir(config.GRAPH_STORE_PATH)
    data = nx.node_link_data(G)
    _save_json(config.GRAPH_STORE_PATH, data)


def load_graph() -> nx.DiGraph:
    """Load NetworkX graph from JSON, or return empty graph."""
    if os.path.exists(config.GRAPH_STORE_PATH):
        data = _load_json(config.GRAPH_STORE_PATH)
        return nx.node_link_graph(data, directed=True)
    return nx.DiGraph()


# ── FABLE Hierarchy Building ─────────────────────────────────────

CLUSTER_SUMMARY_PROMPT = """Summarize the following group of text passages into a single concise paragraph
that captures the main topics, entities, and relationships discussed.

Passages:
{texts}

Concise summary:"""


def _llm_summarize(text: str, llm_client: OpenAI) -> str:
    """Generate a concise summary of text using the LLM."""
    response = llm_client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a concise summarizer. Produce a single paragraph summary."},
            {"role": "user", "content": CLUSTER_SUMMARY_PROMPT.format(texts=text)},
        ],
        temperature=0.0,
        max_tokens=config.FABLE_SUMMARY_MAX_TOKENS,
        seed=random.randint(0, 2**31),
    )
    return response.choices[0].message.content.strip()


def build_fable_hierarchy(
    children: list[ChildChunk],
    embedding_model: SentenceTransformer,
    llm_client: OpenAI,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Build a hierarchical clustering tree over child chunks for FABLE retrieval.

    Steps:
    1. Encode all child chunks with the embedding model
    2. Agglomerative clustering at level 0 (leaf clusters)
    3. LLM-summarize each cluster
    4. If NUM_LEVELS >= 2, cluster the summaries into super-clusters
    5. Persist hierarchy to JSON

    Returns the hierarchy dict.
    """
    if len(children) < 2:
        hierarchy = {
            "levels": 1,
            "root": {
                "summary": children[0].text if children else "",
                "summary_embedding": [],
                "children": [],
            },
            "clusters": {},
            "child_to_cluster": {},
        }
        _ensure_dir(config.FABLE_HIERARCHY_PATH)
        _save_json(config.FABLE_HIERARCHY_PATH, hierarchy)
        return hierarchy

    if progress_callback:
        progress_callback("Building FABLE hierarchy: computing embeddings...")

    # Step 1: Get embeddings for all children
    child_texts = [c.text for c in children]
    child_ids = [c.child_id for c in children]
    embeddings = embedding_model.encode(child_texts, show_progress_bar=False)

    # Step 2: Level-0 clustering
    n_clusters_l0 = min(
        config.FABLE_MAX_CLUSTERS,
        max(2, len(children) // config.FABLE_MIN_CLUSTER_SIZE),
    )
    # Clamp to actual number of children
    n_clusters_l0 = min(n_clusters_l0, len(children))

    dist_matrix = pdist(embeddings, metric="cosine")
    # Replace NaN distances (identical vectors) with 0
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)
    Z = linkage(dist_matrix, method="average")
    labels_l0 = fcluster(Z, t=n_clusters_l0, criterion="maxclust")

    if progress_callback:
        progress_callback(f"Building FABLE hierarchy: created {n_clusters_l0} level-0 clusters...")

    # Group children by cluster label
    clusters_l0: dict[int, list[int]] = {}
    for idx, label in enumerate(labels_l0):
        clusters_l0.setdefault(int(label), []).append(idx)

    # Step 3: Summarize each level-0 cluster
    cluster_data: dict[str, dict] = {}
    child_to_cluster: dict[str, str] = {}
    l0_cluster_ids: list[str] = []
    l0_summaries: list[str] = []
    l0_embeddings: list = []

    for label, indices in sorted(clusters_l0.items()):
        cluster_id = f"cluster_0_{label}"
        l0_cluster_ids.append(cluster_id)

        cluster_child_ids = [child_ids[i] for i in indices]
        cluster_texts = [child_texts[i] for i in indices]

        for cid in cluster_child_ids:
            child_to_cluster[cid] = cluster_id

        if progress_callback:
            progress_callback(f"Summarizing cluster {cluster_id} ({len(indices)} chunks)...")

        combined = "\n\n---\n\n".join(cluster_texts)[:3000]
        summary = _llm_summarize(combined, llm_client)
        summary_emb = embedding_model.encode(summary, show_progress_bar=False).tolist()

        l0_summaries.append(summary)
        l0_embeddings.append(summary_emb)

        cluster_data[cluster_id] = {
            "level": 0,
            "summary": summary,
            "summary_embedding": summary_emb,
            "children": [],
            "leaf_child_ids": cluster_child_ids,
        }

    # Step 4: Level-1 clustering (if configured and enough clusters)
    root_children = l0_cluster_ids

    if config.FABLE_NUM_LEVELS >= 2 and len(l0_cluster_ids) >= 3:
        if progress_callback:
            progress_callback("Building FABLE hierarchy: level-1 super-clusters...")

        l0_emb_array = np.array(l0_embeddings)
        n_clusters_l1 = min(config.FABLE_MAX_CLUSTERS, max(2, len(l0_cluster_ids) // 2))

        dist_l1 = pdist(l0_emb_array, metric="cosine")
        dist_l1 = np.nan_to_num(dist_l1, nan=0.0)
        Z_l1 = linkage(dist_l1, method="average")
        labels_l1 = fcluster(Z_l1, t=n_clusters_l1, criterion="maxclust")

        clusters_l1: dict[int, list[int]] = {}
        for idx, label in enumerate(labels_l1):
            clusters_l1.setdefault(int(label), []).append(idx)

        l1_cluster_ids: list[str] = []
        for label, indices in sorted(clusters_l1.items()):
            cluster_id = f"cluster_1_{label}"
            l1_cluster_ids.append(cluster_id)

            sub_cluster_ids = [l0_cluster_ids[i] for i in indices]
            sub_summaries = [l0_summaries[i] for i in indices]

            combined = "\n\n---\n\n".join(sub_summaries)[:3000]
            summary = _llm_summarize(combined, llm_client)
            summary_emb = embedding_model.encode(summary, show_progress_bar=False).tolist()

            all_leaf_ids = []
            for scid in sub_cluster_ids:
                all_leaf_ids.extend(cluster_data[scid]["leaf_child_ids"])

            cluster_data[cluster_id] = {
                "level": 1,
                "summary": summary,
                "summary_embedding": summary_emb,
                "children": sub_cluster_ids,
                "leaf_child_ids": all_leaf_ids,
            }

        root_children = l1_cluster_ids

    # Step 5: Build root node
    root_summary_parts = [cluster_data[cid]["summary"] for cid in root_children]
    root_summary = _llm_summarize(
        "\n\n---\n\n".join(root_summary_parts)[:3000], llm_client
    )

    hierarchy = {
        "levels": config.FABLE_NUM_LEVELS,
        "root": {
            "summary": root_summary,
            "summary_embedding": embedding_model.encode(root_summary, show_progress_bar=False).tolist(),
            "children": root_children,
        },
        "clusters": cluster_data,
        "child_to_cluster": child_to_cluster,
    }

    _ensure_dir(config.FABLE_HIERARCHY_PATH)
    _save_json(config.FABLE_HIERARCHY_PATH, hierarchy)

    if progress_callback:
        progress_callback(f"FABLE hierarchy built: {len(cluster_data)} clusters across {config.FABLE_NUM_LEVELS} levels")

    return hierarchy


def load_fable_hierarchy() -> dict | None:
    """Load FABLE hierarchy from JSON, or return None if not built."""
    if os.path.exists(config.FABLE_HIERARCHY_PATH):
        return _load_json(config.FABLE_HIERARCHY_PATH)
    return None


# ── Top-Level Ingestion Orchestration ────────────────────────────

def ingest_documents(
    file_paths: list[str],
    embedding_model: SentenceTransformer,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Full ingestion pipeline: Load → Chunk → Embed → Store → Build KG.

    Returns stats dict with counts of parents, children, triplets, graph nodes.
    """
    llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    chunker = TokenChunker()

    all_parents: dict[str, str] = {}
    all_children: list[ChildChunk] = []

    # Phase 1: Load and chunk documents
    for path in file_paths:
        if progress_callback:
            progress_callback(f"Loading {Path(path).name}...")
        text = load_document(path)
        parent_dict, children = chunk_document(text, Path(path).name, chunker)
        all_parents.update(parent_dict)
        all_children.extend(children)

    # Phase 2: Embed and store in ChromaDB
    if progress_callback:
        progress_callback("Embedding and storing chunks...")
    store_chunks(all_children, all_parents, embedding_model)

    # Phase 3: Build Knowledge Graph
    if progress_callback:
        progress_callback("Building knowledge graph...")
    G = build_knowledge_graph(all_parents, llm_client, progress_callback)

    # Phase 4: Build FABLE hierarchy
    if progress_callback:
        progress_callback("Building FABLE hierarchy...")
    fable_hierarchy = build_fable_hierarchy(
        all_children, embedding_model, llm_client, progress_callback,
    )

    return {
        "num_parents": len(all_parents),
        "num_children": len(all_children),
        "num_triplets": G.number_of_edges(),
        "num_nodes": G.number_of_nodes(),
        "fable_clusters": len(fable_hierarchy.get("clusters", {})),
    }


def clear_all_data() -> None:
    """Remove all ingested data (ChromaDB, graph, parent store)."""
    import shutil
    for dir_path in (config.CHROMA_DB_PATH, "graph_store"):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


# ── Helpers ──────────────────────────────────────────────────────

def _ensure_dir(file_path: str) -> None:
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def _save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: str, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
