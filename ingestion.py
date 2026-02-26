"""
Data ingestion pipeline: document loading, parent-child chunking,
numpy vector storage, BM25 index building, and Knowledge Graph extraction.
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import tiktoken
from openai import OpenAI
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


# ── Numpy Vector Store ───────────────────────────────────────────

def store_chunks(
    children: list[ChildChunk],
    parent_dict: dict[str, str],
    embedding_model: SentenceTransformer,
) -> None:
    """Embed child chunks and store as numpy arrays on disk."""
    if not children:
        return

    # Embed all child texts
    texts = [c.text for c in children]
    new_embeddings = embedding_model.encode(texts, show_progress_bar=False)

    # Load existing vectors if present
    _ensure_dir(config.VECTOR_STORE_PATH)
    if os.path.exists(config.VECTOR_STORE_PATH):
        data = np.load(config.VECTOR_STORE_PATH, allow_pickle=False)
        existing_embeddings = data["embeddings"]
        new_embeddings = np.vstack([existing_embeddings, new_embeddings])

    # Save embeddings
    np.savez_compressed(config.VECTOR_STORE_PATH, embeddings=new_embeddings)

    # Persist parent chunks to JSON
    _ensure_dir(config.PARENT_STORE_PATH)
    existing_parents = _load_json(config.PARENT_STORE_PATH, default={})
    existing_parents.update(parent_dict)
    _save_json(config.PARENT_STORE_PATH, existing_parents)

    # Persist child metadata for BM25 and retrieval
    existing_children = _load_json(config.CHILD_TEXTS_PATH, default=[])
    existing_children.extend([
        {
            "child_id": c.child_id,
            "text": c.text,
            "parent_id": c.parent_id,
            "source_file": c.source_file,
        }
        for c in children
    ])
    _save_json(config.CHILD_TEXTS_PATH, existing_children)


def vector_search(
    query_embedding: np.ndarray, top_k: int = 10
) -> list[tuple[int, float]]:
    """
    Cosine similarity search against stored vectors.
    Returns list of (index, similarity_score) tuples, highest first.
    """
    if not os.path.exists(config.VECTOR_STORE_PATH):
        return []

    data = np.load(config.VECTOR_STORE_PATH, allow_pickle=False)
    embeddings = data["embeddings"]

    if embeddings.shape[0] == 0:
        return []

    # Cosine similarity: dot product of normalized vectors
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normalized = embeddings / norms
    similarities = normalized @ query_norm

    # Get top-k indices
    k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[::-1][:k]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]


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

    # Phase 2: Embed and store vectors
    if progress_callback:
        progress_callback("Embedding and storing chunks...")
    store_chunks(all_children, all_parents, embedding_model)

    # Phase 3: Build Knowledge Graph
    if progress_callback:
        progress_callback("Building knowledge graph...")
    G = build_knowledge_graph(all_parents, llm_client, progress_callback)

    return {
        "num_parents": len(all_parents),
        "num_children": len(all_children),
        "num_triplets": G.number_of_edges(),
        "num_nodes": G.number_of_nodes(),
    }


def clear_all_data() -> None:
    """Remove all ingested data (vectors, graph, parent store)."""
    import shutil
    for dir_path in ("vector_store", "graph_store"):
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
