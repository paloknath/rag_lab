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

    return {
        "num_parents": len(all_parents),
        "num_children": len(all_children),
        "num_triplets": G.number_of_edges(),
        "num_nodes": G.number_of_nodes(),
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
