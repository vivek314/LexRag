# indexing.py — Build baseline and LexRAG FAISS indices.

import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from src.data.processor import Document, PageContent
from src.data.chunking import get_chunker
from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore
from src.providers.base import EmbeddingProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# D1 — provenance tags stamped onto every chunk so retrieval/answer-selection can
# prefer authoritative + current sources on points of law (see golden-set handoff §2).
_PROVENANCE_KEYS = (
    "source_type", "authority", "currency",
    "publication_date", "in_force_date", "as_amended_by", "repeals",
)


def _stamp_provenance(doc: Document, chunks: list) -> list:
    """Merge the document's D1 provenance tags into each chunk's metadata."""
    tags = {k: doc.metadata.get(k) for k in _PROVENANCE_KEYS if doc.metadata.get(k) is not None}
    if not tags:
        return chunks
    for c in chunks:
        c.metadata.update(tags)
    return chunks


def load_documents(processed_dir: str) -> list[Document]:
    docs = []
    for json_file in Path(processed_dir).glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            raw = json.load(f)
        pages = [PageContent(**p) for p in raw["pages"]]
        raw["pages"] = pages
        docs.append(Document(**raw))
    logger.info("Loaded %d documents", len(docs))
    return docs


def build_indices(cfg: dict, embedding_provider: Optional[EmbeddingProvider] = None) -> None:
    if embedding_provider is None:
        from src.providers.factory import get_providers
        _, embedding_provider = get_providers()

    cache_dir = cfg["embedding"].get("cache_dir", "data/processed/embedding_cache")
    embedder = Embedder(embedding_provider, cache_dir)

    docs = load_documents(cfg["data"]["processed_dir"])

    # --- Baseline: naive chunks → single index ---
    naive_chunker = get_chunker("naive", cfg["chunking"]["chunk_size"], cfg["chunking"]["overlap"])
    naive_chunks = []
    for doc in docs:
        naive_chunks.extend(_stamp_provenance(doc, naive_chunker.chunk(doc)))
    logger.info("Naive chunks: %d", len(naive_chunks))

    naive_vectors = embedder.embed_chunks(naive_chunks)
    baseline_store = FaissStore(cfg)
    baseline_store.add(naive_chunks, naive_vectors)
    baseline_store.save("baseline")

    # --- LexRAG: hierarchical chunks → two indices ---
    hier_chunker = get_chunker("hierarchical", cfg["chunking"]["chunk_size"], cfg["chunking"]["overlap"])
    hier_chunks = []
    for doc in docs:
        hier_chunks.extend(_stamp_provenance(doc, hier_chunker.chunk(doc)))

    page_chunks = [c for c in hier_chunks if c.metadata.get("level") == "page"]
    sub_chunks = [c for c in hier_chunks if c.metadata.get("level") == "chunk"]

    page_store = FaissStore(cfg)
    page_store.add(page_chunks, embedder.embed_chunks(page_chunks))
    page_store.save("lexrag_pages")

    sub_store = FaissStore(cfg)
    sub_store.add(sub_chunks, embedder.embed_chunks(sub_chunks))
    sub_store.save("lexrag_chunks")

    # --- Statute section index → cross-reference DFS targets (lexrag_sections) ---
    # Only statute docs; uses the 2025-Act-aware chunker. These section-level chunks are
    # the resolution targets when a retrieved chunk cites "section N".
    statute_chunker = get_chunker("statute_2025", cfg["chunking"]["chunk_size"], cfg["chunking"]["overlap"])
    section_chunks = []
    for doc in docs:
        if doc.metadata.get("authority") != "statute":
            continue
        secs = [c for c in statute_chunker.chunk(doc) if c.metadata.get("level") == "section"]
        section_chunks.extend(_stamp_provenance(doc, secs))
    if section_chunks:
        logger.info("Statute section chunks: %d", len(section_chunks))
        sec_store = FaissStore(cfg)
        sec_store.add(section_chunks, embedder.embed_chunks(section_chunks))
        sec_store.save("lexrag_sections")

    logger.info("All indices built and saved.")


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (python-dotenv is not a hard dependency here)."""
    p = Path(path)
    if not p.exists():
        return
    import os
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


if __name__ == "__main__":
    _load_dotenv()
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    build_indices(cfg)
