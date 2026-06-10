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
        naive_chunks.extend(naive_chunker.chunk(doc))
    logger.info("Naive chunks: %d", len(naive_chunks))

    naive_vectors = embedder.embed_chunks(naive_chunks)
    baseline_store = FaissStore(cfg)
    baseline_store.add(naive_chunks, naive_vectors)
    baseline_store.save("baseline")

    # --- LexRAG: hierarchical chunks → two indices ---
    hier_chunker = get_chunker("hierarchical", cfg["chunking"]["chunk_size"], cfg["chunking"]["overlap"])
    hier_chunks = []
    for doc in docs:
        hier_chunks.extend(hier_chunker.chunk(doc))

    page_chunks = [c for c in hier_chunks if c.metadata.get("level") == "page"]
    sub_chunks = [c for c in hier_chunks if c.metadata.get("level") == "chunk"]

    page_store = FaissStore(cfg)
    page_store.add(page_chunks, embedder.embed_chunks(page_chunks))
    page_store.save("lexrag_pages")

    sub_store = FaissStore(cfg)
    sub_store.add(sub_chunks, embedder.embed_chunks(sub_chunks))
    sub_store.save("lexrag_chunks")

    logger.info("All indices built and saved.")


if __name__ == "__main__":
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    build_indices(cfg)
