import logging
import numpy as np

from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore
from src.data.chunking import Chunk
from src.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class BaselineRetriever:
    def __init__(self, cfg: dict, embedding_provider: EmbeddingProvider):
        self.cfg = cfg
        self.top_k = cfg["retrieval"]["top_k"]
        self.embedder = Embedder(embedding_provider, cfg["embedding"]["cache_dir"])
        self.store = FaissStore(cfg)
        self.store.load("baseline")
        logger.info("BaselineRetriever ready")

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        query_vector = self.embedder.embed_chunks([
            Chunk(chunk_id="q", doc_id="q", text=query,
                  page_number=-1, chunk_index=0, char_count=len(query))
        ])[0]
        results = self.store.search(query_vector, self.top_k)
        logger.info("Baseline retrieved %d chunks", len(results))
        return results
