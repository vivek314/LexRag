import logging
from src.data.chunking import Chunk

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    CrossEncoder = None  # type: ignore
    _HAS_CROSS_ENCODER = False


class Reranker:
    def __init__(self, cfg: dict):
        self.top_k = cfg["retrieval"]["rerank_top_k"]
        if _HAS_CROSS_ENCODER:
            model_name = cfg["retrieval"]["reranker"]["model"]
            logger.info("Loading cross-encoder: %s", model_name)
            self.model = CrossEncoder(model_name, max_length=512)
            logger.info("Reranker ready")
        else:
            self.model = None
            logger.warning("sentence-transformers not installed — reranker disabled, using score passthrough")

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]],
               top_n: int | None = None) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        n = top_n or self.top_k

        if not self.model:
            # Degrade gracefully: sort by existing (RRF/fused) score and truncate
            return sorted(candidates, key=lambda x: x[1], reverse=True)[:n]

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.model.predict(pairs)
        scored = sorted(
            zip([chunk for chunk, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        results = [(chunk, float(score)) for chunk, score in scored[:n]]
        logger.info("Reranked %d → %d chunks", len(candidates), len(results))
        return results
