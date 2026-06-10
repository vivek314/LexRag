import hashlib
import logging
from pathlib import Path

import numpy as np

from src.data.chunking import Chunk
from src.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, provider: EmbeddingProvider, cache_dir: str = "data/processed/embedding_cache"):
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        # Provider type is part of cache key so OpenAI and fastembed vectors don't collide.
        prefix = type(self.provider).__name__
        key = hashlib.md5(f"{prefix}:{text}".encode()).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def _load(self, text: str) -> np.ndarray | None:
        p = self._cache_path(text)
        return np.load(str(p)) if p.exists() else None

    def _save(self, text: str, vec: np.ndarray) -> None:
        np.save(str(self._cache_path(text)), vec)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        vectors: list[np.ndarray | None] = []
        miss_texts: list[str] = []
        miss_idx: list[int] = []

        for i, chunk in enumerate(chunks):
            cached = self._load(chunk.text)
            if cached is not None:
                vectors.append(cached)
            else:
                vectors.append(None)
                miss_texts.append(chunk.text)
                miss_idx.append(i)

        if miss_texts:
            new_vecs = self.provider.embed(miss_texts)
            for j, idx in enumerate(miss_idx):
                vectors[idx] = new_vecs[j]
                self._save(miss_texts[j], new_vecs[j])
            logger.info("Embedded %d new / %d from cache", len(miss_texts), len(chunks) - len(miss_texts))

        return np.vstack(vectors)
