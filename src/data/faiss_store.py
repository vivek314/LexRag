import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.data.chunking import Chunk

logger = logging.getLogger(__name__)


class FaissStore:
    def __init__(self, cfg: dict):
        self.index_type = cfg["faiss"]["index_type"]
        self.normalize = cfg["faiss"]["normalize_vectors"]
        self.index_dir = Path(cfg["data"]["indices_dir"])
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index = None          # FAISS index object
        self.chunks: list[Chunk] = []   # parallel list — chunks[i] matches index row i

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if self.normalize:
            faiss.normalize_L2(vectors)   # modifies in-place, makes all lengths = 1

        dim = vectors.shape[1]           # 1536

        if self.index is None:
            if self.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(dim)
            elif self.index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(dim)
            else:
                self.index = faiss.index_factory(dim, self.index_type)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info(f"Index now has {self.index.ntotal} vectors")
    
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # query_vector shape must be (1, 1536) — FAISS expects 2D
        q = query_vector.reshape(1, -1).astype(np.float32)

        if self.normalize:
            faiss.normalize_L2(q)

        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:       # FAISS returns -1 when fewer results than top_k exist
                continue
            results.append((self.chunks[idx], float(score)))

        return results  # list of (Chunk, similarity_score)
    
    def save(self, name: str) -> None:
        faiss.write_index(self.index, str(self.index_dir / f"{name}.faiss"))
        with open(self.index_dir / f"{name}_chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved index: {name}")

    def load(self, name: str) -> None:
        self.index = faiss.read_index(str(self.index_dir / f"{name}.faiss"))
        with open(self.index_dir / f"{name}_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Loaded index: {name} ({self.index.ntotal} vectors)")