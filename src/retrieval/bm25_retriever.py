"""BM25 keyword retriever — no embedding API required, works fully offline."""
import math
import re
from collections import Counter

from src.data.chunking import Chunk


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._corpus = [self._tokenize(c.text) for c in chunks]
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_index(self) -> None:
        N = len(self._corpus)
        self._avgdl = sum(len(d) for d in self._corpus) / max(N, 1)
        df: Counter = Counter()
        for doc in self._corpus:
            for term in set(doc):
                df[term] += 1
        self._idf = {
            t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        qtokens = self._tokenize(query)
        scores: list[tuple[Chunk, float]] = []
        for chunk, doc in zip(self.chunks, self._corpus):
            tf = Counter(doc)
            dl = len(doc)
            score = 0.0
            for t in qtokens:
                if t not in tf:
                    continue
                idf = self._idf.get(t, 0.0)
                freq = tf[t]
                score += idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
            if score > 0:
                scores.append((chunk, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
