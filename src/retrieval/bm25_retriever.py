"""BM25 keyword retriever — no embedding API required, works fully offline.

Two retrieval modes:
  retrieve()              — Baseline: independent chunk scoring, take top-k
  retrieve_lexrag_style() — LexRAG:   page-level grouping → neighbor expansion
                            → context-boosted re-ranking (mirrors the real
                            LexRAG pipeline without requiring any API calls)
"""
import math
import re
from collections import Counter, defaultdict

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

    def _bm25_score(self, qtokens: list[str], doc_tokens: list[str]) -> float:
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for t in qtokens:
            if t not in tf:
                continue
            idf = self._idf.get(t, 0.0)
            freq = tf[t]
            score += idf * (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            )
        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Baseline: score every chunk independently, return top-k."""
        qtokens = self._tokenize(query)
        scored = [
            (chunk, self._bm25_score(qtokens, doc))
            for chunk, doc in zip(self.chunks, self._corpus)
        ]
        scored = [(c, s) for c, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_lexrag_style(
        self, query: str, top_k: int = 5, neighbor_pages: int = 1
    ) -> list[tuple[Chunk, float]]:
        """LexRAG-style retrieval (offline BM25 variant):

        1. Score every chunk with BM25.
        2. Aggregate scores per (doc_id, page_number) to find the top pages.
        3. Expand each top page by ±neighbor_pages.
        4. Re-rank all chunks from those pages using:
               final_score = chunk_score + 0.4 * page_context_score
           The page_context_score is the max BM25 score of any chunk on that
           page — this rewards chunks that sit on highly-relevant pages, even
           if the individual chunk text doesn't perfectly match the query.
        5. Return top-k from the re-ranked set.

        This mirrors the real LexRAG pipeline (page-level search → neighbor
        expansion → sub-chunk retrieval → re-ranking) without any API calls.
        """
        qtokens = self._tokenize(query)

        # Step 1: score every chunk
        chunk_scores: list[float] = [
            self._bm25_score(qtokens, doc) for doc in self._corpus
        ]

        # Step 2: aggregate to page level (best chunk score per page)
        page_best: dict[tuple, float] = defaultdict(float)
        for chunk, score in zip(self.chunks, chunk_scores):
            key = (chunk.doc_id, chunk.page_number)
            page_best[key] = max(page_best[key], score)

        # Step 3: pick top pages and expand to neighbors
        top_pages = sorted(page_best.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
        expanded_keys: set[tuple] = set()
        for (doc_id, page_num), _ in top_pages:
            for offset in range(-neighbor_pages, neighbor_pages + 1):
                expanded_keys.add((doc_id, page_num + offset))

        # Step 4: re-rank chunks that fall in the expanded page set
        candidates: list[tuple[Chunk, float]] = []
        for chunk, chunk_score in zip(self.chunks, chunk_scores):
            key = (chunk.doc_id, chunk.page_number)
            if key not in expanded_keys:
                continue
            page_ctx = page_best.get(key, 0.0)
            boosted = chunk_score + 0.4 * page_ctx
            if boosted > 0:
                candidates.append((chunk, boosted))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
