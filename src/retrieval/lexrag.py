import logging
import numpy as np

from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore
from src.data.chunking import Chunk
from src.retrieval.reranker import Reranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.data.chunking import StatuteChunker
from src.providers.base import LLMProvider, EmbeddingProvider

logger = logging.getLogger(__name__)


class LexRAGRetriever:
    def __init__(self, cfg: dict, llm: LLMProvider, embedding_provider: EmbeddingProvider):
        self.cfg = cfg
        self.top_k = cfg["retrieval"]["top_k"]
        self.neighbor_pages = cfg["retrieval"]["neighbor_pages"]

        self.embedder = Embedder(embedding_provider, cfg["embedding"]["cache_dir"])
        self.reranker = Reranker(cfg)
        self.llm = llm

        # Two separate indices — page level + sub-chunk level
        self.page_store = FaissStore(cfg)
        self.page_store.load("lexrag_pages")

        self.chunk_store = FaissStore(cfg)
        self.chunk_store.load("lexrag_chunks")

        # Section-level statute index → cross-reference DFS targets. Built by
        # indexing.build_indices from the 2025-Act-aware chunker; optional.
        self._section_lookup: dict[str, Chunk] = {}
        try:
            self.section_store = FaissStore(cfg)
            self.section_store.load("lexrag_sections")
            self._section_lookup = {
                c.section_id: c for c in self.section_store.chunks if c.section_id
            }
        except (FileNotFoundError, RuntimeError):
            self.section_store = None
        # Reused only for its text→section-reference extractor.
        self._ref_extractor = StatuteChunker()
        self.ref_depth = cfg["retrieval"].get("ref_resolve_depth", 2)
        self.currency_aware = cfg["retrieval"].get("currency_aware", False)

        # Hybrid dense+BM25 retrieval (RRF). BM25 indexes the same sub-chunks as the
        # dense chunk_store, so the two ranked lists fuse over a shared id space.
        self.hybrid = cfg["retrieval"].get("hybrid_enabled", False)
        self.rrf_k = cfg["retrieval"].get("rrf_k", 60)
        # Explainer-aware fusion: rerank a wider pool, then down-weight stale-explainer
        # chunks relative to statute before the final top-k cut.
        self.rerank_pool_n = cfg["retrieval"].get("rerank_pool_n", 15)
        self.explainer_weight = cfg["retrieval"].get("explainer_weight", 0.5)
        self.bm25 = BM25Retriever(list(self.chunk_store.chunks)) if self.hybrid else None
        logger.info(
            "LexRAGRetriever ready — section lookup: %d sections | hybrid=%s",
            len(self._section_lookup), self.hybrid,
        )

    def _rrf_fuse(
        self, ranked_lists: list[list[tuple[Chunk, float]]],
    ) -> list[tuple[Chunk, float]]:
        """Reciprocal Rank Fusion: merge ranked lists by 1/(k+rank), summed across lists.

        Robust to score-scale differences between dense (cosine) and BM25 (tf-idf) —
        only the RANK within each list matters. Returns chunks sorted by fused score.
        """
        fused: dict[str, list] = {}
        for ranked in ranked_lists:
            for rank, (chunk, _) in enumerate(ranked):
                contrib = 1.0 / (self.rrf_k + rank + 1)
                entry = fused.get(chunk.chunk_id)
                if entry is None:
                    fused[chunk.chunk_id] = [chunk, contrib]
                else:
                    entry[1] += contrib
        merged = [(c, s) for c, s in fused.values()]
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    # Old (1961-Act) → new (2025-Act) terminology. The previous/assessment-year →
    # tax-year rename is the Act's headline change; a query in old terms otherwise
    # never matches the new section (s.3). Production would maintain a fuller map.
    _GLOSSARY = {
        "previous year": "tax year",
        "assessment year": "tax year",
    }

    def _expand_query(self, query: str) -> str:
        ql = query.lower()
        extra = [new for old, new in self._GLOSSARY.items() if old in ql and new not in ql]
        return f"{query} {' '.join(extra)}".strip() if extra else query

    def _hyde(self, query: str, llm: LLMProvider | None = None) -> str:
        """Generate a hypothetical answer to improve embedding quality."""
        provider = llm or self.llm
        hypothetical = provider.generate(
            messages=[
                {"role": "system", "content": "You are a legal document expert. Write a concise hypothetical passage that would answer the query. Be factual and specific."},
                {"role": "user", "content": f"Write a hypothetical document passage that answers: {query}"},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        logger.info("HyDE generated: %s...", hypothetical[:80])
        return hypothetical

    def _expand_pages(self, page_numbers: list[int], doc_id: str) -> list[int]:
        """Expand retrieved pages to include neighbors (prev/next page)."""
        expanded = set()
        for page_num in page_numbers:
            for neighbor in range(
                page_num - self.neighbor_pages,
                page_num + self.neighbor_pages + 1
            ):
                if neighbor > 0:
                    expanded.add(neighbor)
        return list(expanded)

    def _get_sub_chunks(self, page_numbers: list[int], doc_ids: list[str]) -> list[tuple[Chunk, float]]:
        """Get sub-chunks that belong to the retrieved pages."""
        allowed_pages = set(page_numbers)
        allowed_docs = set(doc_ids)
        candidates = []
        for chunk in self.chunk_store.chunks:
            if chunk.page_number in allowed_pages and chunk.doc_id in allowed_docs:
                candidates.append((chunk, 0.0))
        logger.info("Sub-chunk candidates from %d pages: %d", len(allowed_pages), len(candidates))
        return candidates

    def _chunk_references(self, chunk: Chunk) -> list[str]:
        """Section IDs this chunk cites. Use precomputed `references` if present
        (statute chunks), else extract "section N" mentions from the text."""
        if getattr(chunk, "references", None):
            return chunk.references
        return self._ref_extractor._extract_references(chunk.text, chunk.section_id or "")

    def _resolve_references(
        self,
        chunks: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """
        Cross-reference resolver: when a retrieved chunk cites "section N", pull
        section N's full text into the candidate pool so it can be reranked into the
        context. Runs BEFORE rerank so resolved sections compete for the top-k.
        BFS to `ref_depth` hops. No-op if the section index is absent.
        """
        if not self._section_lookup:
            return chunks

        visited: set[str] = {chunk.chunk_id for chunk, _ in chunks}
        result = list(chunks)
        frontier = [chunk for chunk, _ in chunks]
        depth = self.ref_depth
        added = 0
        while depth > 0 and frontier:
            nxt: list[Chunk] = []
            for chunk in frontier:
                for sec_id in self._chunk_references(chunk):
                    target = self._section_lookup.get(sec_id)
                    if target is None or target.chunk_id in visited:
                        continue
                    visited.add(target.chunk_id)
                    result.append((target, 0.0))   # rerank assigns the real score
                    nxt.append(target)
                    added += 1
            frontier = nxt
            depth -= 1

        if added:
            logger.info("Reference resolver added %d section chunks (depth=%d)", added, self.ref_depth)
        return result

    def retrieve(self, query: str, llm: LLMProvider | None = None) -> list[tuple[Chunk, float]]:
        # Step 0: expand old→new terminology so a query in 1961-Act terms can still
        # match the 2025 Act's renamed concepts (e.g. "previous year" → "tax year").
        query = self._expand_query(query)

        # Step 1: HyDE — embed a hypothetical answer instead of the raw query
        hypothetical = self._hyde(query, llm)
        search_text = hypothetical

        search_vector = self.embedder.embed_chunks([
            Chunk(chunk_id="q", doc_id="q", text=search_text,
                  page_number=-1, chunk_index=0, char_count=len(search_text))
        ])[0]

        if self.hybrid:
            # Step 2 (hybrid): fuse a dense sub-chunk ranking with a BM25 keyword ranking.
            # BM25 nails rare exact tokens ("tax year", "182 days") that embeddings blur.
            dense_ranked = self.chunk_store.search(search_vector, self.top_k)
            bm25_ranked = self.bm25.retrieve(query, self.top_k)
            ranked_lists = [dense_ranked, bm25_ranked]
            # Also rank section-level chunks: their distinctive headings (e.g. "Definition
            # of tax year") make whole-section *definitions* directly retrievable, which
            # sub-chunks of a ubiquitous term ("tax year") cannot surface.
            if self.section_store is not None:
                ranked_lists.append(self.section_store.search(search_vector, self.top_k))
            candidates = self._rrf_fuse(ranked_lists)

            # Step 3: neighbor-page expansion (spanning lever) around the fused top hits.
            seen = {c.chunk_id for c, _ in candidates}
            top_pages = [c.page_number for c, _ in candidates[:self.top_k]]
            top_docs = [c.doc_id for c, _ in candidates[:self.top_k]]
            expanded_pages = set(self._expand_pages(top_pages, top_docs[0] if top_docs else ""))
            allowed_docs = set(top_docs)
            for chunk in self.chunk_store.chunks:
                if (chunk.page_number in expanded_pages and chunk.doc_id in allowed_docs
                        and chunk.chunk_id not in seen):
                    candidates.append((chunk, 0.0))  # context chunk; scored by reranker
                    seen.add(chunk.chunk_id)
        else:
            # Step 2 (dense-only): page search → neighbor expansion → sub-chunks.
            page_results = self.page_store.search(search_vector, self.top_k)
            page_numbers = [chunk.page_number for chunk, _ in page_results]
            doc_ids = [chunk.doc_id for chunk, _ in page_results]
            expanded_pages = self._expand_pages(page_numbers, doc_ids[0] if doc_ids else "")
            candidates = self._get_sub_chunks(expanded_pages, doc_ids)
            if not candidates:
                return page_results  # fallback to page results

        if not candidates:
            return []

        # Step 5: Follow cross-references BEFORE rerank — pull cited sections into the
        # pool so they compete for the top-k (statute only; no-op when no section index).
        candidates = self._resolve_references(candidates)

        # Step 6: Re-rank to a WIDER pool (CrossEncoder if available; else RRF order),
        # so the currency weighting in step 7 has room to reorder before the top-k cut.
        final_k = self.reranker.top_k
        reranked = self.reranker.rerank(query, candidates, top_n=self.rerank_pool_n)

        # Step 7: Currency-aware selection (D1) — explainer-aware fusion. Down-weight
        # stale-explainer chunks RELATIVE to authoritative statute, on the normalized
        # relevance score, then take the top-k. Relevance still leads (a strongly-
        # relevant booklet chunk beats a weak statute one); no-op for single-source sets.
        if self.currency_aware:
            resolved = self._currency_select(reranked, final_k)
        else:
            resolved = reranked[:final_k]

        logger.info("LexRAG final: %d chunks", len(resolved))
        return resolved

    def _currency_select(
        self, scored: list[tuple[Chunk, float]], k: int,
    ) -> list[tuple[Chunk, float]]:
        def is_explainer(c: Chunk) -> bool:
            return c.metadata.get("source_type") == "explainer" or c.metadata.get("authority") == "none"

        has_statute = any(c.metadata.get("authority") == "statute" for c, _ in scored)
        has_explainer = any(is_explainer(c) for c, _ in scored)
        if not (has_statute and has_explainer):
            return scored[:k]   # single-source set — authority is irrelevant

        vals = [s for _, s in scored]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        weighted = []
        for c, s in scored:
            norm = (s - lo) / rng                       # relevance in [0,1]
            w = self.explainer_weight if is_explainer(c) else 1.0
            weighted.append((c, s, norm * w))           # keep original score for return
        weighted.sort(key=lambda x: x[2], reverse=True)
        return [(c, s) for c, s, _ in weighted[:k]]
