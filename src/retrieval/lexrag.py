import logging
import numpy as np

from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore
from src.data.chunking import Chunk
from src.retrieval.reranker import Reranker
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

        # Build section_id -> section-level Chunk lookup for reference resolution.
        # Only populated when statute chunks exist (section_id is set).
        self._section_lookup: dict[str, Chunk] = {
            c.section_id: c
            for c in self.chunk_store.chunks
            if c.section_id and c.metadata.get("level") == "section"
        }
        self.ref_depth = cfg["retrieval"].get("ref_resolve_depth", 2)
        logger.info(
            "LexRAGRetriever ready — section lookup: %d sections", len(self._section_lookup)
        )

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

    def _resolve_references(
        self,
        chunks: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """
        DFS reference resolver for statute documents.
        Score decays by 0.5 per hop so directly retrieved chunks always rank above resolved ones.
        """
        if not self._section_lookup:
            return chunks

        visited: set[str] = {chunk.chunk_id for chunk, _ in chunks}
        result = list(chunks)

        def dfs(chunk: Chunk, score: float, depth: int) -> None:
            if depth == 0:
                return
            for sec_id in getattr(chunk, "references", []):
                target = self._section_lookup.get(sec_id)
                if target is None or target.chunk_id in visited:
                    continue
                visited.add(target.chunk_id)
                decayed = score * 0.5
                result.append((target, decayed))
                dfs(target, decayed, depth - 1)

        for chunk, score in chunks:
            dfs(chunk, score, self.ref_depth)

        logger.info(
            "Reference resolver added %d chunks (depth=%d)",
            len(result) - len(chunks),
            self.ref_depth,
        )
        return result

    def retrieve(self, query: str, llm: LLMProvider | None = None) -> list[tuple[Chunk, float]]:
        # Step 1: HyDE — embed a hypothetical answer instead of the raw query
        hypothetical = self._hyde(query, llm)
        search_text = hypothetical

        search_vector = self.embedder.embed_chunks([
            Chunk(chunk_id="q", doc_id="q", text=search_text,
                  page_number=-1, chunk_index=0, char_count=len(search_text))
        ])[0]

        # Step 2: Search page-level index → find relevant pages
        page_results = self.page_store.search(search_vector, self.top_k)
        page_numbers = [chunk.page_number for chunk, _ in page_results]
        doc_ids = [chunk.doc_id for chunk, _ in page_results]

        # Step 3: Expand to neighboring pages
        expanded_pages = self._expand_pages(page_numbers, doc_ids[0] if doc_ids else "")

        # Step 4: Get sub-chunks within those pages
        candidates = self._get_sub_chunks(expanded_pages, doc_ids)

        # Step 5: Re-rank candidates
        if not candidates:
            return page_results  # fallback to page results

        reranked = self.reranker.rerank(query, candidates)

        # Step 6: Follow cross-references (statute docs only, no-op for others)
        resolved = self._resolve_references(reranked)
        logger.info("LexRAG final: %d chunks", len(resolved))
        return resolved
