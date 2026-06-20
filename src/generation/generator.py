import logging
import re
import time

from src.data.chunking import Chunk
from src.generation.prompts import build_prompt, build_currency_aware_prompt
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, llm: LLMProvider, cfg: dict):
        self.llm = llm
        self.max_tokens = cfg["generation"]["max_tokens"]
        self.temperature = cfg["generation"]["temperature"]
        logger.info("Generator ready — provider: %s", type(llm).__name__)

    def generate(self, query: str, chunks: list[tuple[Chunk, float]],
                 currency_aware: bool = False) -> dict:
        # currency_aware (LexRAG-only): annotate sources with D1 authority/currency tags
        # and instruct the model to prefer authoritative+current law on points of law.
        messages = (build_currency_aware_prompt if currency_aware else build_prompt)(query, chunks)

        start = time.time()
        answer = self.llm.generate(
            messages, max_tokens=self.max_tokens, temperature=self.temperature
        )
        latency_ms = int((time.time() - start) * 1000)

        cited_indices = [
            int(n) - 1 for n in re.findall(r'\[SOURCE (\d+)\]', answer)
        ]
        citations = [
            {
                "source_num": i + 1,
                "doc_id": chunks[i][0].doc_id,
                "page_number": chunks[i][0].page_number,
                "text_snippet": chunks[i][0].text[:100],
            }
            for i in cited_indices
            if i < len(chunks)
        ]

        uncertainty_phrases = ["do not contain", "not found", "cannot find", "no information"]
        confidence = "low" if any(p in answer.lower() for p in uncertainty_phrases) else "high"

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "generation_time_ms": latency_ms,
            "model": type(self.llm).__name__,
            "chunks_used": len(chunks),
        }
