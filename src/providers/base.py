from abc import ABC, abstractmethod
import numpy as np


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Size of the embedding vector."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts. Returns ndarray of shape (N, dimensions)."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate text from a messages list.
        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        Returns the assistant reply string.
        """
