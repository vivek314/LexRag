import numpy as np
from openai import OpenAI
from src.providers.base import EmbeddingProvider, LLMProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._dims = 1536
        self._model = model
        self._client = OpenAI(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self._dims

    # OpenAI limits each embeddings request to 300K tokens / 2048 inputs.
    # Batch by a conservative token budget (≈ chars/4) so large corpora don't 400.
    _MAX_TOKENS_PER_REQUEST = 250_000
    _MAX_INPUTS_PER_REQUEST = 2048

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        batch: list[str] = []
        batch_tokens = 0
        for t in texts:
            est = max(1, len(t) // 4)
            if batch and (
                batch_tokens + est > self._MAX_TOKENS_PER_REQUEST
                or len(batch) >= self._MAX_INPUTS_PER_REQUEST
            ):
                resp = self._client.embeddings.create(model=self._model, input=batch)
                out.extend(r.embedding for r in resp.data)
                batch, batch_tokens = [], 0
            batch.append(t)
            batch_tokens += est
        if batch:
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(r.embedding for r in resp.data)
        return np.array(out, dtype=np.float32)


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._model = model
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
