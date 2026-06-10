import logging
import os
import numpy as np
import httpx
from src.providers.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
_HF_API_URL = f"https://api-inference.huggingface.co/models/{_HF_MODEL}"


class FastEmbedProvider(EmbeddingProvider):
    """
    ONNX-based text embeddings — no API key required.
    Uses fastembed when available (Vercel), falls back to sentence_transformers locally.
    Both use BAAI/bge-small-en-v1.5, producing compatible 384-dim vectors.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        self._dims = 384
        self._backend = None
        self._st_model = None
        self._fe_model = None
        self._init_backend()

    def _init_backend(self):
        # Try fastembed first (lightweight, works on Vercel without torch)
        try:
            from fastembed import TextEmbedding
            # Use a local cache dir to avoid Windows symlink permission errors
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "fastembed")
            cache_dir = os.path.abspath(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            self._fe_model = TextEmbedding(self.MODEL_NAME, cache_dir=cache_dir)
            self._backend = "fastembed"
            logger.info("FastEmbedProvider using fastembed backend")
        except Exception as fe_err:
            logger.warning("fastembed unavailable (%s), trying sentence_transformers", fe_err)
            # Fall back to sentence_transformers (needs torch, but works locally)
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.MODEL_NAME)
                self._backend = "sentence_transformers"
                logger.info("FastEmbedProvider using sentence_transformers backend")
            except Exception as st_err:
                raise RuntimeError(
                    f"No embedding backend available. fastembed error: {fe_err}. "
                    f"sentence_transformers error: {st_err}"
                )

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._backend == "fastembed":
            vectors = list(self._fe_model.embed(texts))
            return np.array(vectors, dtype=np.float32)
        else:
            vectors = self._st_model.encode(texts, normalize_embeddings=True)
            return np.array(vectors, dtype=np.float32)


class HuggingFaceLLMProvider(LLMProvider):
    """HuggingFace Inference API — free tier, no key required (rate-limited)."""

    def __init__(self, hf_token: str | None = None):
        self._headers = {"Content-Type": "application/json"}
        if hf_token:
            self._headers["Authorization"] = f"Bearer {hf_token}"

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 800,
        temperature: float = 0.1,
    ) -> str:
        prompt = self._format_messages(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.05),  # HF API requires > 0
                "return_full_text": False,
            },
        }
        try:
            response = httpx.post(
                _HF_API_URL,
                headers=self._headers,
                json=payload,
                timeout=60.0,
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "").strip()
            logger.warning("HF API status %s: %s", response.status_code, response.text[:200])
        except Exception as exc:
            logger.warning("HF API failed: %s", exc)
        return (
            "The open-source LLM service is temporarily unavailable. "
            "Add an OpenAI API key in Settings for full functionality, or retry shortly."
        )

    def _format_messages(self, messages: list[dict]) -> str:
        """Serialize messages to Mistral [INST] format."""
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_parts = [m["content"] for m in messages if m["role"] == "user"]
        user_text = "\n".join(user_parts)
        if system:
            return f"<s>[INST] {system}\n\n{user_text} [/INST]"
        return f"<s>[INST] {user_text} [/INST]"
