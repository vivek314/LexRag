import logging
import numpy as np
import httpx
from src.providers.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
_HF_API_URL = f"https://api-inference.huggingface.co/models/{_HF_MODEL}"


class FastEmbedProvider(EmbeddingProvider):
    """Local ONNX-based text embeddings — no API key required."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name)
        self._dims = 384

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._model.embed(texts))
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
