"""Groq LLM provider — OpenAI-compatible API, free tier, fast Llama models.
Used for the public live demo (no paid OpenAI usage)."""
from openai import OpenAI
from src.providers.base import LLMProvider


class GroqLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def generate(self, messages: list[dict], max_tokens: int = 1500,
                 temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
