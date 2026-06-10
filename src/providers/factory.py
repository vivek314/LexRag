import os
import logging
from src.providers.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)


def get_providers(
    openai_api_key: str | None = None,
    hf_token: str | None = None,
) -> tuple[LLMProvider, EmbeddingProvider]:
    """
    Return (llm, embedder) based on available keys.

    Priority: explicit openai_api_key arg > OPENAI_API_KEY env var > OSS fallback.
    The OSS path uses fastembed (local ONNX) + HuggingFace Inference API (free tier).
    """
    resolved_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if resolved_key:
        logger.info("Provider mode: OpenAI")
        from src.providers.openai_provider import OpenAILLMProvider, OpenAIEmbeddingProvider
        return (
            OpenAILLMProvider(api_key=resolved_key),
            OpenAIEmbeddingProvider(api_key=resolved_key),
        )

    logger.info("Provider mode: open-source (fastembed + HuggingFace Inference API)")
    from src.providers.hf_provider import HuggingFaceLLMProvider, FastEmbedProvider
    return (
        HuggingFaceLLMProvider(hf_token=hf_token or os.getenv("HF_TOKEN")),
        FastEmbedProvider(),
    )
