# LexRag Provider Strategy + GitHub + Vercel Deployment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement strategy pattern for LLM + embeddings (OpenAI if user provides key, otherwise open-source fallback), strip all sensitive info, push to GitHub, and deploy to Vercel with a working link.

**Architecture:** The app always uses `fastembed` (BAAI/bge-small-en-v1.5, 384 dims) for FAISS retrieval on Vercel — these pre-built OSS indices are committed to git. The LLM strategy switches between GPT-4o-mini (user provides key in UI) and HuggingFace Inference API with Mistral-7B (free fallback). Per-request LLM override via `X-OpenAI-Api-Key` header allows dynamic switching without reloading indices.

**Tech Stack:** Python 3.12, FastAPI, FAISS, fastembed (ONNX, no torch), HuggingFace Inference API (free tier), OpenAI SDK, Vercel Python runtime (`@vercel/python`).

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| CREATE | `src/providers/__init__.py` | Package exports |
| CREATE | `src/providers/base.py` | Abstract `LLMProvider`, `EmbeddingProvider` |
| CREATE | `src/providers/openai_provider.py` | OpenAI implementations of both strategies |
| CREATE | `src/providers/hf_provider.py` | fastembed + HuggingFace Inference API implementations |
| CREATE | `src/providers/factory.py` | `get_providers()` factory function |
| MODIFY | `src/data/embedder.py` | Accept `EmbeddingProvider` instead of calling OpenAI directly |
| MODIFY | `src/data/indexing.py` | Accept optional `EmbeddingProvider` param |
| MODIFY | `src/generation/generator.py` | Accept `LLMProvider` instead of calling OpenAI directly |
| MODIFY | `src/retrieval/baseline.py` | Accept `EmbeddingProvider` in constructor |
| MODIFY | `src/retrieval/lexrag.py` | Accept `LLMProvider`+`EmbeddingProvider`; `retrieve()` takes optional LLM override |
| MODIFY | `src/retrieval/reranker.py` | Graceful degrade when sentence-transformers not installed |
| MODIFY | `src/api/main.py` | Wire providers, read `X-OpenAI-Api-Key` header |
| MODIFY | `src/api/static/index.html` | Add settings button + API key modal |
| MODIFY | `src/api/static/app.js` | API key localStorage management + inject header |
| MODIFY | `src/api/static/style.css` | Modal styles |
| MODIFY | `requirements.txt` | Add `fastembed==0.3.1`; keep existing deps |
| CREATE | `requirements-vercel.txt` | Vercel-safe deps (no torch, no sentence-transformers) |
| CREATE | `scripts/build_oss_indices.py` | One-time script: build fastembed FAISS indices → `data/indices/oss/` |
| MODIFY | `.gitignore` | Unblock `data/processed/*.json` and `data/indices/oss/` |
| CREATE | `.env.example` | Key template (no real values) |
| CREATE | `api/index.py` | Vercel ASGI entrypoint (imports FastAPI `app`) |
| CREATE | `vercel.json` | Vercel build + routing config |

---

### Task 1: Security Hardening

**Files:**
- Modify: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Update .gitignore**

Replace `C:\Users\Vivek\LexRag\.gitignore` entirely with:

```
venv/
__pycache__/
*.pyc
*.pyo
.env
data/raw/
data/processed/embedding_cache/
data/processed/embedding_cache_oss/
data/indices/
!data/indices/oss/
!data/indices/oss/**
*.npy
logs/
.DS_Store
*.egg-info/
dist/
build/
.pytest_cache/
```

Key changes vs original:
- `data/processed/` is now ALLOWED (JSON files are small, needed for Vercel)
- Only `embedding_cache/` subdirs are blocked (they contain `.npy` files)
- `data/indices/oss/` is explicitly un-blocked (pre-built fastembed indices for Vercel)

- [ ] **Step 2: Create .env.example**

Create `C:\Users\Vivek\LexRag\.env.example`:
```
# Copy this to .env and fill in values. Never commit .env!

# OpenAI (optional) — the app works without this using open-source LLM
OPENAI_API_KEY=

# HuggingFace token (optional) — improves rate limits on free HF Inference API
HF_TOKEN=
```

- [ ] **Step 3: Verify .env is not tracked**

```bash
git ls-files --error-unmatch .env
```
Expected output: `error: pathspec '.env' did not match any file(s) known to git`
That error is GOOD — means .env is not tracked.
If it IS tracked (no error), run: `git rm --cached .env`

- [ ] **Step 4: Commit**

```bash
git add .gitignore .env.example
git commit -m "security: harden .gitignore, add .env.example template"
```

---

### Task 2: Abstract Provider Base

**Files:**
- Create: `src/providers/__init__.py`
- Create: `src/providers/base.py`

- [ ] **Step 1: Create the providers package**

Create `C:\Users\Vivek\LexRag\src\providers\__init__.py` as empty file.

- [ ] **Step 2: Create base.py**

Create `C:\Users\Vivek\LexRag\src\providers\base.py`:

```python
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
```

- [ ] **Step 3: Verify import works**

```bash
cd C:\Users\Vivek\LexRag && python -c "from src.providers.base import EmbeddingProvider, LLMProvider; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/providers/
git commit -m "feat(providers): add abstract EmbeddingProvider and LLMProvider interfaces"
```

---

### Task 3: OpenAI Provider

**Files:**
- Create: `src/providers/openai_provider.py`
- Create: `tests/test_providers.py`

- [ ] **Step 1: Write the failing tests**

Create `C:\Users\Vivek\LexRag\tests\test_providers.py`:

```python
import numpy as np
import os
import pytest
from unittest.mock import MagicMock, patch


def test_openai_embedder_dimensions():
    with patch("src.providers.openai_provider.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        MockOpenAI.return_value = mock_client
        from src.providers.openai_provider import OpenAIEmbeddingProvider
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        assert provider.dimensions == 1536


def test_openai_embedder_returns_ndarray():
    with patch("src.providers.openai_provider.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        MockOpenAI.return_value = mock_client
        from src.providers.openai_provider import OpenAIEmbeddingProvider
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        result = provider.embed(["hello world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1536)


def test_openai_llm_generate():
    with patch("src.providers.openai_provider.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test answer"))]
        )
        MockOpenAI.return_value = mock_client
        from src.providers.openai_provider import OpenAILLMProvider
        provider = OpenAILLMProvider(api_key="sk-test")
        result = provider.generate([{"role": "user", "content": "test"}])
        assert result == "Test answer"


def test_factory_returns_openai_when_key_provided():
    with patch("src.providers.openai_provider.OpenAI"):
        from src.providers.factory import get_providers
        from src.providers.openai_provider import OpenAILLMProvider, OpenAIEmbeddingProvider
        llm, embedder = get_providers(openai_api_key="sk-test")
        assert isinstance(llm, OpenAILLMProvider)
        assert isinstance(embedder, OpenAIEmbeddingProvider)


def test_factory_returns_oss_when_no_key():
    # Temporarily clear env key if set
    original = os.environ.pop("OPENAI_API_KEY", None)
    try:
        from importlib import reload
        import src.providers.factory as fac
        reload(fac)
        from src.providers.factory import get_providers
        from src.providers.hf_provider import HuggingFaceLLMProvider, FastEmbedProvider
        llm, embedder = get_providers(openai_api_key=None)
        assert isinstance(llm, HuggingFaceLLMProvider)
        assert isinstance(embedder, FastEmbedProvider)
    finally:
        if original:
            os.environ["OPENAI_API_KEY"] = original


def test_hf_llm_provider_returns_string():
    with patch("src.providers.hf_provider.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"generated_text": "Legal answer here"}],
        )
        from src.providers.hf_provider import HuggingFaceLLMProvider
        provider = HuggingFaceLLMProvider()
        result = provider.generate([{"role": "user", "content": "test"}])
        assert isinstance(result, str)
        assert len(result) > 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd C:\Users\Vivek\LexRag && python -m pytest tests/test_providers.py -v 2>&1 | head -20
```
Expected: `ImportError` or `ModuleNotFoundError` for `src.providers.openai_provider`

- [ ] **Step 3: Implement OpenAI provider**

Create `C:\Users\Vivek\LexRag\src\providers\openai_provider.py`:

```python
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

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return np.array([r.embedding for r in response.data], dtype=np.float32)


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
```

- [ ] **Step 4: Run the 3 OpenAI tests**

```bash
cd C:\Users\Vivek\LexRag && python -m pytest tests/test_providers.py::test_openai_embedder_dimensions tests/test_providers.py::test_openai_embedder_returns_ndarray tests/test_providers.py::test_openai_llm_generate -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/providers/openai_provider.py tests/test_providers.py
git commit -m "feat(providers): implement OpenAIEmbeddingProvider and OpenAILLMProvider"
```

---

### Task 4: Open-Source Provider (fastembed + HuggingFace)

**Files:**
- Modify: `requirements.txt`
- Create: `src/providers/hf_provider.py`

- [ ] **Step 1: Add fastembed to requirements.txt**

Edit `C:\Users\Vivek\LexRag\requirements.txt` — append `fastembed==0.3.1` after the openai line:

```
fastapi==0.111.0
uvicorn==0.30.1
PyMuPDF==1.24.5
python-dotenv==1.0.1
pyyaml==6.0.1
requests==2.32.3
tqdm==4.66.4
numpy==1.26.4
faiss-cpu==1.8.0
sentence-transformers==3.0.0
torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
transformers==4.41.2
openai==1.30.5
fastembed==0.3.1
pytest==8.2.1
python-multipart
httpx<0.28.0
```

- [ ] **Step 2: Install fastembed**

```bash
cd C:\Users\Vivek\LexRag && pip install fastembed==0.3.1
```
Expected: `Successfully installed fastembed-...`

- [ ] **Step 3: Implement hf_provider.py**

Create `C:\Users\Vivek\LexRag\src\providers\hf_provider.py`:

```python
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
```

- [ ] **Step 4: Run all provider tests**

```bash
cd C:\Users\Vivek\LexRag && python -m pytest tests/test_providers.py::test_hf_llm_provider_returns_string -v
```
Expected: 1 PASSED (fastembed tests skip model download in mock)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/providers/hf_provider.py
git commit -m "feat(providers): implement FastEmbedProvider and HuggingFaceLLMProvider"
```

---

### Task 5: Provider Factory

**Files:**
- Create: `src/providers/factory.py`
- Modify: `src/providers/__init__.py`

- [ ] **Step 1: Implement factory.py**

Create `C:\Users\Vivek\LexRag\src\providers\factory.py`:

```python
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
    The OSS path always uses fastembed + HuggingFace Inference API (no key needed).
    """
    resolved_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if resolved_key:
        logger.info("Provider mode: OpenAI")
        from src.providers.openai_provider import OpenAILLMProvider, OpenAIEmbeddingProvider
        return (
            OpenAILLMProvider(api_key=resolved_key),
            OpenAIEmbeddingProvider(api_key=resolved_key),
        )

    logger.info("Provider mode: open-source (fastembed + HuggingFace)")
    from src.providers.hf_provider import HuggingFaceLLMProvider, FastEmbedProvider
    return (
        HuggingFaceLLMProvider(hf_token=hf_token or os.getenv("HF_TOKEN")),
        FastEmbedProvider(),
    )
```

- [ ] **Step 2: Populate __init__.py**

Replace `C:\Users\Vivek\LexRag\src\providers\__init__.py`:

```python
from src.providers.factory import get_providers
from src.providers.base import EmbeddingProvider, LLMProvider

__all__ = ["get_providers", "EmbeddingProvider", "LLMProvider"]
```

- [ ] **Step 3: Run factory tests**

```bash
cd C:\Users\Vivek\LexRag && python -m pytest tests/test_providers.py -k "factory" -v
```
Expected: 2 PASSED

- [ ] **Step 4: Commit**

```bash
git add src/providers/factory.py src/providers/__init__.py
git commit -m "feat(providers): add get_providers factory with OpenAI/OSS strategy selection"
```

---

### Task 6: Refactor Embedder

**Files:**
- Modify: `src/data/embedder.py`

The `Embedder` class becomes infrastructure-only (caching + batching) and delegates the actual API call to an injected `EmbeddingProvider`.

- [ ] **Step 1: Replace src/data/embedder.py**

```python
import hashlib
import logging
from pathlib import Path

import numpy as np

from src.data.chunking import Chunk
from src.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, provider: EmbeddingProvider, cache_dir: str = "data/processed/embedding_cache"):
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        # Provider type is part of the cache key — OpenAI and fastembed vectors must not collide.
        prefix = type(self.provider).__name__
        key = hashlib.md5(f"{prefix}:{text}".encode()).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def _load(self, text: str) -> np.ndarray | None:
        p = self._cache_path(text)
        return np.load(str(p)) if p.exists() else None

    def _save(self, text: str, vec: np.ndarray) -> None:
        np.save(str(self._cache_path(text)), vec)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        vectors: list[np.ndarray | None] = []
        miss_texts: list[str] = []
        miss_idx: list[int] = []

        for i, chunk in enumerate(chunks):
            cached = self._load(chunk.text)
            if cached is not None:
                vectors.append(cached)
            else:
                vectors.append(None)
                miss_texts.append(chunk.text)
                miss_idx.append(i)

        if miss_texts:
            new_vecs = self.provider.embed(miss_texts)
            for j, idx in enumerate(miss_idx):
                vectors[idx] = new_vecs[j]
                self._save(miss_texts[j], new_vecs[j])
            logger.info("Embedded %d new / %d from cache", len(miss_texts), len(chunks) - len(miss_texts))

        return np.vstack(vectors)
```

- [ ] **Step 2: Verify import**

```bash
cd C:\Users\Vivek\LexRag && python -c "from src.data.embedder import Embedder; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/data/embedder.py
git commit -m "refactor(embedder): delegate to EmbeddingProvider strategy, preserve cache logic"
```

---

### Task 7: Refactor Generator

**Files:**
- Modify: `src/generation/generator.py`

- [ ] **Step 1: Replace src/generation/generator.py**

```python
import logging
import re
import time

from src.data.chunking import Chunk
from src.generation.prompts import build_prompt
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, llm: LLMProvider, cfg: dict):
        self.llm = llm
        self.max_tokens = cfg["generation"]["max_tokens"]
        self.temperature = cfg["generation"]["temperature"]
        logger.info("Generator ready — provider: %s", type(llm).__name__)

    def generate(self, query: str, chunks: list[tuple[Chunk, float]]) -> dict:
        messages = build_prompt(query, chunks)

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
```

- [ ] **Step 2: Verify import**

```bash
cd C:\Users\Vivek\LexRag && python -c "from src.generation.generator import Generator; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/generation/generator.py
git commit -m "refactor(generator): use LLMProvider strategy instead of direct OpenAI client"
```

---

### Task 8: Refactor Retrievers

**Files:**
- Modify: `src/retrieval/baseline.py`
- Modify: `src/retrieval/lexrag.py`
- Modify: `src/retrieval/reranker.py`

- [ ] **Step 1: Update baseline.py**

Replace `C:\Users\Vivek\LexRag\src\retrieval\baseline.py`:

```python
import logging
import numpy as np

from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore
from src.data.chunking import Chunk
from src.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class BaselineRetriever:
    def __init__(self, cfg: dict, embedding_provider: EmbeddingProvider):
        self.cfg = cfg
        self.top_k = cfg["retrieval"]["top_k"]
        self.embedder = Embedder(embedding_provider, cfg["embedding"]["cache_dir"])
        self.store = FaissStore(cfg)
        self.store.load("baseline")
        logger.info("BaselineRetriever ready")

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        query_vector = self.embedder.embed_chunks([
            Chunk(chunk_id="q", doc_id="q", text=query,
                  page_number=-1, chunk_index=0, char_count=len(query))
        ])[0]
        results = self.store.search(query_vector, self.top_k)
        logger.info("Baseline retrieved %d chunks", len(results))
        return results
```

- [ ] **Step 2: Update reranker.py to gracefully degrade without sentence-transformers**

Read the current `src/retrieval/reranker.py` to understand its structure, then add the graceful fallback at the top:

Open `C:\Users\Vivek\LexRag\src\retrieval\reranker.py` and change the import block from:
```python
from sentence_transformers import CrossEncoder
```
to:
```python
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    _HAS_CROSS_ENCODER = False
    CrossEncoder = None  # type: ignore
```

Then in the `__init__` method, guard model loading:
```python
def __init__(self, cfg: dict):
    if _HAS_CROSS_ENCODER:
        model_name = cfg["retrieval"]["reranker"]["model"]
        self.model = CrossEncoder(model_name, max_length=cfg["retrieval"]["reranker"]["max_length"])
    else:
        self.model = None
    self.top_k = cfg["retrieval"]["rerank_top_k"]
```

And in the `rerank` method, short-circuit when model is unavailable:
```python
def rerank(self, query: str, candidates: list[tuple]) -> list[tuple]:
    if not self.model:
        # No cross-encoder: return candidates sorted by existing score (truncated)
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:self.top_k]
    # ... existing rerank logic unchanged ...
```

- [ ] **Step 3: Update lexrag.py**

In `C:\Users\Vivek\LexRag\src\retrieval\lexrag.py`, make the following changes:

**Change the imports** (top of file) from:
```python
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore
from src.data.chunking import Chunk
from src.retrieval.reranker import Reranker

load_dotenv()
logger = logging.getLogger(__name__)
```
to:
```python
import logging
import numpy as np

from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore
from src.data.chunking import Chunk
from src.retrieval.reranker import Reranker
from src.providers.base import LLMProvider, EmbeddingProvider

logger = logging.getLogger(__name__)
```

**Change `__init__` signature and body** from:
```python
def __init__(self, cfg: dict):
    self.cfg = cfg
    self.top_k = cfg["retrieval"]["top_k"]
    self.neighbor_pages = cfg["retrieval"]["neighbor_pages"]

    self.embedder = Embedder(cfg)
    self.reranker = Reranker(cfg)
    self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    self.page_store = FaissStore(cfg)
    self.page_store.load("lexrag_pages")

    self.chunk_store = FaissStore(cfg)
    self.chunk_store.load("lexrag_chunks")
    ...
```
to:
```python
def __init__(self, cfg: dict, llm: LLMProvider, embedding_provider: EmbeddingProvider):
    self.cfg = cfg
    self.top_k = cfg["retrieval"]["top_k"]
    self.neighbor_pages = cfg["retrieval"]["neighbor_pages"]

    self.embedder = Embedder(embedding_provider, cfg["embedding"]["cache_dir"])
    self.reranker = Reranker(cfg)
    self.llm = llm

    self.page_store = FaissStore(cfg)
    self.page_store.load("lexrag_pages")

    self.chunk_store = FaissStore(cfg)
    self.chunk_store.load("lexrag_chunks")
    ...
    # (keep the _section_lookup and ref_depth lines exactly as they are)
```

**Change `_hyde` method** from:
```python
def _hyde(self, query: str) -> str:
    response = self.client.chat.completions.create(
        model=self.cfg["generation"]["model"],
        messages=[
            {"role": "system", "content": "You are a legal document expert. Write a concise hypothetical passage that would answer the query. Be factual and specific."},
            {"role": "user", "content": f"Write a hypothetical document passage that answers: {query}"}
        ],
        temperature=0.0,
        max_tokens=200
    )
    hypothetical = response.choices[0].message.content
    logger.info(f"HyDE generated: {hypothetical[:80]}...")
    return hypothetical
```
to:
```python
def _hyde(self, query: str, llm: LLMProvider | None = None) -> str:
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
```

**Change `retrieve` signature** to accept an optional LLM override:
```python
def retrieve(self, query: str, llm: LLMProvider | None = None) -> list[tuple[Chunk, float]]:
    # Step 1: HyDE — embed a hypothetical answer instead of the raw query
    hypothetical = self._hyde(query, llm)
    search_text = hypothetical
    # ... rest of method unchanged ...
```

- [ ] **Step 4: Verify both retrievers import cleanly**

```bash
cd C:\Users\Vivek\LexRag && python -c "from src.retrieval.baseline import BaselineRetriever; from src.retrieval.lexrag import LexRAGRetriever; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/baseline.py src/retrieval/lexrag.py src/retrieval/reranker.py
git commit -m "refactor(retrieval): inject LLMProvider+EmbeddingProvider; reranker degrades without sentence-transformers"
```

---

### Task 9: Refactor build_indices

**Files:**
- Modify: `src/data/indexing.py`

`build_indices` currently creates its own `Embedder(cfg)` which requires an OpenAI key. Change it to accept an optional `EmbeddingProvider`.

- [ ] **Step 1: Update build_indices in indexing.py**

Change the function signature and embedder creation:

```python
# At top, add import:
from src.providers.base import EmbeddingProvider

# Change function signature from:
def build_indices(cfg: dict) -> None:

# To:
def build_indices(cfg: dict, embedding_provider: EmbeddingProvider | None = None) -> None:

# Change embedder creation from:
    embedder = Embedder(cfg)

# To:
    if embedding_provider is None:
        from src.providers.factory import get_providers
        _, embedding_provider = get_providers()  # uses env key or OSS fallback
    cache_dir = cfg["embedding"].get("cache_dir", "data/processed/embedding_cache")
    embedder = Embedder(embedding_provider, cache_dir)
```

The rest of the function body is unchanged.

- [ ] **Step 2: Verify import**

```bash
cd C:\Users\Vivek\LexRag && python -c "from src.data.indexing import build_indices; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/data/indexing.py
git commit -m "refactor(indexing): accept optional EmbeddingProvider; defaults to factory selection"
```

---

### Task 10: Build OSS FAISS Indices

These fastembed (384-dim) indices are committed to git for the Vercel demo.

**Files:**
- Create: `scripts/build_oss_indices.py`
- Run locally to produce: `data/indices/oss/`

- [ ] **Step 1: Create the build script**

Create `C:\Users\Vivek\LexRag\scripts\build_oss_indices.py`:

```python
#!/usr/bin/env python
"""
Build FAISS indices using fastembed (open-source, no API key).
Output goes to data/indices/oss/ — commit these files for Vercel demo.
Run once locally before deploying: python scripts/build_oss_indices.py
"""
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.providers.hf_provider import FastEmbedProvider
from src.data.indexing import build_indices

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

oss_cfg = copy.deepcopy(cfg)
oss_cfg["data"]["indices_dir"] = "data/indices/oss"
oss_cfg["embedding"]["cache_dir"] = "data/processed/embedding_cache_oss"
# fastembed produces 384-dim vectors
oss_cfg["embedding"]["dimensions"] = 384

provider = FastEmbedProvider()
build_indices(oss_cfg, embedding_provider=provider)
print("OSS indices built → data/indices/oss/")
```

- [ ] **Step 2: Ensure processed JSON files exist**

```bash
dir C:\Users\Vivek\LexRag\data\processed\*.json
```
Expected: Should list `Copyright_Office_Circular_01.json`, `IRS_Circular_E_2024.json`, `it_act_2000.json`.
If none exist, you need to run the processor first (out of scope — the files should already be there from previous runs).

- [ ] **Step 3: Run the index builder**

```bash
cd C:\Users\Vivek\LexRag && python scripts/build_oss_indices.py
```
Expected: Logs showing chunks being embedded, then `OSS indices built → data/indices/oss/`
This takes 2-5 minutes on first run (downloads the ONNX model, ~25 MB to a temp dir).

- [ ] **Step 4: Verify output files**

```bash
dir C:\Users\Vivek\LexRag\data\indices\oss\
```
Expected: `baseline.faiss`, `baseline_chunks.pkl`, `lexrag_pages.faiss`, `lexrag_pages_chunks.pkl`, `lexrag_chunks.faiss`, `lexrag_chunks_chunks.pkl`

- [ ] **Step 5: Commit the OSS indices and JSON data**

```bash
git add data/indices/oss/ data/processed/*.json scripts/build_oss_indices.py
git commit -m "feat: add pre-built fastembed FAISS indices and processed JSON data for Vercel demo"
```

---

### Task 11: Update FastAPI to Wire Providers

**Files:**
- Modify: `src/api/main.py`

The key changes:
1. Always load OSS retrievers at startup (fastembed indices)
2. Read `X-OpenAI-Api-Key` header per-request to select LLM provider
3. `LexRAGRetriever.retrieve()` receives an override LLM when key is provided
4. `Generator` is created per-request with the right LLM

- [ ] **Step 1: Replace src/api/main.py**

```python
# main.py — FastAPI backend for LexRAG UI

import logging
import os
import re
import time
import json
import datetime
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dataclasses import asdict

from src.data.processor import process_pdf
from src.data.indexing import build_indices
from src.retrieval.baseline import BaselineRetriever
from src.retrieval.lexrag import LexRAGRetriever
from src.generation.generator import Generator
from src.providers.factory import get_providers
from src.providers.hf_provider import FastEmbedProvider, HuggingFaceLLMProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="LexRAG Legal Tech Hub", description="Advanced RAG vs Naive RAG Comparison")

# Config — path is relative to project root, resolved from this file's location
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
CONFIG_PATH = _ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# OSS config: point to fastembed indices and OSS embedding cache
_oss_cfg = {
    **cfg,
    "data": {**cfg["data"], "indices_dir": str(_ROOT / "data" / "indices" / "oss")},
    "embedding": {**cfg["embedding"], "cache_dir": str(_ROOT / "data" / "processed" / "embedding_cache_oss"), "dimensions": 384},
}

# Initialize OSS provider and retrievers at startup (always available, no key needed)
_oss_embedder = FastEmbedProvider()
_oss_llm = HuggingFaceLLMProvider(hf_token=os.getenv("HF_TOKEN"))

_baseline_retriever: Optional[BaselineRetriever] = None
_lexrag_retriever: Optional[LexRAGRetriever] = None


def _load_retrievers():
    global _baseline_retriever, _lexrag_retriever
    try:
        _baseline_retriever = BaselineRetriever(_oss_cfg, _oss_embedder)
        _lexrag_retriever = LexRAGRetriever(_oss_cfg, _oss_llm, _oss_embedder)
        logger.info("OSS retrievers loaded successfully.")
    except Exception as e:
        logger.error("Failed to load retrievers: %s. Run scripts/build_oss_indices.py first.", e)


_load_retrievers()


def _get_llm(openai_api_key: Optional[str]):
    """Return the right LLM for this request."""
    if not openai_api_key:
        return _oss_llm
    from src.providers.openai_provider import OpenAILLMProvider
    return OpenAILLMProvider(api_key=openai_api_key)


class QueryRequest(BaseModel):
    query: str


@app.get("/api/stats")
async def get_stats():
    try:
        raw_dir = Path(cfg["data"]["raw_dir"])
        manifest_path = Path(cfg["data"]["manifest_file"])
        num_docs = total_pages = 0
        doc_details = []
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            num_docs = len(manifest)
            total_pages = sum(d.get("num_pages", 0) for d in manifest)
            doc_details = [
                {"doc_id": d["doc_id"], "title": d["title"], "domain": d["domain"], "num_pages": d.get("num_pages", 0)}
                for d in manifest
            ]
        cache_dir = Path(_oss_cfg["embedding"]["cache_dir"])
        cache_hits = len(list(cache_dir.glob("*.npy"))) if cache_dir.exists() else 0
        return {
            "num_docs": num_docs,
            "total_pages": total_pages,
            "cache_size": cache_hits,
            "baseline_chunks": len(_baseline_retriever.store.chunks) if _baseline_retriever else 0,
            "lexrag_subchunks": len(_lexrag_retriever.chunk_store.chunks) if _lexrag_retriever else 0,
            "documents": doc_details,
        }
    except Exception as e:
        logger.error("Stats error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.post("/api/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    x_openai_api_key: Optional[str] = Header(None),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        slug = re.sub(r'[^a-zA-Z0-9]', '_', Path(file.filename).stem)[:60]
        raw_dir = Path(cfg["data"]["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest_path = raw_dir / f"{slug}.pdf"
        with open(dest_path, "wb") as buf:
            buf.write(await file.read())

        manifest_path = Path(cfg["data"]["manifest_file"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
        manifest = [d for d in manifest if d["doc_id"] != slug]
        meta = {
            "doc_id": slug, "title": Path(file.filename).stem, "source": "Uploaded File",
            "domain": "uploaded_documents", "date": datetime.date.today().isoformat(),
            "num_pages": 0, "local_path": str(dest_path).replace("\\", "/"),
        }
        manifest.append(meta)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        processed_dir = Path(cfg["data"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_file = processed_dir / f"{slug}.json"
        out_file.unlink(missing_ok=True)
        doc = process_pdf(str(dest_path), meta, cfg)
        if doc is None:
            raise HTTPException(status_code=500, detail="PDF text extraction failed.")
        out_file.write_text(json.dumps(asdict(doc), indent=2, ensure_ascii=False), encoding="utf-8")
        meta["num_pages"] = doc.num_pages
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        # Always rebuild OSS indices (fastembed); OpenAI indices are local-only
        build_indices(_oss_cfg, embedding_provider=_oss_embedder)
        _load_retrievers()
        return {"status": "success", "doc_id": slug, "title": meta["title"], "num_pages": doc.num_pages}
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def execute_query(
    payload: QueryRequest,
    x_openai_api_key: Optional[str] = Header(None),
):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not _baseline_retriever or not _lexrag_retriever:
        raise HTTPException(status_code=500, detail="RAG system not initialized. Please ingest a document first.")

    llm = _get_llm(x_openai_api_key)
    generator = Generator(llm, cfg)
    provider_mode = "openai" if x_openai_api_key else "open-source"

    try:
        # Baseline RAG
        t0 = time.time()
        baseline_chunks = _baseline_retriever.retrieve(query)[:5]
        baseline_gen = generator.generate(query, baseline_chunks)
        baseline_latency = int((time.time() - t0) * 1000)

        # LexRAG (pass per-request LLM for HyDE)
        t1 = time.time()
        lexrag_chunks = _lexrag_retriever.retrieve(query, llm=llm)
        lexrag_gen = generator.generate(query, lexrag_chunks)
        lexrag_latency = int((time.time() - t1) * 1000)

        return {
            "query": query,
            "provider": provider_mode,
            "baseline": {
                "answer": baseline_gen["answer"],
                "citations": baseline_gen["citations"],
                "confidence": baseline_gen["confidence"],
                "latency_ms": baseline_latency,
                "chunks_used": len(baseline_chunks),
            },
            "lexrag": {
                "answer": lexrag_gen["answer"],
                "citations": lexrag_gen["citations"],
                "confidence": lexrag_gen["confidence"],
                "latency_ms": lexrag_latency,
                "chunks_used": len(lexrag_chunks),
            },
            "comparison": {
                "winner": "lexrag",
                "reasons": [
                    "LexRAG preserves page-level layout for high-fidelity citations.",
                    "LexRAG uses Hypothetical Document Embedding (HyDE) to bridge vocabulary gaps.",
                    "LexRAG performs dynamic neighbor page expansion (±1 page) for multi-page context.",
                    "LexRAG employs CrossEncoder re-ranking for precise chunk scoring.",
                ],
            },
        }
    except Exception as e:
        logger.exception("Query execution failed")
        raise HTTPException(status_code=500, detail=str(e))


# Static files
_static_dir = _HERE / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def get_index():
    index_path = _static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("<html><body><h1>LexRAG UI is loading...</h1></body></html>")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Verify app loads**

```bash
cd C:\Users\Vivek\LexRag && python -c "from src.api.main import app; print('App loaded OK')"
```
Expected: `App loaded OK` (may show warnings about missing indices if OSS not yet built)

- [ ] **Step 3: Commit**

```bash
git add src/api/main.py
git commit -m "feat(api): wire provider strategy; X-OpenAI-Api-Key header for per-request LLM switching"
```

---

### Task 12: Frontend Settings Panel

**Files:**
- Modify: `src/api/static/index.html`
- Modify: `src/api/static/app.js`
- Modify: `src/api/static/style.css`

- [ ] **Step 1: Add settings button and modal to index.html**

In `src/api/static/index.html`, find the line:
```html
            <div class="header-badge">
```
Insert BEFORE it:
```html
            <button class="settings-btn" id="settingsBtn" title="Configure API Key">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 19.07a10 10 0 0 1 0-14.14"/>
                    <path d="M12 2v2M12 20v2M2 12h2M20 12h2"/>
                </svg>
            </button>
```

At the very end of `<body>`, just before `</body>`, add:
```html
    <!-- Settings Modal -->
    <div class="modal-overlay" id="settingsModal">
        <div class="modal-box">
            <div class="modal-header">
                <h3>API Configuration</h3>
                <button class="modal-close" id="settingsClose">&times;</button>
            </div>
            <div class="modal-body">
                <p class="modal-desc">
                    LexRAG works out-of-the-box with a free open-source LLM
                    (HuggingFace Mistral-7B). Provide an OpenAI API key below
                    for higher-quality GPT-4o-mini answers.
                    <br><strong>Your key is stored only in your browser (localStorage) and sent directly to this server — it is never logged.</strong>
                </p>
                <div class="form-group">
                    <label for="apiKeyInput">OpenAI API Key (optional)</label>
                    <input type="password" id="apiKeyInput" placeholder="sk-proj-..." autocomplete="off">
                </div>
                <div class="modal-mode">
                    Current mode: <span class="mode-badge oss" id="modeBadge">Open-Source (Free)</span>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn-clear" id="clearKeyBtn">Clear Key</button>
                <button class="btn-save" id="saveKeyBtn">Save &amp; Use</button>
            </div>
        </div>
    </div>
```

- [ ] **Step 2: Append modal CSS to style.css**

Append to the end of `src/api/static/style.css`:
```css
/* ---- Settings button ---- */
.settings-btn {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 8px;
    padding: 8px;
    cursor: pointer;
    color: var(--text-secondary, #9ca3af);
    display: flex;
    align-items: center;
    transition: all 0.2s;
    margin-right: 12px;
}
.settings-btn:hover { background: rgba(255,255,255,0.14); color: #fff; }

/* ---- Modal overlay ---- */
.modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.65);
    backdrop-filter: blur(4px);
    z-index: 9000;
    align-items: center;
    justify-content: center;
}
.modal-overlay.open { display: flex; }
.modal-box {
    background: #1a1d2e;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    width: min(480px, 92vw);
    overflow: hidden;
}
.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 24px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.modal-header h3 { margin: 0; font-size: 1.05rem; }
.modal-close {
    background: none; border: none;
    font-size: 1.6rem; line-height: 1;
    color: #9ca3af; cursor: pointer;
}
.modal-body { padding: 22px 24px; }
.modal-desc {
    color: #9ca3af; font-size: 0.85rem;
    line-height: 1.55; margin-bottom: 18px;
}
.form-group label { display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 7px; }
.form-group input {
    width: 100%; box-sizing: border-box;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; padding: 9px 13px;
    color: #f3f4f6; font-size: 0.875rem;
}
.form-group input:focus { outline: none; border-color: #6366f1; }
.modal-mode { margin-top: 14px; font-size: 0.85rem; color: #9ca3af; }
.mode-badge {
    padding: 2px 10px; border-radius: 20px;
    font-weight: 600; font-size: 0.78rem;
}
.mode-badge.oss  { background: rgba(34,197,94,0.15); color: #22c55e; }
.mode-badge.openai { background: rgba(99,102,241,0.18); color: #818cf8; }
.modal-footer {
    display: flex; gap: 10px; justify-content: flex-end;
    padding: 14px 24px;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.btn-clear {
    background: none;
    border: 1px solid rgba(255,255,255,0.15);
    color: #9ca3af; padding: 7px 16px;
    border-radius: 8px; cursor: pointer; font-size: 0.85rem;
}
.btn-save {
    background: #6366f1; border: none; color: #fff;
    padding: 7px 18px; border-radius: 8px;
    cursor: pointer; font-weight: 600; font-size: 0.85rem;
}
```

- [ ] **Step 3: Add settings logic to app.js**

Inside `src/api/static/app.js`, inside the `DOMContentLoaded` callback, at the very END (just before the closing `});`), add:

```javascript
    // =========================================================================
    // API Key Settings Modal
    // =========================================================================
    const STORAGE_KEY = 'lexrag_openai_key';
    const settingsBtn   = document.getElementById('settingsBtn');
    const settingsModal = document.getElementById('settingsModal');
    const settingsClose = document.getElementById('settingsClose');
    const apiKeyInput   = document.getElementById('apiKeyInput');
    const saveKeyBtn    = document.getElementById('saveKeyBtn');
    const clearKeyBtn   = document.getElementById('clearKeyBtn');
    const modeBadge     = document.getElementById('modeBadge');

    function getStoredKey() { return localStorage.getItem(STORAGE_KEY) || ''; }

    function updateBadge(key) {
        if (key) {
            modeBadge.textContent = 'OpenAI (GPT-4o-mini)';
            modeBadge.className = 'mode-badge openai';
        } else {
            modeBadge.textContent = 'Open-Source (Free)';
            modeBadge.className = 'mode-badge oss';
        }
    }
    updateBadge(getStoredKey());

    settingsBtn.addEventListener('click', () => {
        apiKeyInput.value = getStoredKey();
        updateBadge(getStoredKey());
        settingsModal.classList.add('open');
    });
    settingsClose.addEventListener('click', () => settingsModal.classList.remove('open'));
    settingsModal.addEventListener('click', e => {
        if (e.target === settingsModal) settingsModal.classList.remove('open');
    });
    saveKeyBtn.addEventListener('click', () => {
        const key = apiKeyInput.value.trim();
        key ? localStorage.setItem(STORAGE_KEY, key) : localStorage.removeItem(STORAGE_KEY);
        updateBadge(key);
        settingsModal.classList.remove('open');
    });
    clearKeyBtn.addEventListener('click', () => {
        localStorage.removeItem(STORAGE_KEY);
        apiKeyInput.value = '';
        updateBadge('');
    });
```

- [ ] **Step 4: Inject API key header into existing fetch calls in app.js**

Find the `fetch('/api/query', {` block (around line 219) and change:
```javascript
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: queryText })
        })
```
to:
```javascript
        const _key = getStoredKey();
        fetch('/api/query', {
            method: 'POST',
            headers: Object.assign(
                { 'Content-Type': 'application/json' },
                _key ? { 'X-OpenAI-Api-Key': _key } : {}
            ),
            body: JSON.stringify({ query: queryText })
        })
```

Find the `xhr.open('POST', '/api/ingest', true);` block (around line 151) and add after `xhr.open(...)`:
```javascript
        const _ingestKey = getStoredKey();
        if (_ingestKey) xhr.setRequestHeader('X-OpenAI-Api-Key', _ingestKey);
```

- [ ] **Step 5: Commit**

```bash
git add src/api/static/
git commit -m "feat(ui): add API key settings modal — OpenAI/OSS mode switching from browser"
```

---

### Task 13: Vercel Configuration

**Files:**
- Create: `api/index.py`
- Create: `vercel.json`
- Create: `requirements-vercel.txt`

- [ ] **Step 1: Create api/index.py (Vercel entrypoint)**

Create `C:\Users\Vivek\LexRag\api\index.py`:

```python
# Vercel Python serverless entrypoint.
# Vercel serves the FastAPI `app` object via its ASGI adapter.
import sys
from pathlib import Path

# Ensure project root is on the Python path so `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app  # noqa: F401 — Vercel discovers this symbol
```

- [ ] **Step 2: Create vercel.json**

Create `C:\Users\Vivek\LexRag\vercel.json`:

```json
{
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

- [ ] **Step 3: Create requirements-vercel.txt (no torch / sentence-transformers)**

Create `C:\Users\Vivek\LexRag\requirements-vercel.txt`:

```
fastapi==0.111.0
uvicorn==0.30.1
PyMuPDF==1.24.5
python-dotenv==1.0.1
pyyaml==6.0.1
requests==2.32.3
tqdm==4.66.4
numpy==1.26.4
faiss-cpu==1.8.0
openai==1.30.5
fastembed==0.3.1
python-multipart
httpx<0.28.0
```

Vercel reads `requirements.txt` by default. We need to swap it for the Vercel build. Two options:
- **Option A**: Rename `requirements.txt` → `requirements-local.txt`, rename `requirements-vercel.txt` → `requirements.txt` for the Vercel deployment. THEN after the Vercel deploy, restore locally.
- **Option B**: Configure Vercel to use a different file via `vercel.json`.

Go with **Option A** (simpler): just before deploying, rename `requirements-vercel.txt` to `requirements.txt` for the push (and keep the full one for local dev as `requirements-dev.txt`).

So rename existing `requirements.txt` → `requirements-dev.txt`:
```bash
Rename-Item C:\Users\Vivek\LexRag\requirements.txt C:\Users\Vivek\LexRag\requirements-dev.txt
Rename-Item C:\Users\Vivek\LexRag\requirements-vercel.txt C:\Users\Vivek\LexRag\requirements.txt
```

NOTE: For local dev, use: `pip install -r requirements-dev.txt`

- [ ] **Step 4: Commit**

```bash
git add api/ vercel.json requirements.txt requirements-dev.txt
git commit -m "feat(vercel): add serverless entrypoint, vercel.json routing, Vercel-safe requirements"
```

---

### Task 14: GitHub Push

- [ ] **Step 1: Final security audit — no real keys in git history**

```bash
cd C:\Users\Vivek\LexRag && git log --all --full-history -- .env
git grep -r "sk-proj" --cached
git grep -r "OPENAI_API_KEY.*=.*sk" --cached
```
Expected for all three: no output (empty = safe).

If `git log` shows `.env` was ever committed, you must purge it from history:
```bash
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch .env" --prune-empty --tag-name-filter cat -- --all
```

- [ ] **Step 2: Create GitHub repository**

Go to https://github.com/new
- Repository name: `LexRag`
- Description: `Advanced Legal RAG Sandbox — hierarchical retrieval, HyDE, CrossEncoder reranking, OpenAI/OSS provider strategy`
- Public: Yes (for public demo)
- Initialize: No (we already have commits)

- [ ] **Step 3: Push to GitHub**

```bash
cd C:\Users\Vivek\LexRag
git remote add origin https://github.com/Vivek1305/LexRag.git
git branch -M main
git push -u origin main
```
Expected: All commits pushed.

- [ ] **Step 4: Verify on GitHub**

Visit `https://github.com/Vivek1305/LexRag` and confirm:
- `.env` is NOT visible in the file tree
- `src/providers/` directory is present with all 4 files
- `data/indices/oss/` is present with the FAISS files
- `requirements.txt` is the Vercel-safe version (no torch line)

---

### Task 15: Vercel Deployment

- [ ] **Step 1: Install Vercel CLI**

```bash
npm install -g vercel
```
Expected: `added N packages` — Vercel CLI installed globally.

- [ ] **Step 2: Login to Vercel**

```bash
vercel login
```
Follow the browser authentication prompt.

- [ ] **Step 3: Deploy to Vercel**

```bash
cd C:\Users\Vivek\LexRag
vercel --prod
```
When prompted:
- "Set up and deploy?" → `Y`
- "Which scope?" → select your personal account
- "Link to existing project?" → `N`
- "Project name?" → `lexrag` (or `lex-rag`)
- "In which directory is your code located?" → `./`
- "Want to override settings?" → `N`

Wait for the build to complete (~2-5 minutes).

- [ ] **Step 4: (Optional) Add HF_TOKEN for better rate limits**

```bash
vercel env add HF_TOKEN production
```
Paste your HuggingFace token when prompted. This improves the free-tier HF Inference API response rate.

- [ ] **Step 5: Verify the live deployment**

Visit the URL printed by Vercel (e.g. `https://lexrag.vercel.app`).

Check:
1. Homepage loads (the LexRAG UI)
2. `GET https://lexrag.vercel.app/api/stats` returns JSON with document list
3. Click the ⚙️ Settings button → modal opens, mode shows "Open-Source (Free)"
4. Type a sample query (e.g. "What are the penalties under the IT Act?") → answer appears from HF Mistral
5. Paste an OpenAI key in Settings → re-run same query → mode shows "OpenAI (GPT-4o-mini)"

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Push to GitHub — Task 14
- ✅ No API keys in GitHub — Task 1 + Task 14 audit
- ✅ Strategy pattern for LLM — `src/providers/base.py` + factory + `Generator` accepting `LLMProvider`
- ✅ Strategy pattern for embeddings — `src/providers/base.py` + factory + `Embedder` accepting `EmbeddingProvider`
- ✅ OpenAI path (user provides key) — `OpenAILLMProvider` + `OpenAIEmbeddingProvider` (Task 3)
- ✅ Open-source fallback — `FastEmbedProvider` + `HuggingFaceLLMProvider` (Task 4)
- ✅ API key input UI — Settings modal with localStorage (Task 12)
- ✅ Deploy to Vercel — Task 13 + Task 15
- ✅ Working link — verified in Task 15 Step 5

**Placeholder scan:** No TBD/TODO/placeholder code found. All code blocks are complete.

**Type consistency:**
- `LLMProvider.generate(messages, max_tokens, temperature)` — used consistently in `Generator`, `LexRAGRetriever._hyde`
- `EmbeddingProvider.embed(texts) -> ndarray` — used consistently in `Embedder.embed_chunks`
- `Embedder(provider, cache_dir)` — constructor matches usage in `BaselineRetriever`, `LexRAGRetriever`
- `build_indices(cfg, embedding_provider)` — matches call in `main.py` and `build_oss_indices.py`
- `LexRAGRetriever.retrieve(query, llm=None)` — matches call in `main.py`
