# LexRAG — Project Handoff

> Current snapshot as of 2026-06-16. Hand this to a developer or AI agent to get up to speed fast.

## What it is

A **dual-pipeline RAG system for legal & medical documents** (Python/FastAPI backend + vanilla JS frontend). Its purpose is to *prove* that page-aware retrieval beats naive RAG by running both pipelines side-by-side in one UI.

It targets four specific failures of naive RAG:

1. **Page-boundary blindness** — naive chunking flattens pages (page = `-1`); LexRAG preserves real page numbers.
2. **Spanning questions** — answers that cross pages (e.g. 34→35); LexRAG expands ±1 neighbor page.
3. **Vocabulary mismatch** — uses **HyDE** (Hypothetical Document Embeddings) to bridge query↔document language.
4. **Precision** — **CrossEncoder re-ranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) as a second pass.

## Architecture (key file paths)

### Provider strategy pattern (`src/providers/`)
- `base.py` — abstract `EmbeddingProvider` + `LLMProvider`
- `openai_provider.py` — OpenAI embedding (`text-embedding-3-small`, 1536-dim) + `gpt-4o-mini`
- `hf_provider.py` — `FastEmbedProvider` (`BAAI/bge-small-en-v1.5`, 384-dim ONNX; falls back fastembed → sentence-transformers → HF Inference API) + `HuggingFaceLLMProvider` (Mistral-7B)
- `factory.py` — `get_providers()`; priority: explicit key > env `OPENAI_API_KEY` > OSS fallback

### Chunking (`src/data/chunking.py`)
- `NaiveChunker` — 512-char sliding window, sets page = `-1` (baseline)
- `PageAwareChunker` — chunks per page, preserves page numbers
- `HierarchicalChunker` — page-level chunks + sub-chunks linked via `parent_chunk_id`
- `StatuteChunker` — regex section detection + cross-reference DFS resolution

### Vector store & indexing
- `src/data/faiss_store.py` — FAISS `IndexFlatIP` (cosine via L2-norm), serialized as `.faiss` + `.pkl` pairs
- `src/data/embedder.py` — embedding with MD5-keyed `.npy` cache
- `src/data/indexing.py` — builds 3 indices per corpus: `baseline`, `lexrag_pages`, `lexrag_chunks`

### Retrieval (`src/retrieval/`)
- `baseline.py` — naive embed → FAISS → top-k
- `lexrag.py` — HyDE → embed hypothetical → page search → ±1 neighbor expansion → sub-chunk extraction → CrossEncoder rerank → (statute cross-ref DFS)
- `bm25_retriever.py` — offline keyword fallback (no embedding API needed)
- `reranker.py` — CrossEncoder re-ranker

### Generation (`src/generation/`)
- `generator.py` — LLM answer generation + citation extraction
- `prompts.py` — system/user prompt builders

## API & Frontend

### Backend — `src/api/main.py`
- `GET /api/stats` — corpus stats (docs, pages, cache size, chunk counts, document list)
- `POST /api/ingest` — PDF upload → process to JSON → rebuild indices
- `POST /api/query` — runs both pipelines; returns `baseline{}`, `lexrag{}`, `comparison{winner, reasons}`; reads optional `x-openai-api-key` header. Without a key → BM25 + extractive answers (no LLM).

### Serverless entry — `api/index.py`
Vercel wrapper importing the FastAPI app from `src.api.main`. Uses `/tmp` for writes on Vercel; project root locally.

### Frontend — `src/api/static/`
Vanilla HTML/CSS/JS (no framework). Glass-morphism dark UI, settings modal for browser-local API key (never logged), drag-drop PDF upload, live corpus stats, side-by-side Baseline-vs-LexRAG comparison + "winner diagnostics."
- `index.html` (~344 lines), `style.css` (~1067 lines), `app.js` (~405 lines)

## Data

3 docs currently ingested (`data/raw/manifest.json`):

| doc_id | domain | pages |
|---|---|---|
| Copyright_Office_Circular_01 | copyright_law | 10 |
| IRS_Circular_E_2024 | tax_law | 59 |
| it_act_2000 | legal (Ministry of Law & Justice, India) | 36 |

- `data/indices/` — FAISS indices (~18 MB)
- `data/processed/` — extracted JSON + embedding cache (`.npy`)
- `data/models/fastembed/` — `BAAI/bge-small-en-v1.5` ONNX cache (~80 MB)

## Configuration

### `configs/config.yaml`
- chunking: `chunk_size 512`, `overlap 50`, `min_chunk_size 100`
- embedding: `text-embedding-3-small`, 1536-dim, batch 100
- faiss: `IndexFlatIP`, `normalize_vectors: true`
- retrieval: `top_k 20`, `rerank_top_k 5`, `hyde_enabled true`, `neighbor_pages 1`
- generation: `gpt-4o-mini`, `temperature 0.0`, `max_tokens 1500`
- api: host `127.0.0.1`, port `8000`, rate_limit `60`/min

### `.env.example`
- `OPENAI_API_KEY` (optional — enables full semantic RAG)
- `HF_TOKEN` (optional — improves HF Inference rate limits)

## Tech stack

**Production (`requirements.txt`):** fastapi 0.111.0, uvicorn 0.30.1, faiss-cpu 1.8.0, openai 1.30.5, numpy 1.26.4, pyyaml, python-dotenv, python-multipart, httpx<0.28.0
**Dev (`requirements-dev.txt`):** + PyMuPDF 1.24.5, sentence-transformers 3.0.0, torch 2.3.0, transformers 4.41.2, fastembed 0.3.1, pytest 8.2.1

> `requirements.txt` is intentionally Vercel-safe (no torch). Heavy local deps live in `requirements-dev.txt`.

## Deployment (multi-target)

- **Vercel** — `vercel.json` (`@vercel/python`, all routes → `api/index.py`); `.vercel/` folder is linked
- **Render** — `render.yaml`, web service, `startCommand: uvicorn src.api.main:app`, healthcheck `/api/stats`
- **Docker** — `docker-compose.yml`: backend (FastAPI+FAISS, port 8000) + frontend (Nginx reverse proxy, 8080→80), persistent hf-cache volume; `Dockerfile.backend`, `Dockerfile.frontend`, `nginx.conf`
- **Local** — `run_server.py`

## Tests
`tests/test_chunking.py`, `tests/test_generation.py`, `tests/test_retrieval.py`

## Known limitations / TODO
- OSS Mistral-7B is lower quality than gpt-4o-mini
- HF free-tier rate limits (set `HF_TOKEN` to ease them)
- `StatuteChunker` is not used by default (only specific legal docs)
- No auth / multi-tenant — single-user sandbox

## Repo / git notes
- LexRag lives inside a monorepo at `C:\Users\Vivek` alongside `Agency-Web`, `ExaverAI`, `simpleDash`, `teacher`.
- GitHub: https://github.com/vivek314/LexRag (pushed via git subtree from the monorepo).
- ⚠️ The current branch `feat/page-load-ingestion` is **NOT** LexRag work — those commits belong to the `teacher/` sibling project (OCR/vision extraction, Next.js scaffold). In LexRag itself only `.gitignore` is modified.

## Reference docs in repo
- `README.md` — user-facing overview
- `PROJECT_KNOWLEDGE.md` (~67 KB) — extended internal knowledge base
