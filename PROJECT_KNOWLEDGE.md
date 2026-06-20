# LexRAG — Exhaustive Project Knowledge Document

**Generated:** 2026-04-02
**Project Root:** `c:\Users\Vivek\LexRag`
**Branch at time of audit:** `feature/autofix-agent`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture (High-Level Design)](#2-architecture-high-level-design)
3. [Data Models & Storage (Low-Level Design)](#3-data-models--storage-low-level-design)
4. [API Contracts & Interfaces](#4-api-contracts--interfaces)
5. [Core Business Logic](#5-core-business-logic)
6. [Code Patterns & Conventions](#6-code-patterns--conventions)
7. [Configuration & Environment](#7-configuration--environment)
8. [Testing Strategy](#8-testing-strategy)
9. [Known Issues, Tech Debt & Trade-offs](#9-known-issues-tech-debt--trade-offs)
10. [Roadmap & Open Questions](#10-roadmap--open-questions)

---

## 1. Project Overview

### 1.1 Project Name and Purpose

**LexRAG** is a Retrieval-Augmented Generation (RAG) system purpose-built for legal document question answering. It solves the core problem of finding precise, cited answers within large collections of legal PDFs — court judgments, regulatory filings, IRS publications, SEC documents — where a single legal argument can span multiple pages and naive text chunking loses critical context.

The project is simultaneously a working system and an interview study vehicle: nearly every design decision is annotated with "INTERVIEW NOTE:" comments explaining the trade-off in technical detail (e.g., why `text-embedding-3-small` over `ada-002`, why `IndexFlatIP` over `IVFFlat` at this corpus size, why `temperature=0` for legal Q&A).

### 1.2 Problem Statement

Standard RAG systems fail on legal documents for three reasons:
1. **Page-boundary blindness** — naive fixed-size chunking splits mid-paragraph across pages; the NaiveChunker in this codebase deliberately demonstrates this failure mode (all chunks get `page_number=-1`).
2. **Query-document vocabulary mismatch** — a user query "what are penalties for late deposits" uses different vocabulary than the IRS passage "failure-to-deposit penalty… 2% for 1-5 days late". HyDE (Hypothetical Document Embedding) bridges this gap.
3. **Single-granularity retrieval** — a page-level match may span too much text; a sub-chunk may be too narrow. The hierarchical two-index design solves this.

### 1.3 Target Users / Consumers

- **Primary:** The developer/researcher themselves (Vivek), building and benchmarking the system as a portfolio/interview project.
- **Secondary:** Interviewers at AI/ML engineering roles evaluating deep understanding of RAG systems, vector search, and LLM engineering.
- **Implied future users:** Legal professionals, compliance teams, or anyone needing grounded Q&A over legal corpora.

### 1.4 Current Stage

**Prototype / working MVP with benchmark harness.** All core pipeline phases are implemented and operational:
- Data ingestion pipeline (download → PDF parse → chunk → embed → index) is complete.
- Two retrievers are built and functional: `BaselineRetriever` and `LexRAGRetriever`.
- Generator with citation extraction is complete.
- Full evaluation harness (retrieval metrics + RAGAS-style LLM judge) is implemented.
- **No REST API is implemented yet** — `src/api/` is an empty package (`__init__.py` only).
- No frontend, no Docker, no CI/CD pipeline.
- Redis caching is defined in config but not yet implemented in any module.

### 1.5 Tech Stack Summary

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12.10 |
| PDF parsing | PyMuPDF (`fitz`) | 1.27.2.2 |
| Embedding | OpenAI `text-embedding-3-small` | via `openai` SDK 1.x |
| Vector search | FAISS (`faiss-cpu`) | 1.13.2 |
| Re-ranking | `sentence-transformers` CrossEncoder | 5.3.0 |
| Cross-encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace |
| LLM generation | OpenAI `gpt-4o-mini` | via `openai` SDK |
| HTTP downloads | `requests` | 2.33.0 |
| Config | PyYAML | 6.0.3 |
| Env vars | `python-dotenv` | 1.2.2 |
| Numerical | NumPy | 2.4.3 |
| Progress bars | `tqdm` | 4.67.3 |
| API framework (planned) | FastAPI | 0.135.2 |
| API server (planned) | Uvicorn | 0.42.0 |
| Testing | pytest | 9.0.2 |
| Caching (planned) | Redis | config defined, not wired |
| ML framework (installed) | PyTorch | 2.11.0 |
| Transformers | HuggingFace `transformers` | 5.3.0 |

Also installed but **not actively used in core code**: `langchain`, `langchain-openai`, `langgraph`, `streamlit`, `pandas`, `scikit-learn`, `SQLAlchemy`.

### 1.6 Repository Structure (2–3 levels deep)

```
LexRag/
├── .claude/
│   └── settings.local.json          # Claude Code permission allowlist
├── .env                             # OPENAI_API_KEY (gitignored)
├── .gitignore                       # Excludes venv, data/raw, data/processed, data/indices, logs
├── configs/
│   └── config.yaml                  # Single source of all hyperparameters
├── data/
│   ├── raw/                         # Downloaded PDFs (gitignored)
│   │   ├── IRS_Circular_E_2024.pdf
│   │   └── Copyright_Office_Circular_01.pdf
│   ├── processed/                   # Parsed JSON per doc + embedding cache (gitignored)
│   │   ├── IRS_Circular_E_2024.json
│   │   ├── Copyright_Office_Circular_01.json
│   │   └── embedding_cache/         # 1,557 .npy files (MD5-keyed per chunk text)
│   └── indices/                     # FAISS .faiss + .pkl files (gitignored)
│       ├── baseline.faiss           # 708 vectors, IndexFlatIP, dim=1536
│       ├── baseline_chunks.pkl      # Parallel chunk list for baseline
│       ├── lexrag_pages.faiss       # 69 page-level vectors
│       ├── lexrag_pages_chunks.pkl
│       ├── lexrag_chunks.faiss      # 743 sub-chunk vectors
│       └── lexrag_chunks_chunks.pkl
├── docs/
│   └── interview_questions.txt      # 25 interview Q&A prompts (partially filled)
├── logs/                            # Empty (log files gitignored)
├── src/
│   ├── __init__.py
│   ├── api/
│   │   └── __init__.py              # Empty — API not yet implemented
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py            # HTTP download + manifest writer
│   │   ├── processor.py             # PyMuPDF PDF-to-JSON pipeline
│   │   ├── chunking.py              # Three chunking strategies
│   │   ├── embedder.py              # OpenAI embeddings with file cache
│   │   ├── faiss_store.py           # FAISS wrapper (add/search/save/load)
│   │   └── indexing.py              # Orchestrates build of baseline + LexRAG indices
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── baseline.py              # Naive bi-encoder retriever
│   │   ├── lexrag.py                # HyDE + hierarchical + rerank retriever
│   │   └── reranker.py              # CrossEncoder re-ranker wrapper
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py             # GPT-4o-mini call + citation extraction
│   │   └── prompts.py               # System prompt + context formatter
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py               # Precision@K, Recall@K, MRR, citation accuracy, LLM judge
│       ├── benchmark.py             # Baseline vs. LexRAG retrieval comparison script
│       └── ragas_eval.py            # RAGAS-style end-to-end evaluation
├── tests/
│   ├── test_chunking.py             # Chunker correctness (page info, parent-child links)
│   ├── test_retrieval.py            # Side-by-side retrieval smoke test
│   └── test_generation.py           # End-to-end generation smoke test
└── venv/                            # Virtual environment (gitignored)
```

### 1.7 How to Run Locally

**Prerequisites:**
- Python 3.12.x
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini`

**Setup:**
```bash
# 1. Create and activate virtualenv
python -m venv venv
# Windows:
venv\Scripts\activate
# Unix/Mac:
source venv/bin/activate

# 2. Install all dependencies (no requirements.txt exists — install from venv freeze or manually)
pip install openai python-dotenv PyMuPDF pyyaml requests tqdm numpy faiss-cpu \
            sentence-transformers torch transformers fastapi uvicorn pytest

# 3. Set up environment variable
# Create .env in project root:
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Download the legal corpus (optional — pre-downloaded PDFs already in data/raw/)
python src/data/downloader.py --config configs/config.yaml

# 5. Process PDFs into JSON
python src/data/processor.py --input data/raw --output data/processed --config configs/config.yaml

# 6. Build FAISS indices (requires OpenAI API for embeddings)
python src/data/indexing.py

# 7. Run retrieval smoke test
python tests/test_retrieval.py

# 8. Run generation smoke test
python tests/test_generation.py

# 9. Run retrieval benchmark
python src/evaluation/benchmark.py

# 10. Run RAGAS end-to-end evaluation
python src/evaluation/ragas_eval.py
```

**Note:** There is no `requirements.txt` or `pyproject.toml` in the repository. Dependencies must be inferred from imports or installed from venv.

### 1.8 How to Run Tests, Lint, Build, Deploy

```bash
# Tests (pytest-style, but all test files use if __name__ == '__main__': pattern)
python tests/test_chunking.py       # Unit-level, no OpenAI needed
python tests/test_retrieval.py      # Requires loaded indices + OpenAI key
python tests/test_generation.py     # Requires loaded indices + OpenAI key

# pytest runner (tests are not pytest-idiomatic — no pytest fixtures/assert syntax)
pytest tests/test_chunking.py -v

# Linting: No linter configured (.eslintrc, .flake8, .pylintrc, ruff.toml absent)

# Build: N/A — no packaging configured

# Deploy: N/A — no Dockerfile, no CI/CD pipeline
```

---

## 2. Architecture (High-Level Design)

### 2.1 System Architecture Description

LexRAG is a **pipeline-style RAG system** with offline index construction and online query serving. The components connect as follows:

```
[OFFLINE PIPELINE — run once]
PDF URLs
  └─→ downloader.py (HTTP + retry)
         └─→ data/raw/*.pdf
               └─→ processor.py (PyMuPDF page extraction + cleaning)
                     └─→ data/processed/*.json  (Document objects)
                           └─→ chunking.py  (NaiveChunker / PageAwareChunker / HierarchicalChunker)
                                 └─→ embedder.py (OpenAI API, MD5 file cache)
                                       └─→ faiss_store.py (IndexFlatIP, L2-normalized)
                                             └─→ data/indices/*.faiss + *.pkl

[ONLINE QUERY PIPELINE — per query]
User Query
  └─→ [BASELINE PATH]
  │     └─→ BaselineRetriever.retrieve(query)
  │           ├─ embed query → FaissStore.search() → top_k chunks
  │           └─→ [(Chunk, score), ...]
  │
  └─→ [LEXRAG PATH]
        └─→ LexRAGRetriever.retrieve(query)
              ├─ _hyde(query)  → OpenAI gpt-4o-mini → hypothetical_answer
              ├─ embed(hypothetical_answer) → page_store.search() → top_k pages
              ├─ _expand_pages(page_numbers) → neighboring pages
              ├─ _get_sub_chunks(pages) → candidate sub-chunks
              ├─ Reranker.rerank(query, candidates) → CrossEncoder scored
              └─→ [(Chunk, score), ...]  (top rerank_top_k=5)

[GENERATION — follows either retrieval path]
[(Chunk, score), ...]
  └─→ prompts.format_context() → numbered [SOURCE X] blocks
        └─→ build_prompt() → system + user messages
              └─→ Generator.generate() → OpenAI gpt-4o-mini
                    └─→ {answer, citations, confidence, generation_time_ms, model, chunks_used}

[EVALUATION]
9 benchmark queries × (baseline_results + lexrag_results)
  └─→ metrics.py: Precision@K, Recall@K, MRR, citation accuracy
  └─→ ragas_eval.py: context_precision, context_recall, faithfulness, answer_relevance (LLM judge)
```

### 2.2 Architectural Pattern

**Offline-indexed RAG with dual-path comparison.** Key pattern choices:

- **Strategy Pattern** for chunking: `ChunkingStrategy` ABC → `NaiveChunker`, `PageAwareChunker`, `HierarchicalChunker`. Factory function `get_chunker(strategy)`.
- **Repository Pattern** for vector storage: `FaissStore` encapsulates all FAISS operations (add, search, persist, load).
- **Pipeline Pattern** for data ingestion: each step is an independent script with CLI entry points (`downloader.py`, `processor.py`, `indexing.py`).
- **Facade Pattern** for retrieval: `BaselineRetriever` and `LexRAGRetriever` each expose a single `retrieve(query: str) -> list[tuple[Chunk, float]]` interface despite very different internal logic.

### 2.3 External Services, APIs, and Integrations

| Service | What it provides | Where used | Auth method |
|---|---|---|---|
| **OpenAI API** (`text-embedding-3-small`) | 1536-dim vector embeddings | `src/data/embedder.py` — `Embedder.embed_chunks()` | `OPENAI_API_KEY` env var |
| **OpenAI API** (`gpt-4o-mini`) | HyDE hypothetical answer generation | `src/retrieval/lexrag.py` — `LexRAGRetriever._hyde()` | `OPENAI_API_KEY` env var |
| **OpenAI API** (`gpt-4o-mini`) | Final answer generation with citations | `src/generation/generator.py` — `Generator.generate()` | `OPENAI_API_KEY` env var |
| **OpenAI API** (`gpt-4o-mini`) | LLM judge evaluation (RAGAS metrics) | `src/evaluation/metrics.py` — `llm_judge()`, `src/evaluation/ragas_eval.py` — `RAGASEvaluator._judge()` | `OPENAI_API_KEY` env var |
| **HuggingFace Hub** | Downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` on first run | `src/retrieval/reranker.py` — `Reranker.__init__()` via `CrossEncoder(model_name)` | No auth required (public model) |
| **Indian Supreme Court** (`main.sci.gov.in`) | Source legal PDFs | `src/data/downloader.py` `LEGAL_PDF_URLS` | No auth, User-Agent header |
| **SEC EDGAR** (`www.sec.gov`) | Source legal PDFs | `src/data/downloader.py` `FALLBACK_PDF_URLS` | No auth |
| **US DOJ, FTC, IRS, FDIC, etc.** | Source legal PDFs | `src/data/downloader.py` `FALLBACK_PDF_URLS` | No auth |
| **Redis** (planned) | Embedding + response caching | `configs/config.yaml` — `cache.*` section, not wired in code | `localhost:6379` |

### 2.4 Data Flow: Key User Flows

#### Flow A: Full Corpus Ingestion (one-time setup)

1. **`downloader.py` → `download_corpus(config)`**
   - Iterates `LEGAL_PDF_URLS` + `FALLBACK_PDF_URLS` (up to `max_docs=30`)
   - HTTP GET with `stream=True`, validates `%PDF` magic bytes
   - Saves to `data/raw/{slug}.pdf`; idempotent (skips if file ≥ 1KB exists)
   - Writes `data/raw/manifest.json` with `DocMetadata` (doc_id, title, source, domain, date, num_pages, local_path)

2. **`processor.py` → `process_corpus(raw_dir, output_dir, cfg)`**
   - For each PDF: opens with `fitz.open()`, extracts pages
   - Per page: `extract_page()` strips header/footer zones (top/bottom 8%), joins text block spans, detects tables via `\d+\s{3,}\d+` pattern
   - Skips: PDFs < 2 pages, PDFs > 200 pages, blank pages (< 50 chars)
   - Outputs: `data/processed/{doc_id}.json` as serialized `Document` object; incremental

3. **`indexing.py` → `build_indices(cfg)`**
   - Loads all JSON from `data/processed/`
   - **Baseline index:** `NaiveChunker(512, 50)` → `Embedder.embed_chunks()` → `FaissStore.add()` → `save("baseline")`
   - **LexRAG page index:** `HierarchicalChunker(512, 50)` → filter `level==page` → embed → `save("lexrag_pages")`
   - **LexRAG sub-chunk index:** same hierarchical chunks → filter `level==chunk` → embed → `save("lexrag_chunks")`
   - Each call to `Embedder.embed_chunks()` checks MD5-keyed `.npy` files in `data/processed/embedding_cache/` before calling OpenAI API

#### Flow B: LexRAG Query (online)

1. Query arrives as `str`
2. `LexRAGRetriever._hyde(query)` → OpenAI gpt-4o-mini (temp=0, max_tokens=200) → hypothetical passage
3. `Embedder.embed_chunks([synthetic_chunk])` → 1536-dim float32 vector (cache-checked first)
4. `FaissStore.search(query_vector, top_k=20)` on page-level index → L2-normalize → `IndexFlatIP.search()` → 20 `(Chunk, score)` pairs
5. Extract page numbers + doc_ids from results; expand ±1 neighbor pages
6. Linear scan of `chunk_store.chunks` list for matching page/doc pairs → candidate sub-chunks
7. `Reranker.rerank(query, candidates)` → `CrossEncoder.predict([(query, text), ...])` → sort by score → top 5
8. `Generator.generate(query, reranked_chunks)` → `build_prompt()` → OpenAI gpt-4o-mini → answer
9. Regex extract `[SOURCE N]` from answer → build citations list
10. Keyword-detect uncertainty phrases → `confidence = "low"` or `"high"`
11. Return `{answer, citations, confidence, generation_time_ms, model, chunks_used}`

#### Flow C: Baseline Query (online)

1. Query → `BaselineRetriever.retrieve(query)` → embed raw query (no HyDE) → `FaissStore.search()` on `baseline` index → top 20 chunks directly
2. Optionally feed into `Generator.generate()` (done in `test_generation.py`)

### 2.5 Communication Patterns

- All current communication is **synchronous in-process Python function calls**.
- No HTTP API server is running. No inter-process messaging. No async.
- OpenAI calls are synchronous (`client.chat.completions.create`, `client.embeddings.create`).
- `time.sleep(0.1)` is inserted after each embedding batch to avoid OpenAI rate limits.

### 2.6 Authentication and Authorization Model

**No authentication model exists.** The system is a single-user local research tool.

- The only secret is `OPENAI_API_KEY`, loaded from `.env` via `python-dotenv` in `embedder.py`, `lexrag.py`, `generator.py`, `ragas_eval.py`.
- `.env` is listed in `.gitignore`.
- No user sessions, tokens, or RBAC.

### 2.7 Caching Strategy

Two distinct caching layers are present or planned:

| Layer | What is cached | Where | Key | Format | TTL | Status |
|---|---|---|---|---|---|---|
| **Embedding file cache** | OpenAI embedding vectors | `data/processed/embedding_cache/` | `MD5(chunk_text).hexdigest()` → `{hash}.npy` | NumPy `.npy` float32 array | Permanent (no expiry) | **Implemented** |
| **Redis embedding cache** | Embeddings (runtime) | Redis `localhost:6379 db=0` | Not defined in code | Not defined | `embedding_ttl: 86400` (24h) | **Config only — not wired** |
| **Redis response cache** | Full Q&A responses | Redis `localhost:6379 db=0` | Not defined in code | Not defined | `response_ttl: 3600` (1h) | **Config only — not wired** |

The file cache (`embedding_cache/`) contains 1,557 `.npy` files — one per unique chunk text seen during the two `indexing.py` runs (baseline + lexrag corpus).

**Cache invalidation:** The file cache has no invalidation mechanism. If chunk text changes (e.g., re-chunking with different `chunk_size`), old `.npy` files accumulate and are never cleaned. The MD5 key means content-identical chunks across documents always hit cache correctly.

**Collision risk:** The MD5 key uses 128-bit hashes. The probability of a collision across ~1,557 entries is negligible (~4.4×10⁻³¹) but the code has no collision detection or fallback (noted as interview question #25).

### 2.8 Database Architecture

**No relational database.** All storage is flat files:

| Store | Format | Location | Contents |
|---|---|---|---|
| Manifest | JSON | `data/raw/manifest.json` | List of `DocMetadata` dicts |
| Processed docs | JSON (one per doc) | `data/processed/{doc_id}.json` | `Document` object with `pages: [PageContent]` |
| Embedding cache | NumPy `.npy` | `data/processed/embedding_cache/{md5}.npy` | Single 1536-dim float32 vector |
| FAISS index | Binary `.faiss` | `data/indices/{name}.faiss` | FAISS `IndexFlatIP` binary |
| Chunk list | Pickle `.pkl` | `data/indices/{name}_chunks.pkl` | `list[Chunk]` (parallel to FAISS rows) |

### 2.9 Infrastructure / Deployment Topology

**Local development only.** No cloud deployment, no Docker, no CI/CD.

- Runs on Windows 11 (from `.env` path patterns `data\raw\...`), but paths use `pathlib.Path` which normalizes across OS.
- HuggingFace cross-encoder model is downloaded to local HuggingFace cache (`~/.cache/huggingface/`) on first run.
- All FAISS index files are stored on local disk (not in cloud object storage).

---

## 3. Data Models & Storage (Low-Level Design)

### 3.1 Core Data Classes

All data classes use Python `@dataclass` from the standard library.

#### `DocMetadata` — `src/data/downloader.py`

| Field | Type | Description |
|---|---|---|
| `doc_id` | `str` | Slugified title (max 60 chars), e.g., `IRS_Circular_E_2024` |
| `title` | `str` | Human-readable title from `LEGAL_PDF_URLS` config |
| `source` | `str` | Original URL the PDF was downloaded from |
| `domain` | `str` | Corpus domain: `indian_kanoon`, `sec_edgar`, `us_irs`, `us_doj`, `us_ftc`, `us_fdic`, `us_fed`, `us_occ`, `us_copyright`, `us_dol`, `us_hud`, `us_eeoc`, `cornell_law`, `world_bank`, `us_court` |
| `date` | `str` | Publication date (`YYYY-MM-DD` or `"unknown"`) |
| `num_pages` | `int` | Page count; filled by `processor.py` after PDF parse (`0` when from downloader) |
| `local_path` | `str` | Relative path like `data/raw/IRS_Circular_E_2024.pdf` |

Serialized to `data/raw/manifest.json` as JSON array via `dataclasses.asdict()`.

#### `PageContent` — `src/data/processor.py`

| Field | Type | Description |
|---|---|---|
| `page_number` | `int` | 1-indexed page number (fitz is 0-indexed; +1 applied at extraction) |
| `text` | `str` | Cleaned text with headers/footers stripped; newline-joined lines |
| `char_count` | `int` | `len(text)` |
| `has_table` | `bool` | True if ≥3 lines matched `\d+\s{3,}\d+` pattern |

Serialized as part of `Document.pages` list in JSON.

#### `Document` — `src/data/processor.py`

| Field | Type | Description |
|---|---|---|
| `doc_id` | `str` | Matches `DocMetadata.doc_id` |
| `title` | `str` | Matches `DocMetadata.title` |
| `source` | `str` | Source URL |
| `domain` | `str` | Corpus domain |
| `date` | `str` | Publication date |
| `local_path` | `str` | Path to raw PDF |
| `num_pages` | `int` | Number of readable pages extracted |
| `pages` | `list[PageContent]` | Ordered list of extracted pages |
| `metadata` | `dict` | Arbitrary extra metadata (defaults to `{}`) |

Serialized to `data/processed/{doc_id}.json` via `dataclasses.asdict()`.

**Known data in corpus:**
- `IRS_Circular_E_2024`: 59 pages
- `Copyright_Office_Circular_01`: 10 pages

#### `Chunk` — `src/data/chunking.py`

| Field | Type | Constraints | Description |
|---|---|---|---|
| `chunk_id` | `str` | Unique per corpus | Format varies by strategy: `{doc_id}_c{index}` (naive), `{doc_id}_p{page}_c{index}` (page_aware), `{doc_id}_page_{page}` or `{doc_id}_page_{page}_c{index}` (hierarchical) |
| `doc_id` | `str` | Foreign key to `Document.doc_id` | Source document identifier |
| `text` | `str` | `chunk_size` chars max | The chunk text content |
| `page_number` | `int` | -1 for naive chunks | 1-indexed page number; -1 means page info was lost |
| `chunk_index` | `int` | Sequential | Position in the document's chunk sequence |
| `char_count` | `int` | `len(text)` | Character count |
| `parent_chunk_id` | `Optional[str]` | References page-level chunk_id | Used in `HierarchicalChunker` for the parent-child B-tree link |
| `metadata` | `dict` | Default `{}` | `{"level": "page"}` or `{"level": "chunk"}` for hierarchical; empty for naive/page_aware |

**Persisted as:** `list[Chunk]` pickled to `data/indices/{name}_chunks.pkl`. The FAISS index row `i` corresponds to `chunks[i]`.

### 3.2 Entity Relationships

```
DocMetadata (1) ──────────────── (0+) (implied — same doc_id)
Document (1) ──────────────────── (N) PageContent
Document (1) ──────────────────── (N) Chunk  [via chunking strategies]
Chunk (child, level="chunk") ──── (1) Chunk (parent, level="page")  [hierarchical only, via parent_chunk_id]
FaissStore.index (row i) ─────── (1) FaissStore.chunks[i]  [parallel list structure]
```

### 3.3 FAISS Index Structure

Three named indices are persisted:

| Index Name | File | Vectors | Dim | Type | Contents |
|---|---|---|---|---|---|
| `baseline` | `baseline.faiss` + `baseline_chunks.pkl` | 708 | 1536 | `IndexFlatIP` | `NaiveChunker` output — page_number=-1 for all |
| `lexrag_pages` | `lexrag_pages.faiss` + `lexrag_pages_chunks.pkl` | 69 | 1536 | `IndexFlatIP` | `HierarchicalChunker` page-level chunks (`metadata.level == "page"`) |
| `lexrag_chunks` | `lexrag_chunks.faiss` + `lexrag_chunks_chunks.pkl` | 743 | 1536 | `IndexFlatIP` | `HierarchicalChunker` sub-chunks (`metadata.level == "chunk"`) |

All vectors are L2-normalized before insertion (because `normalize_vectors: true` in config), making `IndexFlatIP` equivalent to cosine similarity.

### 3.4 Embedding Cache

- **Location:** `data/processed/embedding_cache/`
- **File count:** 1,557 `.npy` files
- **Key scheme:** `MD5(chunk_text.encode()).hexdigest()` → filename
- **Value:** Single NumPy array, shape `(1536,)`, dtype `float32`
- **No TTL/expiry**; files persist indefinitely

### 3.5 Processed Documents (JSON Schema)

```json
{
  "doc_id": "IRS_Circular_E_2024",
  "title": "IRS_Circular_E_2024",
  "source": "",
  "domain": "unknown",
  "date": "unknown",
  "local_path": "data\\raw\\IRS_Circular_E_2024.pdf",
  "num_pages": 59,
  "pages": [
    {
      "page_number": 1,
      "text": "...",
      "char_count": 3366,
      "has_table": false
    }
  ],
  "metadata": {}
}
```

### 3.6 Data Validation Rules

- PDFs with < 2 pages are skipped (`min_pages: 2` in config)
- PDFs with > 200 pages are truncated to 200 pages (`max_pages: 200`)
- Pages with < 50 chars are skipped as blank
- Chunks with len < 1 char are skipped (checked as `if len(text) > 0`)
- `min_chunk_size: 100` is defined in config but is **not enforced in any chunking code** — this is a tech debt item
- PDF magic bytes `%PDF` are verified before saving; non-PDF responses are discarded
- Files ≤ 1024 bytes are considered invalid downloads and re-downloaded

---

## 4. API Contracts & Interfaces

### 4.1 REST API

**No REST API is implemented.** `src/api/__init__.py` is empty. `configs/config.yaml` defines `api.host`, `api.port`, `api.rate_limit`, `api.max_concurrent` but these are aspirational.

FastAPI 0.135.2 and Uvicorn 0.42.0 are installed, indicating the intent to build a REST API. Based on config and codebase, the expected API would be:

```
POST /query
  Body: { "query": "string", "retriever": "lexrag" | "baseline" }
  Response: {
    "answer": "string",
    "citations": [{"source_num": int, "doc_id": "string", "page_number": int, "text_snippet": "string"}],
    "confidence": "high" | "low",
    "generation_time_ms": int,
    "model": "string",
    "chunks_used": int
  }

GET /health
  Response: { "status": "ok" }
```

### 4.2 Python Module Interfaces

The primary programmatic interfaces are:

#### `BaselineRetriever.retrieve(query: str) -> list[tuple[Chunk, float]]`
- Loads `baseline` FAISS index on init
- Returns top `top_k=20` chunks with cosine similarity scores

#### `LexRAGRetriever.retrieve(query: str) -> list[tuple[Chunk, float]]`
- Loads `lexrag_pages` and `lexrag_chunks` FAISS indices on init
- Returns top `rerank_top_k=5` chunks with CrossEncoder scores

#### `Generator.generate(query: str, chunks: list[tuple[Chunk, float]]) -> dict`
- Returns: `{answer: str, citations: list[dict], confidence: str, generation_time_ms: int, model: str, chunks_used: int}`

#### `Embedder.embed_chunks(chunks: list[Chunk]) -> np.ndarray`
- Returns: shape `(N, 1536)` float32 array

#### `FaissStore.search(query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]`
- `query_vector` shape: `(1536,)` — reshaped internally to `(1, 1536)`
- Returns: `[(Chunk, float_score), ...]` sorted by descending score

#### `Reranker.rerank(query: str, candidates: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]`
- Returns top `rerank_top_k=5` sorted by CrossEncoder score

#### `get_chunker(strategy: str, chunk_size: int = 512, overlap: int = 50) -> ChunkingStrategy`
- Valid strategies: `"naive"`, `"page_aware"`, `"hierarchical"`
- Raises `ValueError` on unknown strategy

### 4.3 Rate Limiting, Pagination

- No rate limiting in current code (config defines `rate_limit: 60` for the unimplemented API)
- No pagination anywhere
- OpenAI rate limit mitigation: `time.sleep(0.1)` after each embedding batch call in `embedder.py`

### 4.4 Error Response Format

No standardized error format. Errors surface as:
- Python exceptions propagated to caller
- `logger.warning()` / `logger.error()` with return of `None`, `False`, or `[]`

---

## 5. Core Business Logic

### 5.1 Module: `src/data/downloader.py`

**Purpose:** Download a reproducible legal PDF corpus from public government and legal databases.

**Key algorithm — `download_pdf()`:**
1. Sets `User-Agent: LexRAG-Research-Bot/1.0` to identify the bot to servers
2. Uses `stream=True` to avoid loading large PDFs into memory
3. Validates PDF magic bytes (`%PDF`) post-download; deletes file if invalid
4. HTTP 404 → immediate abort (no retry); HTTP 429/503 → retry with backoff `2.0^attempt`
5. Max 3 retries; each subsequent sleep = `2.0^attempt` seconds (2s, 4s, 8s)

**Idempotency:** `download_corpus()` skips files where `dest.exists() and dest.stat().st_size > 1024`.

**Corpus structure:** 5 primary URLs (`LEGAL_PDF_URLS`) covering Indian SC judgments + SEC filings; 21 fallback URLs (`FALLBACK_PDF_URLS`) covering IRS, DOJ, FTC, FDIC, SEC, Fed, OCC, DOL, HUD, EEOC, Copyright Office.

**Corpus cap:** `max_docs: 30` in config; actual downloaded = 2 PDFs currently present.

### 5.2 Module: `src/data/processor.py`

**Purpose:** Extract structured, cleaned text from legal PDFs with page-boundary preservation.

**Key algorithm — `extract_page(page: fitz.Page) -> PageContent`:**
1. Calls `page.get_text("dict")` for block-level structured extraction
2. Calculates `header_cutoff = page_height * 0.08` and `footer_cutoff = page_height * 0.92`
3. Skips blocks entirely within the header or footer zone (block_y1 ≤ header_cutoff OR block_y0 ≥ footer_cutoff)
4. Only processes `block["type"] == 0` (text blocks; skips type 1 = images)
5. Joins spans within each line with a space; joins lines with newlines
6. Detects table lines via `re.search(r"\d+\s{3,}\d+", line_text)` — if ≥ 3 such lines exist on a page, `has_table = True`
7. Converts 0-indexed fitz page numbers to 1-indexed (`page.number + 1`)

**Edge cases:**
- Pages with < 50 chars after extraction are skipped as blank
- Incremental processing: if `data/processed/{doc_id}.json` exists, the doc is loaded from JSON, not re-processed

### 5.3 Module: `src/data/chunking.py`

**Purpose:** Split documents into retrievable units using three strategies with distinct trade-offs.

**`NaiveChunker.chunk(doc: Document) -> list[Chunk]`:**
- Concatenates ALL page texts into one string (page boundaries lost)
- Slides a 512-char window with 50-char overlap: `start += chunk_size - overlap`
- All chunks get `page_number=-1` — citation impossible
- Produces 708 chunks for 2-document corpus

**`PageAwareChunker.chunk(doc: Document) -> list[Chunk]`:**
- Chunking is done per-page, never crossing page boundaries
- Same 512/50 sliding window, but applied within each `PageContent.text`
- All chunks have valid `page_number` from the source page
- Produces fewer total chars from last page per document (no cross-page overlap)

**`HierarchicalChunker.chunk(doc: Document) -> list[Chunk]`:**
- Two passes per page:
  - **Pass 1 (index node):** Creates one `Chunk` per page, `metadata={"level": "page"}`, `parent_chunk_id=None`, text = full page text
  - **Pass 2 (leaf nodes):** Applies 512/50 sliding window to page text; each sub-chunk sets `parent_chunk_id = page_chunk_id` and `metadata={"level": "chunk"}`
- Produces 69 page chunks + 743 sub-chunks for current 2-document corpus
- The parent-child link enables "retrieve by page, answer with sub-chunk" pattern

**`get_chunker(strategy, chunk_size, overlap)`:** Factory function returning the appropriate strategy instance.

**Known design gap:** `config.yaml` defines `min_chunk_size: 100` and `section_patterns` for section-aware splitting, but neither is implemented in any chunker.

### 5.4 Module: `src/data/embedder.py`

**Purpose:** Convert text chunks to 1536-dim vectors via OpenAI API, with MD5 file cache.

**`Embedder.embed_chunks(chunks: list[Chunk]) -> np.ndarray`:**
1. Iterates chunks in batches of `batch_size=100`
2. For each chunk: check `data/processed/embedding_cache/{md5}.npy`; if hit, load and use
3. Collect misses; call `client.embeddings.create(model="text-embedding-3-small", input=texts_to_embed)` once per batch
4. Convert each `response.data[i].embedding` to `np.array(dtype=np.float32)`
5. Save each new vector to cache via `np.save()`
6. `time.sleep(0.1)` after any API call
7. Return `np.vstack(vectors)` — shape `(N, 1536)`

**Note:** Cache lookup is per-chunk inside the batch loop. For a batch of 100, it makes up to 100 individual file reads before deciding which subset needs API calls.

### 5.5 Module: `src/data/faiss_store.py`

**Purpose:** Thin wrapper around FAISS for add/search/persist/load operations.

**`FaissStore.add(chunks, vectors)`:**
- `faiss.normalize_L2(vectors)` in-place (modifies input array!)
- Creates `IndexFlatIP(dim=1536)` on first call if `self.index is None`
- Also supports `IndexFlatL2` and `index_factory()` for other types
- `self.index.add(vectors)` — appends to index
- `self.chunks.extend(chunks)` — maintains parallel list

**`FaissStore.search(query_vector, top_k)`:**
- Reshapes to `(1, 1536)`, normalizes in-place
- `self.index.search(q, top_k)` → `scores[0]` and `indices[0]`
- Filters `idx == -1` (FAISS sentinel for "fewer results than requested")
- Returns `[(self.chunks[idx], float(score)), ...]`

**`FaissStore.save(name)` / `FaissStore.load(name)`:**
- Save: `faiss.write_index(index, f"{name}.faiss")` + `pickle.dump(chunks, f"{name}_chunks.pkl")`
- Load: reverse; loaded index is ready to search immediately

### 5.6 Module: `src/retrieval/lexrag.py`

**Purpose:** Implement the full LexRAG retrieval pipeline: HyDE → page search → neighbor expansion → sub-chunk filtering → CrossEncoder re-rank.

**`LexRAGRetriever._hyde(query) -> str`:**
- Prompt: "Write a hypothetical document passage that answers: {query}"
- System: "You are a legal document expert. Write a concise hypothetical passage... Be factual and specific."
- `temperature=0.0`, `max_tokens=200`
- Returns the generated text to use as the embedding query instead of the raw question

**`LexRAGRetriever._expand_pages(page_numbers, doc_id) -> list[int]`:**
- For each retrieved page, adds `range(page - neighbor_pages, page + neighbor_pages + 1)`
- `neighbor_pages=1` → each retrieved page expands to {page-1, page, page+1}
- Filters `neighbor > 0` to avoid negative page numbers
- Note: `doc_id` parameter is accepted but unused — expansion is page-number-only, not per-doc

**`LexRAGRetriever._get_sub_chunks(page_numbers, doc_ids) -> list[tuple[Chunk, float]]`:**
- Linear O(N) scan over `self.chunk_store.chunks` (743 chunks in current corpus)
- Matches `chunk.page_number in allowed_pages AND chunk.doc_id in allowed_docs`
- All candidates get `score=0.0` placeholder; actual scores come from re-ranker

**`LexRAGRetriever.retrieve(query) -> list[tuple[Chunk, float]]`:**
- Complete 5-step pipeline; falls back to raw `page_results` if no sub-chunk candidates found

### 5.7 Module: `src/retrieval/reranker.py`

**Purpose:** CrossEncoder-based precision re-ranking of candidate chunks.

**`Reranker.rerank(query, candidates) -> list[tuple[Chunk, float]]`:**
- Builds `[(query, chunk.text), ...]` pairs
- `CrossEncoder.predict(pairs)` → list of float scores (logit scale, not probabilities)
- Sorts descending; returns top `rerank_top_k=5`
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, max_length=512 tokens

### 5.8 Module: `src/generation/prompts.py`

**Purpose:** Format retrieved chunks into a [SOURCE X]-numbered context block and build the OpenAI message list.

**`format_context(chunks) -> str`:**
- Each chunk becomes: `[SOURCE {i+1}] (Document: {doc_id}, Page: {page_number})\n{chunk.text}`
- Chunks separated by `\n\n---\n\n`

**`build_prompt(query, chunks) -> list[dict]`:**
- System message: "Answer questions using ONLY the provided sources. Cite sources using [SOURCE X] notation. If the answer is not in the sources, say 'The provided documents do not contain this information'. Never hallucinate."
- User message: `Sources:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:`

### 5.9 Module: `src/generation/generator.py`

**Purpose:** Call OpenAI and extract structured output from the response.

**`Generator.generate(query, chunks) -> dict`:**
- Calls `client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, max_tokens=1500)`
- Extracts citation indices via `re.findall(r'\[SOURCE (\d+)\]', answer)` → 0-indexed
- Builds citations list: `{source_num, doc_id, page_number, text_snippet[:100]}`
- Confidence heuristic: searches for `["do not contain", "not found", "cannot find", "no information"]` in lowercased answer → `"low"` if any match
- Returns timing via `time.time()` before/after API call

**`confidence_threshold: 0.7`** is defined in config but NOT used in code — the confidence field is just `"high"` or `"low"` based on keyword detection, not a numeric threshold.

### 5.10 Module: `src/evaluation/metrics.py`

**Retrieval metrics (all take `relevant_pages: list[int]` as ground truth):**

- **`precision_at_k(retrieved, relevant_pages, k) -> float`:** Fraction of top-K chunks from relevant pages. `hits/k`.
- **`recall_at_k(retrieved, relevant_pages, k) -> float`:** Fraction of relevant pages found in top-K. `|retrieved_pages ∩ relevant_pages| / |relevant_pages|`.
- **`mrr(retrieved, relevant_pages) -> float`:** `1/rank` of first hit. 0.0 if no hit.
- **`page_citation_accuracy(retrieved, relevant_pages, k) -> float`:** Fraction of top-K with `page_number != -1`.

**`llm_judge(query, answer, ground_truth, client, model) -> dict`:**
- LLM-as-judge prompt asking for `{faithfulness (0-10), completeness (0-10), citation_quality (0-10), reasoning: "..."}` as JSON
- Calls `json.loads(response.choices[0].message.content)` — no try/except; invalid JSON causes crash

### 5.11 Module: `src/evaluation/benchmark.py`

**Purpose:** Compare `BaselineRetriever` vs `LexRAGRetriever` across 9 curated queries.

**`BENCHMARK_QUERIES`:** 9 queries in 3 categories, all from **IRS Circular E 2024**:
- `single_page` (3 queries): answer on one page (pages 24, 30, 33)
- `cross_page` (3 queries): answer spans 3–4 pages (pages 30–43)
- `penalty_rate` (3 queries): specific numbers across sections (pages 35–39)

**`run_benchmark(cfg)`:**
- Instantiates both retrievers (loads all 3 FAISS indices + CrossEncoder model)
- Evaluates at `K=5`
- Prints per-query results + category-level averages
- Computed metrics: P@5, R@5, MRR, citation accuracy

### 5.12 Module: `src/evaluation/ragas_eval.py`

**Purpose:** Full end-to-end evaluation of LexRAG pipeline using RAGAS-style LLM judge.

**`RAGASEvaluator._judge(query, chunks, answer, ground_truth) -> dict`:**
- Formats first 300 chars of each chunk as context
- Asks LLM to score 0-1 on: `context_precision`, `context_recall`, `faithfulness`, `answer_relevance`
- Response must be raw JSON (no markdown). `json.loads()` with no error handling.

**`RAGASEvaluator.run()`:**
- For each of 9 queries in `RAGAS_QUERIES`: retrieve → generate → judge
- Prints per-query scores and category averages
- `RAGAS_QUERIES` contains detailed ground truth answers derived from IRS Circular E 2024

### 5.13 Domain Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — retrieval of relevant context before LLM generation |
| **HyDE** | Hypothetical Document Embedding — generate a hypothetical answer, embed it, use that vector for retrieval |
| **Chunk** | A text segment extracted from a document, the basic unit of retrieval |
| **FAISS** | Facebook AI Similarity Search — in-memory approximate nearest neighbor library |
| **IndexFlatIP** | FAISS exact inner product search index; with L2-normalized vectors = cosine similarity |
| **CrossEncoder** | A bi-input transformer that reads (query, passage) jointly for precise relevance scoring |
| **Bi-encoder** | Two separate encoders for query and passage; FAISS uses this model |
| **Re-ranking** | Secondary precision pass that re-scores a candidate set with a more expensive model |
| **Lookback period** | IRS term: the 12-month period ending June 30 of the prior year, used to determine deposit schedule |
| **Trust fund recovery penalty** | 100% penalty on unpaid withheld taxes; personally assessed |
| **Manifest** | JSON file tracking all downloaded documents with metadata |
| **Neighbor expansion** | Including pages adjacent (±1) to retrieved pages to capture cross-page arguments |

---

## 6. Code Patterns & Conventions

### 6.1 Design Patterns Used

- **Strategy Pattern:** `ChunkingStrategy` (ABC) → `NaiveChunker`, `PageAwareChunker`, `HierarchicalChunker`. Factory function `get_chunker()` as the client-facing entry point.
- **Repository Pattern:** `FaissStore` encapsulates all vector storage details. Callers never call FAISS directly.
- **Factory Function:** `get_chunker(strategy: str, ...)` returns concrete strategy from string key.
- **Data Transfer Object:** `@dataclass` used for all cross-module data (`DocMetadata`, `PageContent`, `Document`, `Chunk`).
- **Facade:** `BaselineRetriever` and `LexRAGRetriever` each present a single `retrieve()` method hiding multi-step pipelines.
- **Template Method (partial):** Both retrievers share the pattern "embed query → search index → return (Chunk, float) list" but differ in how they embed (raw vs HyDE) and how many indices they use.

### 6.2 Error Handling Strategy

Inconsistent across the codebase:

- **`downloader.py`:** Catches `requests.HTTPError` and `requests.RequestException` explicitly; returns `False` on failure; 404 = immediate abort; no exception propagation.
- **`processor.py`:** Wraps `fitz.open()` in `try/except Exception`; logs error, returns `None`; callers check for `None`.
- **`faiss_store.py`:** Minimal — returns `[]` for empty index; no exception handling around FAISS operations.
- **`embedder.py`:** No exception handling around OpenAI API calls; network errors will propagate and crash.
- **`lexrag.py`:** No exception handling; falls back to `page_results` if `candidates` is empty.
- **`metrics.py` `llm_judge()`:** Calls `json.loads()` with no try/except — malformed LLM response raises `json.JSONDecodeError`.
- **`ragas_eval.py` `_judge()`:** Same issue — `json.loads(raw)` without error handling.

### 6.3 Logging Strategy

- All modules use `logging.getLogger(__name__)` for module-scoped loggers.
- `downloader.py` and `indexing.py` call `logging.basicConfig(level=logging.INFO)` at module scope.
- Log level: `INFO` everywhere, config says `INFO`.
- Format (configured): `"%(asctime)s | %(levelname)s | %(name)s | %(message)s"` — but the format is only defined in `configs/config.yaml`; actual `basicConfig()` calls don't use the configured format.
- No file handlers — all logs go to stderr/stdout.
- No structured logging (no JSON log output).

### 6.4 Naming Conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase` (`FaissStore`, `LexRAGRetriever`, `BaselineRetriever`, `Reranker`, `Generator`, `Embedder`, `RAGASEvaluator`)
- **Functions/methods:** `snake_case` (`retrieve`, `embed_chunks`, `build_prompt`, `format_context`)
- **Private methods:** `_prefixed_with_underscore` (`_hyde`, `_expand_pages`, `_get_sub_chunks`, `_cache_path`, `_load_from_cache`, `_save_to_cache`, `_judge`)
- **Constants:** `UPPER_SNAKE_CASE` (`LEGAL_PDF_URLS`, `FALLBACK_PDF_URLS`, `BENCHMARK_QUERIES`, `RAGAS_QUERIES`)
- **Config keys:** `snake_case` in YAML (`chunk_size`, `top_k`, `rerank_top_k`)
- **Data classes:** `PascalCase` (`DocMetadata`, `PageContent`, `Document`, `Chunk`)

### 6.5 Code Organization Philosophy

- **Zero magic numbers in code** (stated in config.yaml comment): all tunable values live in `configs/config.yaml`.
- **`cfg: dict` threading:** Config dict is passed as constructor argument to all classes that need it (`Embedder(cfg)`, `FaissStore(cfg)`, `LexRAGRetriever(cfg)`, etc.) — no global config state.
- **Incremental/idempotent pipelines:** Download, processing, and indexing steps all check if outputs exist before re-computing.
- **Annotated design decisions:** Extensive inline comments explaining rationale (e.g., why `IndexFlatIP`, why `temperature=0`), specifically for interview preparation.
- **`sys.path.insert(0, '.')` pattern:** Used in `benchmark.py`, `ragas_eval.py`, and test files to make `src.*` imports work when running scripts from the project root.
- **`if __name__ == "__main__":` CLI entry points:** All pipeline scripts have standalone CLIs (`downloader.py`, `processor.py`, `indexing.py`, `benchmark.py`, `ragas_eval.py`).

### 6.6 Import Style

Imports in all modules follow:
1. Standard library (`re`, `json`, `logging`, `os`, `time`, `pathlib`, `dataclasses`, `hashlib`, `pickle`)
2. Third-party (`numpy`, `faiss`, `openai`, `yaml`, `requests`, `tqdm`, `sentence_transformers`)
3. Internal (`from src.data.processor import Document`, etc.)

No `__all__` declarations. No wildcard imports.

---

## 7. Configuration & Environment

### 7.1 Environment Variables

Only one environment variable is used:

| Variable | Required | Used In | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | `src/data/embedder.py`, `src/retrieval/lexrag.py`, `src/generation/generator.py`, `src/evaluation/ragas_eval.py` | OpenAI API authentication key. Loaded via `os.getenv("OPENAI_API_KEY")` after `load_dotenv()` |

**Loaded via:** `python-dotenv` `load_dotenv()` call at module level in each file that needs it. The `.env` file is in the project root and is gitignored.

### 7.2 Configuration File: `configs/config.yaml`

Full breakdown of all configuration keys:

#### `data` section
| Key | Value | Description |
|---|---|---|
| `data.raw_dir` | `"data/raw"` | Directory for downloaded PDFs |
| `data.processed_dir` | `"data/processed"` | Directory for parsed JSON files |
| `data.indices_dir` | `"data/indices"` | Directory for FAISS index files |
| `data.manifest_file` | `"data/raw/manifest.json"` | Path to download manifest |
| `data.download.max_docs` | `30` | Maximum PDFs to download |
| `data.download.timeout_seconds` | `30` | Per-request HTTP timeout |
| `data.download.max_retries` | `3` | Retry attempts on failure |
| `data.download.retry_backoff` | `2.0` | Exponential backoff multiplier |
| `data.download.min_pages` | `2` | Skip PDFs shorter than this |
| `data.download.max_pages` | `200` | Truncate PDFs longer than this |
| `data.cleaning.min_line_length` | `20` | Not used in code |
| `data.cleaning.max_header_lines` | `3` | Not used in code |
| `data.cleaning.max_footer_lines` | `3` | Not used in code |

#### `chunking` section
| Key | Value | Description |
|---|---|---|
| `chunking.chunk_size` | `512` | Target chunk size in characters |
| `chunking.overlap` | `50` | Overlap between consecutive chunks |
| `chunking.min_chunk_size` | `100` | Minimum chunk size — **defined but not enforced** |
| `chunking.section_patterns` | `[...]` | Regex for section headers — **defined but not used** |

#### `embedding` section
| Key | Value | Description |
|---|---|---|
| `embedding.model` | `"text-embedding-3-small"` | OpenAI embedding model |
| `embedding.dimensions` | `1536` | Vector dimensionality |
| `embedding.batch_size` | `100` | Chunks per API call |
| `embedding.cache_dir` | `"data/processed/embedding_cache"` | File cache directory |
| `embedding.max_tokens_per_chunk` | `8000` | OpenAI hard limit is 8191 — enforced? No |

#### `faiss` section
| Key | Value | Description |
|---|---|---|
| `faiss.index_type` | `"IndexFlatIP"` | FAISS index type |
| `faiss.nlist` | `100` | IVF clusters (unused with FlatIP) |
| `faiss.nprobe` | `10` | IVF probe count (unused with FlatIP) |
| `faiss.hnsw_m` | `16` | HNSW connections (unused with FlatIP) |
| `faiss.normalize_vectors` | `true` | L2-normalize before insert/query |

#### `retrieval` section
| Key | Value | Description |
|---|---|---|
| `retrieval.top_k` | `20` | Candidates from FAISS before re-ranking |
| `retrieval.rerank_top_k` | `5` | Final chunks after re-ranking |
| `retrieval.hyde_enabled` | `true` | Use HyDE — checked in config but always used; no code branch on this flag |
| `retrieval.neighbor_pages` | `1` | Pages to expand left/right |
| `retrieval.reranker.model` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | HuggingFace CrossEncoder model |
| `retrieval.reranker.max_length` | `512` | Token limit for re-ranker |
| `retrieval.reranker.batch_size` | `32` | Not used; `CrossEncoder.predict()` gets all pairs at once |

#### `generation` section
| Key | Value | Description |
|---|---|---|
| `generation.model` | `"gpt-4o-mini"` | OpenAI generation model |
| `generation.temperature` | `0.0` | Deterministic output |
| `generation.max_tokens` | `1500` | Max response length |
| `generation.streaming` | `true` | Defined but streaming not implemented |
| `generation.confidence_threshold` | `0.7` | Defined but not used in code |

#### `cache` section (not wired in code)
| Key | Value | Description |
|---|---|---|
| `cache.enabled` | `true` | Redis cache enabled — not implemented |
| `cache.host` | `"localhost"` | Redis host |
| `cache.port` | `6379` | Redis port |
| `cache.db` | `0` | Redis database index |
| `cache.embedding_ttl` | `86400` | 24h TTL for embeddings |
| `cache.response_ttl` | `3600` | 1h TTL for responses |

#### `evaluation` section
| Key | Value | Description |
|---|---|---|
| `evaluation.top_k_values` | `[1, 3, 5, 10]` | P/R@K values — only K=5 used in benchmark.py |
| `evaluation.llm_judge_model` | `"gpt-4o-mini"` | Judge LLM for evaluation |
| `evaluation.benchmark_queries_file` | `"docs/benchmark_queries.json"` | Not used — hardcoded queries in benchmark.py |
| `evaluation.results_dir` | `"docs/benchmark_results"` | Results not saved to file |

#### `api` section (not wired in code)
| Key | Value | Description |
|---|---|---|
| `api.host` | `"0.0.0.0"` | API listen host |
| `api.port` | `8000` | API listen port |
| `api.rate_limit` | `60` | Requests per minute per IP |
| `api.max_concurrent` | `10` | Max concurrent requests |

#### `logging` section (partially wired)
| Key | Value | Description |
|---|---|---|
| `logging.level` | `"INFO"` | Log level |
| `logging.log_dir` | `"logs"` | Log directory — empty, no file logging implemented |
| `logging.format` | `"%(asctime)s | %(levelname)s | %(name)s | %(message)s"` | Format string — defined but not applied |

### 7.3 Feature Flags

| Flag | Location | Current State | Effect |
|---|---|---|---|
| `retrieval.hyde_enabled` | `config.yaml` | `true` | HyDE is always called in `LexRAGRetriever`; this flag is read from config but never branched on |
| `generation.streaming` | `config.yaml` | `true` | Streaming is not implemented; synchronous API call used |
| `cache.enabled` | `config.yaml` | `true` | Redis caching is not implemented |

---

## 8. Testing Strategy

### 8.1 Test Framework

- **Runner:** `pytest` 9.0.2 (installed), but test files primarily use `if __name__ == '__main__':` pattern
- **pytest-asyncio** 1.3.0 installed but no async tests exist
- No pytest fixtures, conftest.py, or parametrize decorators
- Tests use `assert` statements + `print()` for output

### 8.2 Test File Inventory

#### `tests/test_chunking.py` — Unit tests (no OpenAI dependency)

**Functions:**
- `test_all_strategies()`: Loads `IRS_Circular_E_2024.json`, runs all 3 chunkers, prints chunk counts. No assertions.
- `test_naive_loses_page_info()`: Asserts all naive chunks have `page_number == -1`.
- `test_page_aware_preserves_page_info()`: Asserts no page-aware chunks have `page_number == -1`.
- `test_hierarchical_parent_child_link()`: Asserts all sub-chunks' `parent_chunk_id` exists in the page chunks dict.

**Test data:** Reads from `data/processed/IRS_Circular_E_2024.json` — requires offline pipeline to have been run.

#### `tests/test_retrieval.py` — Integration smoke test (requires OpenAI key + indices)

Not using pytest assertions. Runs both retrievers on "What are the penalties for not paying employer taxes?", prints top-5 results and page comparison.

#### `tests/test_generation.py` — End-to-end smoke test (requires OpenAI key + indices)

Not using pytest assertions. Runs full baseline + LexRAG chains on the same query, prints answers, citations, latency, confidence.

### 8.3 Coverage Areas

| Area | Test Type | Coverage |
|---|---|---|
| `NaiveChunker` | Unit | `page_number=-1` assertion |
| `PageAwareChunker` | Unit | Page info preservation assertion |
| `HierarchicalChunker` | Unit | Parent-child link integrity assertion |
| `BaselineRetriever.retrieve()` | Smoke | Output printed, not asserted |
| `LexRAGRetriever.retrieve()` | Smoke | Output printed, not asserted |
| `Generator.generate()` | Smoke | Output printed, not asserted |
| `Embedder.embed_chunks()` | None | No dedicated test |
| `FaissStore` | None | No dedicated test |
| `Reranker.rerank()` | None | No dedicated test |
| `metrics.py` functions | None | No dedicated test |
| `RAGASEvaluator` | None | Run as script, not tested |
| `downloader.py` | None | No dedicated test |
| `processor.py` | None | No dedicated test |

### 8.4 Known Test Gaps

- No mocking of OpenAI API — integration tests require live API calls and real `.env`
- No assertion-based testing for retrieval quality (smoke tests only)
- No tests for `FaissStore`, `Embedder`, `Reranker`
- No tests for `processor.py` (PDF extraction, table detection)
- No tests for `downloader.py` (download logic, retry behavior, manifest writing)
- No tests for `generator.py` (citation extraction regex, confidence detection)
- No tests for `metrics.py` functions
- No parametrized tests
- Test files run on live `data/processed/` JSON files — no synthetic test fixtures

---

## 9. Known Issues, Tech Debt & Trade-offs

### 9.1 Current Known Issues

**Issue 1: `min_chunk_size` not enforced**
- `config.yaml` defines `chunking.min_chunk_size: 100` but no chunker checks this. Tiny overlap-only chunks at the end of pages (possible when remaining text < 100 chars) will be included.
- **Impact:** Minor noise in retrieval results.

**Issue 2: `section_patterns` not implemented**
- `config.yaml` defines regex patterns for section headers (`^\s*\d+\.\s+[A-Z]`, etc.) for section-aware chunking. None of the three chunkers use these patterns.
- **Impact:** Legal section boundaries are not respected during chunking; a ruling and its supporting rationale may be split across chunks.

**Issue 3: `hyde_enabled` flag not respected**
- `retrieval.hyde_enabled: true` is defined in config but `LexRAGRetriever.retrieve()` always calls `_hyde()` unconditionally. Setting this to `false` would have no effect.
- **Impact:** Cannot disable HyDE without code change.

**Issue 4: `generation.streaming: true` not implemented**
- The config says streaming is enabled but `Generator.generate()` uses synchronous `client.chat.completions.create()` without `stream=True`. A future API endpoint would need this for low-latency UX.

**Issue 5: `cache.enabled` not implemented**
- Redis caching is fully configured but zero code exists to use it. All embeddings go through the file cache; query results are not cached.

**Issue 6: `_expand_pages` ignores doc boundaries**
- `_expand_pages(page_numbers, doc_id)` receives `doc_id` but ignores it. Expanded page numbers are matched against ALL documents in `chunk_store`. For a multi-document corpus, page 5 of Document A could match page 5 of Document B.
- **Impact:** In current 2-doc corpus where one has 59 pages and the other 10, potential cross-document false matches exist for pages 1–10.

**Issue 7: `json.loads()` calls without try/except**
- `metrics.py:llm_judge()` and `ragas_eval.py:RAGASEvaluator._judge()` both call `json.loads()` directly on raw LLM output. LLMs sometimes prefix with markdown (```json ... ```) or reasoning text, which causes `json.JSONDecodeError`.
- **Impact:** Entire benchmark/evaluation run crashes on one malformed response.

**Issue 8: `logging.basicConfig` format not applied**
- `logging.level` and `logging.format` from `configs/config.yaml` are never passed to `basicConfig()` calls. Current format is the Python default (`WARNING:root:message`), not the configured format.

**Issue 9: No `requirements.txt` or `pyproject.toml`**
- Dependency list only exists in the venv. Reproducing the environment requires `pip freeze` from the venv or manual recreation.

**Issue 10: `manifest.json` not present (data/raw not committed)**
- `.gitignore` excludes `data/raw/`. A fresh clone has no manifest and no PDFs. `processor.py` handles missing manifest gracefully but `indexing.py` will fail with empty processed dir.

### 9.2 Technical Debt

- **No `requirements.txt`:** Critical for reproducibility. Should be generated with `pip freeze > requirements.txt` or managed with `pip-tools`.
- **`sys.path.insert(0, '.')` in scripts:** A workaround for the lack of proper package installation. Should use `pyproject.toml` with `pip install -e .` instead.
- **Empty `src/api/`:** The API layer is listed in `.gitignore` permitted directories (Claude Code settings) but contains no code. FastAPI + Uvicorn are installed but unused.
- **Hardcoded benchmark queries:** `BENCHMARK_QUERIES` and `RAGAS_QUERIES` are hardcoded in Python files. Config key `evaluation.benchmark_queries_file` points to a non-existent `docs/benchmark_queries.json`.
- **Evaluation results not saved:** Benchmark and RAGAS runs print to stdout but write nothing to `docs/benchmark_results/` (path defined in config but directory never created).
- **No corpus beyond 2 PDFs:** Config supports `max_docs=30` and lists 26 source URLs, but only 2 PDFs exist in `data/raw/`. Most download URLs may be stale or return 404.
- **`faiss.normalize_L2()` modifies input in-place:** `FaissStore.add()` and `FaissStore.search()` both call `faiss.normalize_L2()` on the passed array, silently mutating the caller's data. This is a correctness footgun.

### 9.3 Architectural Trade-offs

| Trade-off | Choice Made | Rationale | Cost |
|---|---|---|---|
| Exact vs approximate search | Exact (`IndexFlatIP`) | Corpus ≤ 50K chunks; no recall loss | O(N) search time |
| Embedding model | `text-embedding-3-small` (1536d) | 5× cheaper than ada-002, ~5% lower MTEB | Slight precision loss on edge cases |
| LLM for generation | `gpt-4o-mini` | Cost/quality tradeoff for prototype | Less accurate than gpt-4o |
| File-based embedding cache | MD5-keyed `.npy` files | Zero infrastructure dependency (no Redis) | Not shareable across machines; no TTL |
| Chunking in characters | Characters, not tokens | Simpler; no tokenizer dependency | Inconsistent actual context window usage |
| Parallel list (chunks + FAISS) | Two parallel arrays | Simple; efficient serialization | Index consistency fragile (delete/update impossible) |
| Single `config.yaml` | All hyperparameters in one file | Easy to tune without touching code | No per-environment override (dev/prod) |

### 9.4 Performance Bottlenecks

- **Reranker cold start:** `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` downloads and loads a ~90MB PyTorch model from HuggingFace on first instantiation. This takes 5–30 seconds.
- **FAISS index load time:** Loading the 3 FAISS indices and 3 pickle files on every script run. For 708+69+743 vectors this is negligible, but scales linearly.
- **`_get_sub_chunks()` is O(N) linear scan:** For 743 chunks it's fast, but scales poorly. At 1M chunks this becomes a bottleneck.
- **Embedding batching is per-chunk inside batch:** The cache check loop iterates chunk-by-chunk inside the batch, then makes one API call for misses. For a batch of 100 with 0 cache hits, this is fine. For large batches with partial cache, there's overhead.
- **`time.sleep(0.1)` is always called** after any API batch call, even if rate limit headroom exists. For 1,557 embeddings in batches of 100, this adds ~1.5 seconds of unnecessary sleep.

### 9.5 Security Considerations

- **OpenAI API key in `.env`:** Standard practice; gitignored. No risk unless `.env` is accidentally committed.
- **PDF download from external URLs:** PDFs are downloaded from government and public legal databases. Magic byte validation (`%PDF`) provides minimal protection against malicious content disguised as PDFs.
- **Pickle deserialization:** `FaissStore.load()` uses `pickle.load()` on `.pkl` files. If a malicious `.pkl` file is placed in `data/indices/`, it will execute arbitrary code on load. This is acceptable for a local research project but must be addressed before any multi-user deployment.
- **No input sanitization:** Query strings are passed directly into OpenAI API calls. Prompt injection risk if exposed via API — the system prompt is fairly restrictive ("Answer using ONLY the provided sources") but not hardened against adversarial inputs.
- **No rate limiting in place:** The planned API rate limit (60 req/min) is config-only; the actual FastAPI server is not built.

---

## 10. Roadmap & Open Questions

### 10.1 Planned Features / Next Steps

Based on config definitions, installed but unused libraries, empty module stubs, and the interview questions document:

1. **FastAPI REST endpoint** (`src/api/`) — `POST /query`, `GET /health`, streaming support
2. **Redis caching** — wire `cache.*` config to actual Redis client; implement embedding + response caching
3. **`benchmark_queries.json` externalization** — move hardcoded benchmark queries to `docs/benchmark_queries.json`
4. **Benchmark result persistence** — write results to `docs/benchmark_results/` as JSON
5. **Full 30-document corpus** — download and index all 26 source URLs
6. **Section-aware chunking** — implement `section_patterns` in a `SectionAwareChunker`
7. **Streaming generation** — implement `stream=True` in `Generator.generate()`
8. **Multi-tenant support** — namespace FAISS indices per user/org
9. **Streamlit UI** — `streamlit` 1.55.0 is installed but unused
10. **LangChain/LangGraph integration** — `langchain`, `langchain-openai`, `langgraph` are installed; may be planned for agentic query rewriting or multi-hop retrieval
11. **Proper packaging** — `pyproject.toml` + `pip install -e .`

### 10.2 Unresolved Design Decisions

- **When to use `hyde_enabled=false`?** HyDE always fires in `LexRAGRetriever`. There's no A/B path for non-HyDE LexRAG. Should be branched based on the config flag.
- **How to handle multi-document result mixing?** `_expand_pages` conflates page numbers across documents. If page 33 exists in both IRS Circular and Copyright Office, both get included in candidates.
- **Optimal `chunk_size` and `overlap`?** Currently set to 512 chars / 50 chars. The config notes these as tunable but no ablation study has been run.
- **When to use `PageAwareChunker` vs `HierarchicalChunker` for the LexRAG index?** Currently hierarchical is used. The page-aware chunker is not used in any index.
- **Should the embedding file cache be moved to Redis?** Currently two caches are defined (file + Redis). The final architecture should consolidate.
- **Production LLM choice:** Config comment says "Switch to gpt-4o for production" — the decision criterion for switching is undefined.
- **Confidence scoring:** Currently binary (keyword detection). A numeric 0.0–1.0 `confidence_threshold: 0.7` is configured but not implemented.

### 10.3 Scaling Concerns

| Concern | Current Limit | Scaling Path |
|---|---|---|
| FAISS index type | `IndexFlatIP` is O(N) exact search | Switch to `IndexIVFFlat` (nlist=100, nprobe=10 already configured) for >100K vectors |
| FAISS in-memory | All vectors must fit in RAM | Switch to `faiss.IndexHNSWFlat` + on-disk storage, or use a vector DB (Pinecone, Weaviate) |
| Embedding API cost | $0.02/1M tokens for 3-small | File cache mitigates re-embedding; at scale add Redis |
| `_get_sub_chunks` O(N) scan | Negligible at 743 chunks | Build inverted index: `{(doc_id, page_number): [chunk_indices]}` |
| Single-machine deployment | No horizontal scaling | Add Redis for shared cache; move FAISS to dedicated vector DB service |
| No incremental index updates | Rebuild entire FAISS index to add docs | FAISS `IndexIDMap` allows adding by ID; implement `add_document()` pipeline |
| OpenAI API latency | ~200–500ms per embedding call | Batch more aggressively; move to local embedding model (intfloat/e5-large) |

---

*Document generated by systematic audit of all source files in `c:\Users\Vivek\LexRag` on 2026-04-02.*
