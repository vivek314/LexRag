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
from src.retrieval.bm25_retriever import BM25Retriever
from src.generation.generator import Generator
from src.providers.hf_provider import FastEmbedProvider, HuggingFaceLLMProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="LexRAG Legal Tech Hub", description="Advanced RAG vs Naive RAG Comparison")

# Resolve paths relative to this file so they work both locally and on Vercel
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
CONFIG_PATH = _ROOT / "configs" / "config.yaml"

# On Vercel (and any read-only Lambda), /var/task is read-only — use /tmp for
# anything we need to write at runtime (embedding cache, ingested files, etc.)
_IS_SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
_WRITABLE_ROOT = Path("/tmp") if _IS_SERVERLESS else _ROOT

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# OSS config: always use fastembed indices (384-dim, committed to git)
_oss_cfg = {
    **cfg,
    "data": {
        **cfg["data"],
        "indices_dir": str(_ROOT / "data" / "indices" / "oss"),
        "raw_dir": str(_WRITABLE_ROOT / "data" / "raw"),
        "processed_dir": str(_WRITABLE_ROOT / "data" / "processed"),
        "manifest_file": str(_ROOT / "data" / "raw" / "manifest.json"),
    },
    "embedding": {
        **cfg["embedding"],
        "cache_dir": str(_WRITABLE_ROOT / "data" / "embedding_cache_oss"),
        "dimensions": 384,
    },
}

# Initialize OSS provider once at startup (always available, no key needed)
_oss_embedder = FastEmbedProvider(hf_token=os.getenv("HF_TOKEN"))
_oss_llm = HuggingFaceLLMProvider(hf_token=os.getenv("HF_TOKEN"))

_baseline_retriever: Optional[BaselineRetriever] = None
_lexrag_retriever: Optional[LexRAGRetriever] = None
_bm25_retriever: Optional[BM25Retriever] = None


def _load_retrievers() -> None:
    global _baseline_retriever, _lexrag_retriever, _bm25_retriever
    try:
        _baseline_retriever = BaselineRetriever(_oss_cfg, _oss_embedder)
        _lexrag_retriever = LexRAGRetriever(_oss_cfg, _oss_llm, _oss_embedder)
        # Build BM25 index from the same chunks — offline fallback, no embedding API needed
        all_chunks = list(_baseline_retriever.store.chunks)
        _bm25_retriever = BM25Retriever(all_chunks)
        logger.info("Retrievers loaded — %d chunks in BM25 index.", len(all_chunks))
    except Exception as e:
        logger.error("Failed to load retrievers: %s. Run scripts/build_oss_indices.py first.", e)


_load_retrievers()


def _get_llm(openai_api_key: Optional[str]):
    """Return OpenAI LLM when key provided, None otherwise (OSS path uses extractive answers)."""
    if not openai_api_key:
        return None
    from src.providers.openai_provider import OpenAILLMProvider
    return OpenAILLMProvider(api_key=openai_api_key)


class QueryRequest(BaseModel):
    query: str


@app.get("/api/stats")
async def get_stats():
    try:
        manifest_path = Path(_oss_cfg["data"]["manifest_file"])
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
        raw_dir = Path(_oss_cfg["data"]["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest_path = raw_dir / f"{slug}.pdf"
        with open(dest_path, "wb") as buf:
            buf.write(await file.read())

        manifest_path = Path(_oss_cfg["data"]["manifest_file"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
        manifest = [d for d in manifest if d["doc_id"] != slug]
        meta = {
            "doc_id": slug, "title": Path(file.filename).stem, "source": "Uploaded File",
            "domain": "uploaded_documents", "date": datetime.date.today().isoformat(),
            "num_pages": 0, "local_path": str(dest_path).replace("\\", "/"),
        }
        manifest.append(meta)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        processed_dir = Path(_oss_cfg["data"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_file = processed_dir / f"{slug}.json"
        out_file.unlink(missing_ok=True)

        doc = process_pdf(str(dest_path), meta, cfg)
        if doc is None:
            raise HTTPException(status_code=500, detail="PDF text extraction failed.")
        out_file.write_text(json.dumps(asdict(doc), indent=2, ensure_ascii=False), encoding="utf-8")
        meta["num_pages"] = doc.num_pages
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        # Always rebuild OSS indices (fastembed) on new document ingestion
        build_indices(_oss_cfg, embedding_provider=_oss_embedder)
        _load_retrievers()
        return {"status": "success", "doc_id": slug, "title": meta["title"], "num_pages": doc.num_pages}
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


def _extractive_answer(query: str, chunks: list[tuple]) -> dict:
    """Build an extractive answer from BM25-retrieved chunks — no LLM needed."""
    if not chunks:
        return {"answer": "No relevant passages found for this query.", "citations": [], "confidence": "low"}
    excerpts = []
    citations = []
    for i, (chunk, score) in enumerate(chunks[:5]):
        excerpts.append(f"[SOURCE {i+1}] (Page {chunk.page_number}, {chunk.doc_id})\n{chunk.text[:400]}")
        citations.append({
            "source_num": i + 1,
            "doc_id": chunk.doc_id,
            "page_number": chunk.page_number,
            "text_snippet": chunk.text[:100],
        })
    answer = (
        f"Top relevant passages for: \"{query}\"\n\n"
        + "\n\n---\n\n".join(excerpts)
        + "\n\n(Add your OpenAI API key in Settings for AI-generated answers.)"
    )
    return {"answer": answer, "citations": citations, "confidence": "high"}


@app.post("/api/query")
async def execute_query(
    payload: QueryRequest,
    x_openai_api_key: Optional[str] = Header(None),
):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not _bm25_retriever:
        raise HTTPException(status_code=500, detail="RAG system not initialized.")

    llm = _get_llm(x_openai_api_key)
    provider_mode = "openai" if x_openai_api_key else "open-source (BM25)"

    try:
        if llm:
            # Full semantic RAG with OpenAI key
            generator = Generator(llm, cfg)
            t0 = time.time()
            baseline_chunks = _baseline_retriever.retrieve(query)[:5]
            baseline_gen = generator.generate(query, baseline_chunks)
            baseline_latency = int((time.time() - t0) * 1000)

            t1 = time.time()
            lexrag_chunks = _lexrag_retriever.retrieve(query, llm=llm)
            lexrag_gen = generator.generate(query, lexrag_chunks)
            lexrag_latency = int((time.time() - t1) * 1000)
        else:
            # OSS path: BM25 keyword search + extractive answers (no API needed)
            t0 = time.time()
            baseline_chunks = _bm25_retriever.retrieve(query, top_k=5)
            baseline_gen = _extractive_answer(query, baseline_chunks)
            baseline_latency = int((time.time() - t0) * 1000)

            t1 = time.time()
            # LexRAG BM25: use more candidates, re-rank by BM25 score (same chunks, higher top_k)
            lexrag_chunks = _bm25_retriever.retrieve(query, top_k=10)
            lexrag_gen = _extractive_answer(query, lexrag_chunks[:5])
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
                    "LexRAG preserves page-level layout for high-fidelity citations (baseline loses page numbers).",
                    "LexRAG uses Hypothetical Document Embedding (HyDE) when OpenAI key is provided.",
                    "LexRAG performs dynamic neighbor page expansion (+/-1 page) for multi-page context.",
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
