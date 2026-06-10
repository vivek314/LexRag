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
from src.providers.hf_provider import FastEmbedProvider, HuggingFaceLLMProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="LexRAG Legal Tech Hub", description="Advanced RAG vs Naive RAG Comparison")

# Resolve paths relative to this file so they work both locally and on Vercel
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
CONFIG_PATH = _ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# OSS config: always use fastembed indices (384-dim, committed to git)
_oss_cfg = {
    **cfg,
    "data": {
        **cfg["data"],
        "indices_dir": str(_ROOT / "data" / "indices" / "oss"),
        "raw_dir": str(_ROOT / "data" / "raw"),
        "processed_dir": str(_ROOT / "data" / "processed"),
        "manifest_file": str(_ROOT / "data" / "raw" / "manifest.json"),
    },
    "embedding": {
        **cfg["embedding"],
        "cache_dir": str(_ROOT / "data" / "processed" / "embedding_cache_oss"),
        "dimensions": 384,
    },
}

# Initialize OSS provider once at startup (always available, no key needed)
_oss_embedder = FastEmbedProvider()
_oss_llm = HuggingFaceLLMProvider(hf_token=os.getenv("HF_TOKEN"))

_baseline_retriever: Optional[BaselineRetriever] = None
_lexrag_retriever: Optional[LexRAGRetriever] = None


def _load_retrievers() -> None:
    global _baseline_retriever, _lexrag_retriever
    try:
        _baseline_retriever = BaselineRetriever(_oss_cfg, _oss_embedder)
        _lexrag_retriever = LexRAGRetriever(_oss_cfg, _oss_llm, _oss_embedder)
        logger.info("OSS retrievers loaded successfully.")
    except Exception as e:
        logger.error("Failed to load retrievers: %s. Run scripts/build_oss_indices.py first.", e)


_load_retrievers()


def _get_llm(openai_api_key: Optional[str]):
    """Return the right LLM for this request — OpenAI if key provided, HF otherwise."""
    if not openai_api_key:
        return _oss_llm
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

        # LexRAG (pass per-request LLM for HyDE so it uses OpenAI when key is provided)
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
                    "LexRAG preserves page-level layout for high-fidelity citations (baseline loses page numbers).",
                    "LexRAG uses Hypothetical Document Embedding (HyDE) to bridge vocabulary gaps.",
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
