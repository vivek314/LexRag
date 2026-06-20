# Vercel Python serverless entrypoint for the LexRAG live demo.
# Slim: serves the static marketing site + POST /api/ask (Weaviate hybrid + Groq).
# No faiss / torch / sentence-transformers — Vercel-size-safe.
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from src.api.weaviate_rag import answer

app = FastAPI(title="LexRAG demo")

_SITE = ROOT / "web" / "index.html"


@app.post("/api/ask")
async def ask(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    return JSONResponse(answer(body.get("query", "")))


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/{full_path:path}")
async def site(full_path: str):
    # single-page site; everything else falls through to it
    return FileResponse(str(_SITE))
