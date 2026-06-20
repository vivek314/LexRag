"""
weaviate_rag.py — Serverless-friendly LexRAG for the live demo (Vercel).

Pipeline (mirrors the benchmarked local config, minus the torch CrossEncoder which
can't run on Vercel): query-expansion -> HyDE (Groq) -> Weaviate hybrid (BM25+vector
+RRF, native) -> currency-aware selection (D1) -> Groq generation with a currency-aware
prompt + graceful refusal. No faiss / torch / sentence-transformers imports.
"""
from __future__ import annotations
import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from openai import OpenAI

COLLECTION = "LexChunk"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
POOL_N = 15
TOP_K = 5
EXPLAINER_WEIGHT = 0.35

# old (1961-Act) -> new (2025-Act) terminology (the Act's headline rename)
GLOSSARY = {"previous year": "tax year", "assessment year": "tax year"}

_wv = None
_groq = None


def _load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _clients():
    global _wv, _groq
    if _wv is None:
        _load_env()
        _wv = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
            headers={"X-Jinaai-Api-Key": os.environ["JINA_API_KEY"]},
        )
        _groq = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                       base_url="https://api.groq.com/openai/v1")
    return _wv, _groq


def _chat(messages, max_tokens=500):
    _, groq = _clients()
    return groq.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.0, max_tokens=max_tokens
    ).choices[0].message.content


def _expand(query: str) -> str:
    ql = query.lower()
    extra = [new for old, new in GLOSSARY.items() if old in ql and new not in ql]
    return f"{query} {' '.join(extra)}".strip() if extra else query


def _hyde(query: str) -> str:
    return _chat([
        {"role": "system", "content": "Write a concise hypothetical passage from an Indian "
         "income-tax statute or a financial-education guide that would answer the question. "
         "Be specific; name sections/amounts if plausible."},
        {"role": "user", "content": query},
    ], max_tokens=200)


def _is_explainer(p: dict) -> bool:
    return p.get("source_type") == "explainer" or p.get("authority") == "none"


def _currency_select(objs, k):
    """objs: list[(props, score)]. Down-weight explainer relative to statute; relevance leads."""
    has_s = any(p.get("authority") == "statute" for p, _ in objs)
    has_e = any(_is_explainer(p) for p, _ in objs)
    if not (has_s and has_e):
        return objs[:k]
    scores = [s for _, s in objs]
    lo, hi = min(scores), max(scores)
    rng = (hi - lo) or 1.0
    ranked = sorted(
        objs,
        key=lambda x: ((x[1] - lo) / rng) * (EXPLAINER_WEIGHT if _is_explainer(x[0]) else 1.0),
        reverse=True,
    )
    return ranked[:k]


def _provenance(p: dict) -> str:
    if p.get("authority") == "statute":
        bits = ["AUTHORITATIVE STATUTE (current law)"]
        if p.get("in_force_date"):
            bits.append(f"in force {p['in_force_date']}")
        if p.get("as_amended_by"):
            bits.append(f"as amended by {p['as_amended_by']}")
        return " | ".join(bits)
    return f"EXPLANATORY ONLY — not authoritative, dated {p.get('currency','older')}"


SYSTEM_PROMPT = """You are LexRAG, an assistant answering ONLY from the provided sources about
Indian income tax (Income-tax Act, 2025) and SEBI investor education.

Each source is tagged with authority/currency:
- "AUTHORITATIVE STATUTE" = current governing law.
- "EXPLANATORY ONLY" = plain-language education, possibly out of date.

Rules:
- On points of law (which Act/section governs, thresholds, definitions), the AUTHORITATIVE
  STATUTE governs. NEVER present a repealed Act or a superseded section number (e.g. an old
  "80-series" section or the Income-tax Act, 1961) as the current basis — correct it with the statute.
- For general financial-education questions with no statutory conflict, use the explanatory sources.
- If the answer is NOT in the sources, reply EXACTLY: "I don't know — that's outside the documents
  I cover (the Income-tax Act 2025 and the SEBI investor-education booklet)."
- Cite sources as [SOURCE X]. Be concise."""


def answer(query: str) -> dict:
    query = (query or "").strip()
    if not query:
        return {"answer": "Please ask a question.", "sources": [], "refused": True}

    expanded = _expand(query)
    hyde = _hyde(expanded)
    wv, _ = _clients()
    coll = wv.collections.get(COLLECTION)
    res = coll.query.hybrid(query=hyde, alpha=0.5, limit=POOL_N,
                            return_metadata=MetadataQuery(score=True))
    objs = [(o.properties, float(o.metadata.score or 0.0)) for o in res.objects]
    if not objs:
        return {"answer": SYSTEM_PROMPT.split('reply EXACTLY: "')[1].split('"')[0],
                "sources": [], "refused": True}

    chosen = _currency_select(objs, TOP_K)
    context = "\n\n---\n\n".join(
        f"[SOURCE {i+1}] (Document: {p['doc_id']} | {_provenance(p)} | "
        f"Section {p.get('section_id') or '-'} | Page {p.get('page_number')})\n{p['text']}"
        for i, (p, _) in enumerate(chosen)
    )
    ans = _chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"},
    ], max_tokens=500)

    refused = "i don't know" in ans.lower()
    sources = [{
        "doc_id": p["doc_id"], "authority": p.get("authority"),
        "section_id": p.get("section_id") or None, "page": p.get("page_number"),
        "snippet": p["text"][:200],
    } for p, _ in chosen]
    return {"answer": ans, "sources": sources, "refused": refused}
