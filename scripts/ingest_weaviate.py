"""
ingest_weaviate.py — Load the LexRAG corpus into Weaviate Cloud for the live demo.

Collection `LexChunk`:
  - Vectorized server-side by Jina (text2vec-jinaai) on the `text` property only.
  - Carries the D1 provenance tags (source_type / authority / currency / ...) so the
    backend can apply currency-aware selection, plus section_id/references for cross-ref.

Objects: HierarchicalChunker sub-chunks (both docs) + Statute2025Chunker section chunks
(statute) — mirrors the local best-config retrieval pool + cross-reference targets.

Run once:  venv/Scripts/python.exe scripts/ingest_weaviate.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, ".")

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

from src.data.indexing import load_documents, _stamp_provenance
from src.data.chunking import get_chunker

COLLECTION = "LexChunk"


def load_env(path=".env"):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def build_chunks(cfg):
    docs = load_documents(cfg["data"]["processed_dir"])
    cs, ov = cfg["chunking"]["chunk_size"], cfg["chunking"]["overlap"]
    hier = get_chunker("hierarchical", cs, ov)
    statute = get_chunker("statute_2025", cs, ov)
    out = []
    for doc in docs:
        subs = [c for c in hier.chunk(doc) if c.metadata.get("level") == "chunk"]
        out.extend(_stamp_provenance(doc, subs))
        if doc.metadata.get("authority") == "statute":
            secs = [c for c in statute.chunk(doc) if c.metadata.get("level") == "section"]
            out.extend(_stamp_provenance(doc, secs))
    return out


def main():
    load_env()
    import yaml
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
        headers={"X-Jinaai-Api-Key": os.environ["JINA_API_KEY"]},
    )
    try:
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)
            print(f"deleted existing {COLLECTION}")

        client.collections.create(
            COLLECTION,
            vectorizer_config=Configure.Vectorizer.text2vec_jinaai(model="jina-embeddings-v3"),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="doc_id", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="page_number", data_type=DataType.INT, skip_vectorization=True),
                Property(name="level", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="section_id", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="heading", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="source_type", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="authority", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="currency", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="in_force_date", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="as_amended_by", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="references", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
            ],
        )
        print(f"created collection {COLLECTION} (vectorizer: jina-embeddings-v3)")

        import time
        from weaviate.classes.data import DataObject
        chunks = build_chunks(cfg)
        coll = client.collections.get(COLLECTION)

        def to_obj(c):
            m = c.metadata or {}
            return DataObject(properties={
                "text": c.text, "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                "page_number": int(c.page_number) if c.page_number is not None else -1,
                "level": m.get("level", ""), "section_id": c.section_id or "",
                "heading": m.get("heading", ""), "source_type": m.get("source_type", ""),
                "authority": m.get("authority", ""), "currency": str(m.get("currency", "")),
                "in_force_date": m.get("in_force_date", "") or "",
                "as_amended_by": m.get("as_amended_by", "") or "",
                "references": list(c.references or []),
            })

        # Throttle to stay under Jina's 100k-tokens/min: ~70k-token groups, 65s apart.
        TOKEN_BUDGET = 70_000
        groups, cur, cur_tok = [], [], 0
        for c in chunks:
            t = max(1, len(c.text) // 4)
            if cur and cur_tok + t > TOKEN_BUDGET:
                groups.append(cur); cur, cur_tok = [], 0
            cur.append(c); cur_tok += t
        if cur:
            groups.append(cur)

        print(f"ingesting {len(chunks)} chunks in {len(groups)} throttled groups...")
        failed = 0
        for gi, group in enumerate(groups, 1):
            res = coll.data.insert_many([to_obj(c) for c in group])
            failed += len(res.errors)
            done = coll.aggregate.over_all(total_count=True).total_count
            print(f"  group {gi}/{len(groups)}: +{len(group)} | total={done} | failed={failed}")
            if res.errors:
                print("   sample error:", str(list(res.errors.values())[0].message)[:120])
            if gi < len(groups):
                time.sleep(65)
        total = coll.aggregate.over_all(total_count=True).total_count
        print(f"done. objects in collection: {total} | failed: {failed}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
