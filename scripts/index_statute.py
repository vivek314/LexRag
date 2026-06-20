"""
index_statute.py — Process an IT Act PDF and build three sets of FAISS indices
for comparison: naive, hierarchical, and statute-aware.

Usage:
    python scripts/index_statute.py --pdf data/raw/it_act_2000_updated.pdf

Indices created (in data/indices/):
    itact_naive            — NaiveChunker  (flat, no page/section info)
    itact_hier_pages       — HierarchicalChunker level=page
    itact_hier_chunks      — HierarchicalChunker level=chunk
    itact_statute_sections — StatuteChunker level=section (reference targets)
    itact_statute_chunks   — StatuteChunker level=clause (LLM context)
"""

import argparse
import json
import logging
import sys
sys.path.insert(0, ".")

import yaml

from src.data.processor import process_pdf
from src.data.chunking import get_chunker
from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/raw/it_act_2000_updated.pdf")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Step 1: Process PDF ---
    logger.info(f"Processing PDF: {args.pdf}")
    meta = {
        "doc_id": "it_act_2000",
        "title": "The Information Technology Act, 2000",
        "source": "Ministry of Law and Justice, India",
        "domain": "legal",
        "date": "2000",
        "local_path": args.pdf,
    }
    doc = process_pdf(args.pdf, meta, cfg)
    if doc is None:
        logger.error("Failed to process PDF. Check the path and try again.")
        sys.exit(1)

    logger.info(f"Extracted {doc.num_pages} pages, {sum(p.char_count for p in doc.pages)} total chars")

    # Save processed JSON
    import json as _json
    from dataclasses import asdict
    from pathlib import Path
    out_path = Path(cfg["data"]["processed_dir"]) / "it_act_2000.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(asdict(doc), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved processed JSON: {out_path}")

    embedder = Embedder(cfg)
    cs = cfg["chunking"]["chunk_size"]
    ov = cfg["chunking"]["overlap"]

    # -------------------------------------------------------------------------
    # Strategy 1: Naive (flat, no structure)
    # -------------------------------------------------------------------------
    naive_chunks = get_chunker("naive", cs, ov).chunk(doc)
    logger.info(f"Naive: {len(naive_chunks)} chunks")

    naive_store = FaissStore(cfg)
    naive_store.add(naive_chunks, embedder.embed_chunks(naive_chunks))
    naive_store.save("itact_naive")

    # -------------------------------------------------------------------------
    # Strategy 2: Hierarchical (page -> sub-chunk)
    # -------------------------------------------------------------------------
    hier_chunks = get_chunker("hierarchical", cs, ov).chunk(doc)
    hier_pages = [c for c in hier_chunks if c.metadata.get("level") == "page"]
    hier_subs  = [c for c in hier_chunks if c.metadata.get("level") == "chunk"]
    logger.info(f"Hierarchical: {len(hier_pages)} page chunks + {len(hier_subs)} sub-chunks")

    page_store = FaissStore(cfg)
    page_store.add(hier_pages, embedder.embed_chunks(hier_pages))
    page_store.save("itact_hier_pages")

    sub_store = FaissStore(cfg)
    sub_store.add(hier_subs, embedder.embed_chunks(hier_subs))
    sub_store.save("itact_hier_chunks")

    # -------------------------------------------------------------------------
    # Strategy 3: Statute-aware (Chapter -> Section -> Clause)
    # -------------------------------------------------------------------------
    statute_chunks = get_chunker("statute", cs, ov).chunk(doc)
    sec_chunks    = [c for c in statute_chunks if c.metadata.get("level") == "section"]
    clause_chunks = [c for c in statute_chunks if c.metadata.get("level") == "clause"]
    logger.info(f"Statute: {len(sec_chunks)} sections + {len(clause_chunks)} clause chunks")

    # Print reference extraction stats
    total_refs = sum(len(c.references) for c in sec_chunks)
    sections_with_refs = sum(1 for c in sec_chunks if c.references)
    logger.info(f"  Sections with cross-references: {sections_with_refs}/{len(sec_chunks)}")
    logger.info(f"  Total references extracted: {total_refs}")

    # Sample: show top 5 most-referenced sections
    ref_counts: dict[str, int] = {}
    for c in sec_chunks:
        for r in c.references:
            ref_counts[r] = ref_counts.get(r, 0) + 1
    top_refs = sorted(ref_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"  Most cited sections: {top_refs}")

    sec_store = FaissStore(cfg)
    sec_store.add(sec_chunks, embedder.embed_chunks(sec_chunks))
    sec_store.save("itact_statute_sections")

    clause_store = FaissStore(cfg)
    clause_store.add(clause_chunks, embedder.embed_chunks(clause_chunks))
    clause_store.save("itact_statute_chunks")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"  Naive chunks:           {len(naive_chunks):>5}  -> itact_naive")
    print(f"  Hier page chunks:       {len(hier_pages):>5}  -> itact_hier_pages")
    print(f"  Hier sub-chunks:        {len(hier_subs):>5}  -> itact_hier_chunks")
    print(f"  Statute section chunks: {len(sec_chunks):>5}  -> itact_statute_sections")
    print(f"  Statute clause chunks:  {len(clause_chunks):>5}  -> itact_statute_chunks")
    print(f"\nRun comparison: python scripts/compare_statute.py")


if __name__ == "__main__":
    main()
