import sys
import yaml
sys.path.insert(0, '.')

from src.retrieval.baseline import BaselineRetriever
from src.retrieval.lexrag import LexRAGRetriever

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

QUERY = "What are the penalties for not paying employer taxes?"

print("=" * 60)
print(f"QUERY: {QUERY}")
print("=" * 60)

# --- Baseline ---
print("\n[BASELINE RETRIEVAL]")
baseline = BaselineRetriever(cfg)
baseline_results = baseline.retrieve(QUERY)
for i, (chunk, score) in enumerate(baseline_results[:5]):
    print(f"\n  #{i+1} | score={score:.4f} | page={chunk.page_number} | doc={chunk.doc_id}")
    print(f"  {chunk.text[:150]}...")

# --- LexRAG ---
print("\n[LEXRAG RETRIEVAL]")
lexrag = LexRAGRetriever(cfg)
lexrag_results = lexrag.retrieve(QUERY)
for i, (chunk, score) in enumerate(lexrag_results[:5]):
    print(f"\n  #{i+1} | score={score:.4f} | page={chunk.page_number} | doc={chunk.doc_id}")
    print(f"  {chunk.text[:150]}...")

# --- Comparison ---
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
baseline_pages = set(c.page_number for c, _ in baseline_results[:5])
lexrag_pages   = set(c.page_number for c, _ in lexrag_results[:5])
print(f"Baseline top-5 pages : {sorted(baseline_pages)}")
print(f"LexRAG   top-5 pages : {sorted(lexrag_pages)}")
print(f"Pages only in LexRAG : {sorted(lexrag_pages - baseline_pages)}")
