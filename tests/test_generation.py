import sys
import yaml
sys.path.insert(0, '.')

from src.retrieval.baseline import BaselineRetriever
from src.retrieval.lexrag import LexRAGRetriever
from src.generation.generator import Generator

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

QUERY = "What are the penalties for not paying employer taxes?"

generator = Generator(cfg)

print("=" * 60)
print(f"QUERY: {QUERY}")
print("=" * 60)

# --- Baseline chain ---
print("\n[BASELINE CHAIN]")
baseline = BaselineRetriever(cfg)
baseline_chunks = baseline.retrieve(QUERY)
baseline_result = generator.generate(QUERY, baseline_chunks[:5])

print(f"\nAnswer:\n{baseline_result['answer']}")
print(f"\nConfidence  : {baseline_result['confidence']}")
print(f"Latency     : {baseline_result['generation_time_ms']}ms")
print(f"Chunks used : {baseline_result['chunks_used']}")
print(f"\nCitations:")
for c in baseline_result['citations']:
    print(f"  [SOURCE {c['source_num']}] {c['doc_id']} | page {c['page_number']}")
    print(f"  ...{c['text_snippet']}...")

# --- LexRAG chain ---
print("\n" + "=" * 60)
print("[LEXRAG CHAIN]")
lexrag = LexRAGRetriever(cfg)
lexrag_chunks = lexrag.retrieve(QUERY)
lexrag_result = generator.generate(QUERY, lexrag_chunks)

print(f"\nAnswer:\n{lexrag_result['answer']}")
print(f"\nConfidence  : {lexrag_result['confidence']}")
print(f"Latency     : {lexrag_result['generation_time_ms']}ms")
print(f"Chunks used : {lexrag_result['chunks_used']}")
print(f"\nCitations:")
for c in lexrag_result['citations']:
    print(f"  [SOURCE {c['source_num']}] {c['doc_id']} | page {c['page_number']}")
    print(f"  ...{c['text_snippet']}...")
