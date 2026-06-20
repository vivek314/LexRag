"""
compare_statute.py — Compare Naive vs Hierarchical vs Statute retrieval on IT Act queries.

Retrieval strategies:
  Naive      : embed query -> search itact_naive -> top-5
  Hierarchical: embed query -> search itact_hier_pages -> expand neighbors
                -> get sub-chunks -> rerank top-5
  Statute    : embed query -> search itact_statute_sections -> resolve references
                -> rerank top-5

Metrics: P@5, R@5, MRR, SecCov@5

  SecCov@5 (Section Coverage) — the generation quality metric:
    For each relevant section, compute what fraction of its complete text (word-level)
    appears in the combined text of the top-5 retrieved chunks.
    Average across all relevant sections.

    Statute should win here: it returns the FULL section as one chunk.
    Naive/hierarchical return 512-char fragments — even when they "hit" a section
    they may only cover 20-40% of the section's total content.

Relevance matching:
  Statute chunks: exact section_id check (c.section_id in relevant_sections)
  Naive/Hierarchical: text-based keyword match — a chunk "hits" a section if any
                      of that section's distinctive keywords appear in chunk.text.
"""

import sys
sys.path.insert(0, ".")

import logging
import yaml
import numpy as np

from src.data.chunking import Chunk
from src.data.embedder import Embedder
from src.data.faiss_store import FaissStore
from src.retrieval.reranker import Reranker

logging.basicConfig(level=logging.WARNING)  # quiet for clean output

# ---------------------------------------------------------------------------
# Ground truth: (query, relevant_section_ids)
# Verified against IT Act 2000 text
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "query": "What is the penalty for unauthorized access to a computer system?",
        "relevant_sections": ["43", "66"],
        "category": "penalty",
    },
    {
        "query": "What are the duties of a Certifying Authority?",
        "relevant_sections": ["30", "31", "34"],
        "category": "authority",
    },
    {
        "query": "What is the punishment for identity theft using electronic means?",
        "relevant_sections": ["66C", "66D"],
        "category": "offence",
    },
    {
        "query": "Who has the power to adjudicate contraventions and what factors are considered?",
        "relevant_sections": ["46", "47"],
        "category": "adjudication",
    },
    {
        "query": "What is the definition of electronic signature and when is it considered reliable?",
        "relevant_sections": ["2", "3A"],
        "category": "definition",
    },
    {
        "query": "What are the conditions for exemption of intermediary liability?",
        "relevant_sections": ["79"],
        "category": "intermediary",
    },
    {
        "query": "What is cyber terrorism and what is the punishment?",
        "relevant_sections": ["66F", "70"],
        "category": "offence",
    },
    {
        "query": "What are the rules for retention of electronic records?",
        "relevant_sections": ["7", "7A"],
        "category": "governance",
    },
]

# ---------------------------------------------------------------------------
# Section keywords for text-based matching
# Used when a chunk has no section_id (naive / hierarchical strategies).
# Keywords are distinctive phrases / numbers drawn from the actual section text.
# A chunk "hits" section X if ANY of X's keywords appear in chunk.text (case-insensitive).
# ---------------------------------------------------------------------------
SECTION_KEYWORDS: dict[str, list[str]] = {
    "2":   ["definitions", "\"electronic record\"", "\"data\"", "\"computer\"", "\"communication device\""],
    "3A":  ["electronic signature", "reliable", "identified the originator", "integrity of"],
    "7":   ["retention of electronic records", "information contained therein", "format in which it was generated"],
    "7A":  ["audit of documents", "electronic form", "auditing authority"],
    "30":  ["certifying authority", "display its licence", "repository", "issue a certificate"],
    "31":  ["suspend the licence", "revocation of licence", "certifying authority"],
    "34":  ["certifying authority", "certification practice statement", "disclose"],
    "43":  ["penalty for damage", "unauthorized access", "without permission", "penalty not exceeding one crore"],
    "46":  ["power to adjudicate", "adjudicating officer", "hold an inquiry"],
    "47":  ["factors to be taken into account", "adjudicating officer", "amount of gain", "loss caused"],
    "66":  ["computer related offences", "dishonestly", "fraudulently", "section 43", "imprisonment"],
    "66C": ["identity theft", "electronic signature", "password", "unique identification feature"],
    "66D": ["cheating by impersonation", "communication device", "computer resource"],
    "66F": ["cyber terrorism", "unauthorized access", "penetrate", "strike terror", "imprisonment for life"],
    "70":  ["protected system", "vital", "notification", "government"],
    "79":  ["intermediary", "exemption from liability", "due diligence", "third party information"],
}


def _chunk_hits_section(chunk: Chunk, section_id: str) -> bool:
    """
    Return True if this chunk is relevant to the given section.

    Priority 1: exact section_id match (statute chunks)
    Priority 2: text keyword match (naive / hierarchical chunks)
    """
    if chunk.section_id is not None:
        return chunk.section_id == section_id
    keywords = SECTION_KEYWORDS.get(section_id, [])
    text_lower = chunk.text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _chunk_is_relevant(chunk: Chunk, relevant_sections: list[str]) -> bool:
    return any(_chunk_hits_section(chunk, sec) for sec in relevant_sections)


# ---------------------------------------------------------------------------
# Metrics (fair across all strategies)
# ---------------------------------------------------------------------------

def precision_at_k(results: list[tuple[Chunk, float]], relevant: list[str], k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for c, _ in top_k if _chunk_is_relevant(c, relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(results: list[tuple[Chunk, float]], relevant: list[str], k: int) -> float:
    """Fraction of relevant sections covered in top-k results."""
    top_k = results[:k]
    covered = {
        sec for sec in relevant
        if any(_chunk_hits_section(c, sec) for c, _ in top_k)
    }
    return len(covered) / len(relevant) if relevant else 0.0


def mrr(results: list[tuple[Chunk, float]], relevant: list[str]) -> float:
    for rank, (c, _) in enumerate(results, start=1):
        if _chunk_is_relevant(c, relevant):
            return 1.0 / rank
    return 0.0


def section_coverage(
    results: list[tuple[Chunk, float]],
    relevant: list[str],
    statute_store: FaissStore,
    k: int,
) -> float:
    """
    Generation quality metric: what fraction of each relevant section's complete
    text is present in the top-k retrieved context?

    Method: word-level recall.
      coverage(sec) = |words_in_top_k ∩ words_in_full_section| / |words_in_full_section|

    The 'full section' ground truth is taken from the statute index (which holds the
    complete, deduplicated section text). Average is taken over all relevant sections.

    This directly measures how much of the authoritative section text the LLM will
    receive — the key advantage of statute chunking over naive/hierarchical.
    """
    # Build section_id -> full text lookup from statute store
    sec_text: dict[str, str] = {
        c.section_id: c.text
        for c in statute_store.chunks
        if c.section_id and c.metadata.get("level") == "section"
    }

    # Combined word set from top-k results
    retrieved_words = set()
    for c, _ in results[:k]:
        retrieved_words.update(c.text.lower().split())

    coverages = []
    for sec_id in relevant:
        full_text = sec_text.get(sec_id, "")
        if not full_text:
            continue
        full_words = set(full_text.lower().split())
        if not full_words:
            continue
        cov = len(retrieved_words & full_words) / len(full_words)
        coverages.append(cov)

    return sum(coverages) / len(coverages) if coverages else 0.0


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

K = 5
REF_DEPTH = 2
SCORE_DECAY = 0.5


def embed_query(embedder: Embedder, query: str) -> np.ndarray:
    dummy = Chunk(chunk_id="q", doc_id="q", text=query,
                  page_number=-1, chunk_index=0, char_count=len(query))
    return embedder.embed_chunks([dummy])[0]


def retrieve_naive(store: FaissStore, qvec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
    return store.search(qvec, k)


def retrieve_hierarchical(
    page_store: FaissStore,
    chunk_store: FaissStore,
    reranker: Reranker,
    qvec: np.ndarray,
    query: str,
    top_k: int,
) -> list[tuple[Chunk, float]]:
    page_results = page_store.search(qvec, top_k)
    page_numbers = {c.page_number for c, _ in page_results}
    doc_ids      = {c.doc_id for c, _ in page_results}

    # Expand ±1 neighbor pages
    expanded = set()
    for p in page_numbers:
        expanded.update([p - 1, p, p + 1])
    expanded = {p for p in expanded if p > 0}

    candidates = [
        (c, 0.0)
        for c in chunk_store.chunks
        if c.page_number in expanded and c.doc_id in doc_ids
    ]
    if not candidates:
        return page_results
    return reranker.rerank(query, candidates)


def retrieve_statute(
    sec_store: FaissStore,
    reranker: Reranker,
    qvec: np.ndarray,
    query: str,
    top_k: int,
    ref_depth: int,
) -> list[tuple[Chunk, float]]:
    section_lookup: dict[str, Chunk] = {
        c.section_id: c
        for c in sec_store.chunks
        if c.section_id and c.metadata.get("level") == "section"
    }

    initial  = sec_store.search(qvec, top_k)
    reranked = reranker.rerank(query, initial)

    # DFS reference resolver
    visited = {c.chunk_id for c, _ in reranked}
    result  = list(reranked)

    def dfs(chunk: Chunk, score: float, depth: int) -> None:
        if depth == 0:
            return
        for sec_id in getattr(chunk, "references", []):
            target = section_lookup.get(sec_id)
            if target is None or target.chunk_id in visited:
                continue
            visited.add(target.chunk_id)
            decayed = score * SCORE_DECAY
            result.append((target, decayed))
            dfs(target, decayed, depth - 1)

    # Only follow references from positively-scored sections
    for chunk, score in reranked:
        if score > 0:
            dfs(chunk, score, ref_depth)

    # Filter: keep sections with positive reranker score OR reference-resolved ones.
    # Sections with negative scores were explicitly judged irrelevant by the reranker.
    positive = [(c, s) for c, s in result if s > 0]
    return positive if positive else result[:1]  # always return at least one result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading models and indices...")
    embedder = Embedder(cfg)
    reranker = Reranker(cfg)

    naive_store      = FaissStore(cfg); naive_store.load("itact_naive")
    hier_page_store  = FaissStore(cfg); hier_page_store.load("itact_hier_pages")
    hier_chunk_store = FaissStore(cfg); hier_chunk_store.load("itact_hier_chunks")
    statute_store    = FaissStore(cfg); statute_store.load("itact_statute_sections")

    print(f"  Naive index:      {naive_store.index.ntotal} vectors")
    print(f"  Hier page index:  {hier_page_store.index.ntotal} vectors")
    print(f"  Hier chunk index: {hier_chunk_store.index.ntotal} vectors")
    print(f"  Statute index:    {statute_store.index.ntotal} vectors")

    all_results = {"naive": [], "hierarchical": [], "statute": []}

    # Per-query header
    print("\n" + "=" * 100)
    print(f"{'QUERY':<42} | {'NAIVE':^25} | {'HIER':^25} | {'STATUTE':^25}")
    print(f"{'':42} | {'P@5':>5} {'R@5':>5} {'MRR':>5} {'Cov':>5} | {'P@5':>5} {'R@5':>5} {'MRR':>5} {'Cov':>5} | {'P@5':>5} {'R@5':>5} {'MRR':>5} {'Cov':>5}")
    print("=" * 100)

    for item in QUERIES:
        query    = item["query"]
        relevant = item["relevant_sections"]

        qvec = embed_query(embedder, query)

        n_res = retrieve_naive(naive_store, qvec, K)
        h_res = retrieve_hierarchical(hier_page_store, hier_chunk_store, reranker, qvec, query, 20)
        s_res = retrieve_statute(statute_store, reranker, qvec, query, 20, REF_DEPTH)

        scores = {}
        for name, results in [("naive", n_res), ("hierarchical", h_res), ("statute", s_res)]:
            scores[name] = {
                "precision": precision_at_k(results, relevant, K),
                "recall":    recall_at_k(results, relevant, K),
                "mrr":       mrr(results, relevant),
                "coverage":  section_coverage(results, relevant, statute_store, K),
            }
            all_results[name].append(scores[name])

        n, h, s = scores["naive"], scores["hierarchical"], scores["statute"]
        label = query[:42]
        print(
            f"{label:<42} | "
            f"{n['precision']:>5.2f} {n['recall']:>5.2f} {n['mrr']:>5.2f} {n['coverage']:>5.2f} | "
            f"{h['precision']:>5.2f} {h['recall']:>5.2f} {h['mrr']:>5.2f} {h['coverage']:>5.2f} | "
            f"{s['precision']:>5.2f} {s['recall']:>5.2f} {s['mrr']:>5.2f} {s['coverage']:>5.2f}"
        )

        # Which sections did statute resolve (including via references)?
        statute_sections = [(c.section_id, round(sc, 2)) for c, sc in s_res if c.section_id]
        print(f"  Statute sections resolved: {statute_sections[:10]}")

    # --- Summary ---
    print("\n" + "=" * 62)
    print(f"{'STRATEGY':<15} {'P@5':>9} {'R@5':>9} {'MRR':>9} {'SecCov':>9}")
    print("=" * 62)
    for name in ["naive", "hierarchical", "statute"]:
        rows  = all_results[name]
        avg_p = sum(r["precision"] for r in rows) / len(rows)
        avg_r = sum(r["recall"]    for r in rows) / len(rows)
        avg_m = sum(r["mrr"]       for r in rows) / len(rows)
        avg_c = sum(r["coverage"]  for r in rows) / len(rows)
        print(f"{name:<15} {avg_p:>9.3f} {avg_r:>9.3f} {avg_m:>9.3f} {avg_c:>9.3f}")
    print("=" * 62)

    # --- Delta table (statute vs naive) ---
    n_rows = all_results["naive"]
    s_rows = all_results["statute"]
    def _avg(rows, key): return sum(r[key] for r in rows) / len(rows)

    print(
        f"\nStatute vs Naive  "
        f"dP@5={_avg(s_rows,'precision')-_avg(n_rows,'precision'):+.3f}  "
        f"dR@5={_avg(s_rows,'recall')-_avg(n_rows,'recall'):+.3f}  "
        f"dMRR={_avg(s_rows,'mrr')-_avg(n_rows,'mrr'):+.3f}  "
        f"dSecCov={_avg(s_rows,'coverage')-_avg(n_rows,'coverage'):+.3f}"
    )


if __name__ == "__main__":
    main()
