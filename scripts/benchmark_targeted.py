"""
benchmark_targeted.py — Two targeted query sets that isolate when each
chunking strategy has a structural advantage.

HIERARCHICAL_QUERIES
  Chosen from short, page-clustered sections (32, 40, 40A, 77, 10A — all < 200 chars).
  WHY hierarchical wins:
    • Section fits entirely inside one page sub-chunk.
    • Statute index has tiny stubs for these sections → weak embeddings.
    • Hierarchical page-level search finds the right page; sub-chunk covers
      the full section + surrounding context in one hit.

STATUTE_QUERIES
  Chosen from long sections (87: 3196 chars, 46: 2306 chars, 68: 2464 chars,
  18: 1708 chars) and cross-reference chains.
  WHY statute wins:
    • Sections span 3–6 pages → hierarchical sub-chunks fragment them.
    • Statute returns the complete section text as one chunk → SecCov = 1.00.
    • Cross-reference resolver pulls related sections that live on distant pages.
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

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Dataset 1 — Hierarchical should win
# Short sections (< 200 chars) clustered on the same page.
# Statute index holds only the section title stub for these.
# ---------------------------------------------------------------------------
HIERARCHICAL_QUERIES = [
    {
        "query": "Is every Certifying Authority required to display its licence and where must it be displayed?",
        "relevant_sections": ["32"],
        "why": "Section 32 is 153 chars — fits in one page sub-chunk. Statute stub is just the heading.",
    },
    {
        "query": "Are contracts formed through electronic means valid and legally enforceable?",
        "relevant_sections": ["10A"],
        "why": "Section 10A is 135 chars — single short provision. Hierarchical page chunk covers it fully.",
    },
    {
        "query": "Does payment of compensation or penalty under the IT Act bar any other punishment for the same act?",
        "relevant_sections": ["77"],
        "why": "Section 77 is 85 chars — one sentence. Statute stub; hierarchical page chunk includes it with context.",
    },
    {
        "query": "What is the status of Controller, Deputy Controller and Assistant Controller as public servants?",
        "relevant_sections": ["82"],
        "why": "Section 82 is 94 chars — a single declaratory sentence. Page sub-chunk includes it easily.",
    },
    {
        "query": "What are the duties of a subscriber after receiving an Electronic Signature Certificate?",
        "relevant_sections": ["40A"],
        "why": "Section 40A is 62 chars stub in statute index. Hierarchical page chunk around section 40 covers it.",
    },
]

# ---------------------------------------------------------------------------
# Dataset 2 — Statute should win
# Long sections (1700–3200 chars) spanning multiple pages, plus cross-reference chains.
# Hierarchical sub-chunks fragment the answer; statute returns complete section.
# ---------------------------------------------------------------------------
STATUTE_QUERIES = [
    {
        "query": "What are ALL the specific matters on which the Central Government has power to make rules under the IT Act?",
        "relevant_sections": ["87"],
        "why": "Section 87 is 3196 chars — spans ~6 pages. Hierarchical fragments it; statute returns full list.",
    },
    {
        "query": "What is the complete procedure for adjudication of contraventions including all powers and steps?",
        "relevant_sections": ["46"],
        "why": "Section 46 is 2306 chars — ~4 pages. Statute returns it whole; hierarchical misses later sub-clauses.",
    },
    {
        "query": "What are all the functions the Controller of Certifying Authorities is empowered to perform?",
        "relevant_sections": ["18"],
        "why": "Section 18 is 1708 chars with a long enumerated list. Hierarchical sub-chunks cut the list mid-way.",
    },
    {
        "query": "Under what circumstances can the Controller give directions to a Certifying Authority and what must those directions contain?",
        "relevant_sections": ["68"],
        "why": "Section 68 is 2464 chars and references section 24. Statute resolver follows the cross-reference.",
    },
    {
        "query": "What are all the conditions and safeguards for intermediary exemption from liability, and which content offences does it reference?",
        "relevant_sections": ["79"],
        "why": "Section 79 is 1827 chars and references sections 67 and 67A. Statute resolver chains them together.",
    },
]

# ---------------------------------------------------------------------------
# Section keywords for text-based relevance matching (naive / hierarchical)
# ---------------------------------------------------------------------------
SECTION_KEYWORDS: dict[str, list[str]] = {
    # Short / hierarchical-favored
    "32":  ["display its licence", "certifying authority shall display", "conspicuous position"],
    "10A": ["contracts formed through electronic means", "electronically formed", "legally enforceable"],
    "77":  ["compensation, penalties or confiscation not to interfere", "other punishment", "conviction"],
    "82":  ["public servant", "controller", "deputy controller", "assistant controller"],
    "40A": ["duties of subscriber", "electronic signature certificate", "key pair", "acceptance"],
    # Long / statute-favored
    "87":  ["power of central government to make rules", "central government may", "prescribed"],
    "46":  ["power to adjudicate", "adjudicating officer", "inquiry", "civil court", "contraventions"],
    "18":  ["functions of controller", "controller may perform", "licences", "controller shall"],
    "68":  ["power of controller to give directions", "certifying authority shall comply", "directions"],
    "79":  ["exemption from liability of intermediary", "due diligence", "third party information"],
    # Referenced sections (for cross-reference bonus)
    "67":  ["publishing obscene material", "sexually explicit", "lascivious"],
    "67A": ["sexually explicit act", "material containing", "imprisonment"],
    "24":  ["certifying authority", "disclosure", "private key"],
    "47":  ["factors to be taken into account", "amount of gain", "loss caused"],
}


def _chunk_hits_section(chunk: Chunk, section_id: str) -> bool:
    if chunk.section_id is not None:
        return chunk.section_id == section_id
    keywords = SECTION_KEYWORDS.get(section_id, [])
    text_lower = chunk.text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _chunk_is_relevant(chunk: Chunk, relevant: list[str]) -> bool:
    return any(_chunk_hits_section(chunk, s) for s in relevant)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
K = 5

def precision_at_k(results, relevant, k):
    hits = sum(1 for c, _ in results[:k] if _chunk_is_relevant(c, relevant))
    return hits / k if k else 0.0

def recall_at_k(results, relevant, k):
    covered = {s for s in relevant if any(_chunk_hits_section(c, s) for c, _ in results[:k])}
    return len(covered) / len(relevant) if relevant else 0.0

def mrr(results, relevant):
    for rank, (c, _) in enumerate(results, 1):
        if _chunk_is_relevant(c, relevant):
            return 1.0 / rank
    return 0.0

def section_coverage(results, relevant, statute_store, k):
    sec_text = {
        c.section_id: c.text
        for c in statute_store.chunks
        if c.section_id and c.metadata.get("level") == "section"
    }
    retrieved_words = set()
    for c, _ in results[:k]:
        retrieved_words.update(c.text.lower().split())

    coverages = []
    for sec_id in relevant:
        full = sec_text.get(sec_id, "")
        if not full:
            continue
        full_words = set(full.lower().split())
        if full_words:
            coverages.append(len(retrieved_words & full_words) / len(full_words))
    return sum(coverages) / len(coverages) if coverages else 0.0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
SCORE_DECAY = 0.5
REF_DEPTH   = 2


def embed_query(embedder, query):
    dummy = Chunk(chunk_id="q", doc_id="q", text=query,
                  page_number=-1, chunk_index=0, char_count=len(query))
    return embedder.embed_chunks([dummy])[0]


def retrieve_naive(store, qvec, k):
    return store.search(qvec, k)


def retrieve_hierarchical(page_store, chunk_store, reranker, qvec, query, top_k):
    page_results = page_store.search(qvec, top_k)
    page_numbers = {c.page_number for c, _ in page_results}
    doc_ids      = {c.doc_id      for c, _ in page_results}

    expanded = set()
    for p in page_numbers:
        expanded.update([p - 1, p, p + 1])
    expanded = {p for p in expanded if p > 0}

    candidates = [
        (c, 0.0) for c in chunk_store.chunks
        if c.page_number in expanded and c.doc_id in doc_ids
    ]
    return reranker.rerank(query, candidates) if candidates else page_results


def retrieve_statute(sec_store, reranker, qvec, query, top_k, ref_depth):
    section_lookup = {
        c.section_id: c
        for c in sec_store.chunks
        if c.section_id and c.metadata.get("level") == "section"
    }
    initial  = sec_store.search(qvec, top_k)
    reranked = reranker.rerank(query, initial)

    visited = {c.chunk_id for c, _ in reranked}
    result  = list(reranked)

    def dfs(chunk, score, depth):
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

    for chunk, score in reranked:
        if score > 0:
            dfs(chunk, score, ref_depth)

    positive = [(c, s) for c, s in result if s > 0]
    return positive if positive else result[:1]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_query_set(label, queries, embedder, reranker,
                  naive_store, hier_page_store, hier_chunk_store, statute_store):
    print(f"\n{'=' * 105}")
    print(f"  DATASET: {label}")
    print(f"{'=' * 105}")
    print(f"{'QUERY':<55} | {'NAIVE':^25} | {'HIER':^25} | {'STATUTE':^25}")
    print(f"{'':55} | {'P@5':>5} {'R@5':>5} {'Cov':>5} {'MRR':>5} | {'P@5':>5} {'R@5':>5} {'Cov':>5} {'MRR':>5} | {'P@5':>5} {'R@5':>5} {'Cov':>5} {'MRR':>5}")
    print("-" * 105)

    totals = {name: {"p": 0, "r": 0, "c": 0, "m": 0}
              for name in ["naive", "hier", "statute"]}

    for item in queries:
        query   = item["query"]
        relev   = item["relevant_sections"]
        qvec    = embed_query(embedder, query)

        n_res = retrieve_naive(naive_store, qvec, K)
        h_res = retrieve_hierarchical(hier_page_store, hier_chunk_store, reranker, qvec, query, 20)
        s_res = retrieve_statute(statute_store, reranker, qvec, query, 20, REF_DEPTH)

        def scores(res):
            return {
                "p": precision_at_k(res, relev, K),
                "r": recall_at_k(res, relev, K),
                "c": section_coverage(res, relev, statute_store, K),
                "m": mrr(res, relev),
            }

        ns, hs, ss = scores(n_res), scores(h_res), scores(s_res)
        for name, sc in [("naive", ns), ("hier", hs), ("statute", ss)]:
            for k2 in totals[name]:
                totals[name][k2] += sc[k2]

        label_q = query[:55]
        print(
            f"{label_q:<55} | "
            f"{ns['p']:>5.2f} {ns['r']:>5.2f} {ns['c']:>5.2f} {ns['m']:>5.2f} | "
            f"{hs['p']:>5.2f} {hs['r']:>5.2f} {hs['c']:>5.2f} {hs['m']:>5.2f} | "
            f"{ss['p']:>5.2f} {ss['r']:>5.2f} {ss['c']:>5.2f} {ss['m']:>5.2f}"
        )
        # Show statute sections resolved
        resolved = [(c.section_id, round(sc2, 2)) for c, sc2 in s_res if c.section_id]
        print(f"  {'why: ' + item['why'][:90]}")
        print(f"  statute resolved: {resolved[:8]}")

    n = len(queries)
    print("-" * 105)
    print(f"{'AVERAGE':<55} | ", end="")
    for name in ["naive", "hier", "statute"]:
        t = totals[name]
        print(f"{t['p']/n:>5.2f} {t['r']/n:>5.2f} {t['c']/n:>5.2f} {t['m']/n:>5.2f} | ", end="")
    print()

    return totals, n


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

    h_totals, hn = run_query_set(
        "HIERARCHICAL-FAVORABLE (short page-clustered sections)",
        HIERARCHICAL_QUERIES,
        embedder, reranker,
        naive_store, hier_page_store, hier_chunk_store, statute_store,
    )

    s_totals, sn = run_query_set(
        "STATUTE-FAVORABLE (long multi-page sections + cross-references)",
        STATUTE_QUERIES,
        embedder, reranker,
        naive_store, hier_page_store, hier_chunk_store, statute_store,
    )

    # ---- Head-to-head summary ----
    print(f"\n{'=' * 70}")
    print("HEAD-TO-HEAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<35} {'Strategy':<12} {'P@5':>6} {'R@5':>6} {'Cov':>6} {'MRR':>6}")
    print("-" * 70)

    for ds_label, totals, n in [
        ("Hierarchical-Favorable", h_totals, hn),
        ("Statute-Favorable",      s_totals, sn),
    ]:
        for name in ["naive", "hier", "statute"]:
            t = totals[name]
            print(f"{ds_label:<35} {name:<12} {t['p']/n:>6.3f} {t['r']/n:>6.3f} {t['c']/n:>6.3f} {t['m']/n:>6.3f}")
        print("-" * 70)


if __name__ == "__main__":
    main()
