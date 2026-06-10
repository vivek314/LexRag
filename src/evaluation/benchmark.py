import sys
import yaml
sys.path.insert(0, '.')

from src.retrieval.baseline import BaselineRetriever
from src.retrieval.lexrag import LexRAGRetriever
from src.evaluation.metrics import precision_at_k, recall_at_k, mrr, page_citation_accuracy

# Ground truth: (query, category, relevant_pages)
# Pages verified manually from IRS Circular E 2024
BENCHMARK_QUERIES = [
    # --- Single-page queries (answer lives on one page) ---
    {
        "query": "What is the lookback period for determining deposit schedule?",
        "category": "single_page",
        "relevant_pages": [33],
    },
    {
        "query": "How do you calculate federal income tax withholding using the wage bracket method?",
        "category": "single_page",
        "relevant_pages": [24],
    },
    {
        "query": "What is the additional Medicare tax rate for high earners?",
        "category": "single_page",
        "relevant_pages": [30],
    },

    # --- Cross-page queries (answer spans multiple pages) ---
    {
        "query": "What are the deposit schedules and what happens if you miss a deposit deadline?",
        "category": "cross_page",
        "relevant_pages": [33, 34, 35, 36],
    },
    {
        "query": "How are social security and Medicare taxes calculated and when must they be deposited?",
        "category": "cross_page",
        "relevant_pages": [30, 32, 33],
    },
    {
        "query": "What are the rules for hiring and paying household employees including tax obligations?",
        "category": "cross_page",
        "relevant_pages": [41, 42, 43],
    },

    # --- Penalty-rate queries (specific numbers across sections) ---
    {
        "query": "What are all the penalty percentages for late tax deposits?",
        "category": "penalty_rate",
        "relevant_pages": [35, 36],
    },
    {
        "query": "What is the trust fund recovery penalty and who is personally liable?",
        "category": "penalty_rate",
        "relevant_pages": [36],
    },
    {
        "query": "What penalties apply for failure to file returns and failure to pay taxes?",
        "category": "penalty_rate",
        "relevant_pages": [39],
    },
]

K = 5  # Evaluate top-5 results


def run_benchmark(cfg: dict) -> None:
    print("Loading retrievers...")
    baseline = BaselineRetriever(cfg)
    lexrag   = LexRAGRetriever(cfg)

    results = {"single_page": [], "cross_page": [], "penalty_rate": []}

    for i, q in enumerate(BENCHMARK_QUERIES):
        query    = q["query"]
        category = q["category"]
        relevant = q["relevant_pages"]

        print(f"\n[{i+1}/{len(BENCHMARK_QUERIES)}] {category}: {query[:60]}...")

        b_results = baseline.retrieve(query)
        l_results = lexrag.retrieve(query)

        b_scores = {
            "precision": precision_at_k(b_results, relevant, K),
            "recall":    recall_at_k(b_results, relevant, K),
            "mrr":       mrr(b_results, relevant),
            "citation":  page_citation_accuracy(b_results, relevant, K),
        }
        l_scores = {
            "precision": precision_at_k(l_results, relevant, K),
            "recall":    recall_at_k(l_results, relevant, K),
            "mrr":       mrr(l_results, relevant),
            "citation":  page_citation_accuracy(l_results, relevant, K),
        }

        results[category].append({"baseline": b_scores, "lexrag": l_scores})

        print(f"  Baseline  -> P@5={b_scores['precision']:.2f}  R@5={b_scores['recall']:.2f}  MRR={b_scores['mrr']:.2f}  Citation={b_scores['citation']:.2f}")
        print(f"  LexRAG    -> P@5={l_scores['precision']:.2f}  R@5={l_scores['recall']:.2f}  MRR={l_scores['mrr']:.2f}  Citation={l_scores['citation']:.2f}")

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print(f"{'CATEGORY':<15} {'METRIC':<12} {'BASELINE':>10} {'LEXRAG':>10} {'DELTA':>10}")
    print("=" * 70)

    for category, category_results in results.items():
        if not category_results:
            continue
        for metric in ["precision", "recall", "mrr", "citation"]:
            b_avg = sum(r["baseline"][metric] for r in category_results) / len(category_results)
            l_avg = sum(r["lexrag"][metric]   for r in category_results) / len(category_results)
            delta = l_avg - b_avg
            sign  = "+" if delta >= 0 else ""
            print(f"{category:<15} {metric:<12} {b_avg:>10.3f} {l_avg:>10.3f} {sign}{delta:>9.3f}")
        print("-" * 70)


if __name__ == "__main__":
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_benchmark(cfg)