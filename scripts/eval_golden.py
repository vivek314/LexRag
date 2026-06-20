"""
eval_golden.py — Baseline-vs-LexRAG evaluation over the golden set.

Implements handoff §4/§6 step 3:
  - Runs BOTH pipelines (naive baseline + LexRAG) over `input data/golden_set.jsonl`.
  - Scores the 4 RAGAS metrics (faithfulness, answer relevancy, context precision,
    context recall) PLUS the `fail_if` correctness gate (D4) — RAGAS faithfulness
    measures grounding, not correctness, so a system that confidently cites the
    repealed 1961 Act can score high on faithfulness while being wrong as current law.
  - Emits a PER-CATEGORY and PER-DOC table (D5), never a single pooled average.

Judge model: pinned and configurable via env RAGAS_JUDGE_MODEL (default gpt-4o).
  NOTE (handoff §6): ideally a *different family* than the generator (gpt-4o-mini)
  to avoid self-preference bias. With only an OpenAI key available, gpt-4o is the
  pragmatic default; set RAGAS_JUDGE_MODEL / ANTHROPIC etc. for a cross-family judge.

Usage:
  venv/Scripts/python.exe scripts/eval_golden.py [--limit N] [--out docs/golden_results.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
import yaml

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_golden")
logger.setLevel(logging.INFO)


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


GOLDEN_PATH = "input data/golden_set.jsonl"
REFUSAL_MARKERS = [
    "do not contain", "does not contain", "not contain this information",
    "not found", "cannot find", "no information", "not in the provided",
    "not present in", "cannot answer", "unable to answer", "outside the scope",
]


def load_golden(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def strip_fences(s: str) -> str:
    """Strip ```json ... ``` fences from an LLM response (handoff §7 gotcha)."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUSAL_MARKERS)


class GeminiGateAdapter:
    """Adapts ChatGoogleGenerativeAI to the .generate(messages,...) interface used by the gate."""
    def __init__(self, model: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self._chat = ChatGoogleGenerativeAI(
            model=model, temperature=0, google_api_key=os.environ["GEMINI_API_KEY"]
        )

    def generate(self, messages: list[dict], max_tokens: int = 300, temperature: float = 0.0) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        lc = []
        for m in messages:
            lc.append(SystemMessage(m["content"]) if m["role"] == "system"
                      else HumanMessage(m["content"]))
        return self._chat.invoke(lc).content


def make_gate_judge(judge_model: str):
    if judge_model.startswith("gemini"):
        return GeminiGateAdapter(judge_model)
    from src.providers.openai_provider import OpenAILLMProvider
    return OpenAILLMProvider(api_key=os.environ["OPENAI_API_KEY"], model=judge_model)


def judge_fail_if(judge, question: str, answer: str, fail_if: str, ground_truth: str) -> dict:
    """LLM-judge the `fail_if` correctness gate. Returns {violates: bool, reason: str}."""
    sys_prompt = (
        "You are a strict evaluator of legal-answer correctness. "
        "You are given a question, the model's answer, the correct ground truth, "
        "and a FAIL condition. Decide whether the model's answer TRIGGERS the fail "
        "condition. Citing an outdated provision only to say it is NOT the current "
        "basis does NOT trigger the condition; presenting it as the current/correct "
        "basis DOES. Respond ONLY with JSON: {\"violates\": true|false, \"reason\": \"...\"}."
    )
    user = (
        f"QUESTION:\n{question}\n\nGROUND TRUTH:\n{ground_truth}\n\n"
        f"MODEL ANSWER:\n{answer}\n\nFAIL CONDITION:\n{fail_if}\n\nJSON verdict:"
    )
    raw = judge.generate(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        max_tokens=300, temperature=0.0,
    )
    try:
        obj = json.loads(strip_fences(raw))
        return {"violates": bool(obj.get("violates")), "reason": str(obj.get("reason", ""))}
    except (json.JSONDecodeError, ValueError):
        # Fail safe: if the judge output is unparseable, flag for manual review.
        return {"violates": None, "reason": f"UNPARSEABLE: {raw[:120]}"}


def correctness_gate(case: dict, answer: str, judge) -> dict:
    """
    Per-case correctness (D4). Returns {passed: bool|None, kind, detail}.
      - refusal cases   -> pass iff the system declined.
      - fail_if cases   -> pass iff the answer does NOT trigger the fail condition.
      - everything else -> not gated here (RAGAS metrics carry it); passed=None.
    """
    if case["category"] == "refusal":
        refused = is_refusal(answer)
        return {"passed": refused, "kind": "refusal_gate",
                "detail": "declined" if refused else "answered (should have refused)"}
    if case.get("fail_if"):
        v = judge_fail_if(judge, case["question"], answer, case["fail_if"], case["ground_truth"])
        if v["violates"] is None:
            return {"passed": None, "kind": "fail_if", "detail": v["reason"]}
        return {"passed": not v["violates"], "kind": "fail_if", "detail": v["reason"]}
    return {"passed": None, "kind": "none", "detail": ""}


def run_pipelines(cases: list[dict], cfg: dict):
    """Run baseline + lexrag retrieval and generation for every case."""
    from src.providers.factory import get_providers
    from src.retrieval.baseline import BaselineRetriever
    from src.retrieval.lexrag import LexRAGRetriever
    from src.generation.generator import Generator

    llm, embedding = get_providers()
    logger.info("Generator/HyDE provider: %s", type(llm).__name__)
    baseline = BaselineRetriever(cfg, embedding)
    lexrag = LexRAGRetriever(cfg, llm, embedding)
    generator = Generator(llm, cfg)
    rerank_k = cfg["retrieval"]["rerank_top_k"]

    rows = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        logger.info("[%d/%d] %s | %s", i, len(cases), case["id"], q[:60])

        b_chunks = baseline.retrieve(q)[:rerank_k]
        b_ans = generator.generate(q, b_chunks)

        l_chunks = lexrag.retrieve(q)[:rerank_k]
        l_ans = generator.generate(q, l_chunks, currency_aware=True)

        rows.append({
            "case": case,
            "baseline": {
                "answer": b_ans["answer"],
                "contexts": [c.text for c, _ in b_chunks],
                "doc_ids": [c.doc_id for c, _ in b_chunks],
            },
            "lexrag": {
                "answer": l_ans["answer"],
                "contexts": [c.text for c, _ in l_chunks],
                "doc_ids": [c.doc_id for c, _ in l_chunks],
            },
        })
    return rows


def ragas_scores(rows: list[dict], system: str, judge_model: str):
    """Score the 4 RAGAS metrics for one system. Returns {case_id: {metric: float}}."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        Faithfulness, ResponseRelevancy,
        LLMContextPrecisionWithReference, LLMContextRecall,
    )

    if judge_model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model=judge_model, temperature=0, google_api_key=os.environ["GEMINI_API_KEY"]))
    else:
        judge_llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    # Embeddings (for answer-relevancy) stay on OpenAI — not a judge-bias concern.
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    # Free-tier judges are rate-limited; keep concurrency low and let RAGAS retry.
    run_config = RunConfig(max_workers=3, timeout=120)
    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_emb),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    samples = []
    ids = []
    for r in rows:
        ids.append(r["case"]["id"])
        samples.append({
            "user_input": r["case"]["question"],
            "response": r[system]["answer"],
            "retrieved_contexts": r[system]["contexts"] or [" "],
            "reference": r["case"]["ground_truth"],
        })
    ds = EvaluationDataset.from_list(samples)
    result = evaluate(ds, metrics=metrics, llm=judge_llm, embeddings=judge_emb,
                      run_config=run_config, show_progress=True)
    df = result.to_pandas()
    out = {}
    metric_cols = [c for c in df.columns
                   if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    for idx, cid in enumerate(ids):
        out[cid] = {col: (None if df[col].isna()[idx] else float(df[col][idx]))
                    for col in metric_cols}
    return out


def fmt(x):
    return "  -  " if x is None else f"{x:.2f}"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def print_tables(rows, gates, ragas_b, ragas_l):
    metric_keys = sorted({k for d in list(ragas_b.values()) + list(ragas_l.values()) for k in d})

    # ---- Per-category ----
    cats = defaultdict(list)
    for r in rows:
        cats[r["case"]["category"]].append(r["case"]["id"])

    print("\n" + "=" * 100)
    print("PER-CATEGORY  (B = baseline, L = LexRAG)")
    print("=" * 100)
    header = f"{'category':<24} {'sys':<4}" + "".join(f"{m[:14]:>16}" for m in metric_keys) + f"{'gate_pass':>12}"
    print(header)
    print("-" * len(header))
    for cat, ids in cats.items():
        for sys_name, ragas in (("B", ragas_b), ("L", ragas_l)):
            mrow = [mean([ragas.get(i, {}).get(m) for i in ids]) for m in metric_keys]
            g = [gates[i][sys_name] for i in ids if gates[i][sys_name]["passed"] is not None]
            gp = mean([1.0 if x["passed"] else 0.0 for x in g]) if g else None
            print(f"{cat:<24} {sys_name:<4}" + "".join(f"{fmt(v):>16}" for v in mrow) + f"{fmt(gp):>12}")

    # ---- Per-doc ----
    docs = defaultdict(list)
    for r in rows:
        d = r["case"].get("source_doc") or "none/refusal"
        docs[d].append(r["case"]["id"])
    print("\n" + "=" * 100)
    print("PER-DOC")
    print("=" * 100)
    print(header)
    print("-" * len(header))
    for d, ids in docs.items():
        for sys_name, ragas in (("B", ragas_b), ("L", ragas_l)):
            mrow = [mean([ragas.get(i, {}).get(m) for i in ids]) for m in metric_keys]
            g = [gates[i][sys_name] for i in ids if gates[i][sys_name]["passed"] is not None]
            gp = mean([1.0 if x["passed"] else 0.0 for x in g]) if g else None
            print(f"{d:<24} {sys_name:<4}" + "".join(f"{fmt(v):>16}" for v in mrow) + f"{fmt(gp):>12}")

    # ---- Correctness gate detail (the D4 headline) ----
    print("\n" + "=" * 100)
    print("CORRECTNESS GATE (D4) — the number that catches confidently-wrong-but-grounded answers")
    print("=" * 100)
    print(f"{'id':<8}{'category':<24}{'baseline':<12}{'lexrag':<12}")
    print("-" * 56)
    for r in rows:
        cid = r["case"]["id"]
        gb, gl = gates[cid]["B"], gates[cid]["L"]
        if gb["passed"] is None and gl["passed"] is None:
            continue
        def mark(g):
            return "n/a" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
        print(f"{cid:<8}{r['case']['category']:<24}{mark(gb):<12}{mark(gl):<12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N cases (smoke test)")
    ap.add_argument("--golden", default=GOLDEN_PATH, help="path to the golden/held-out set jsonl")
    ap.add_argument("--out", default="docs/golden_results.json")
    ap.add_argument("--no-ragas", action="store_true", help="skip RAGAS, only run gate")
    args = ap.parse_args()

    load_dotenv()
    # Two judges, by call volume:
    #  - GATE (D4 correctness, ~1 call/case): the bias-sensitive judgment → prefer a
    #    CROSS-FAMILY judge (Gemini) vs the gpt-4o-mini generator. Few calls fit the
    #    Gemini free tier (20 req/day).
    #  - RAGAS metrics (~hundreds of calls): can't run on the Gemini free tier → gpt-4o.
    gate_judge_model = os.getenv(
        "GATE_JUDGE_MODEL", "gemini-2.5-flash" if os.getenv("GEMINI_API_KEY") else "gpt-4o")
    ragas_judge_model = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o")
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cases = load_golden(args.golden)
    if args.limit:
        cases = cases[:args.limit]
    logger.info("Loaded %d golden cases | generator=%s | gate_judge=%s | ragas_judge=%s",
                len(cases), cfg["generation"]["model"], gate_judge_model, ragas_judge_model)
    if ragas_judge_model.startswith("gpt"):
        logger.warning("RAGAS judge %s is the SAME family as the generator (gpt-4o-mini). "
                       "The cross-family check lives in the GATE (judge=%s).",
                       ragas_judge_model, gate_judge_model)

    rows = run_pipelines(cases, cfg)

    # Correctness gate (D4) — cross-family judge.
    gate_judge = make_gate_judge(gate_judge_model)
    gates = {}
    for r in rows:
        cid = r["case"]["id"]
        gates[cid] = {
            "B": correctness_gate(r["case"], r["baseline"]["answer"], gate_judge),
            "L": correctness_gate(r["case"], r["lexrag"]["answer"], gate_judge),
        }

    ragas_b, ragas_l = {}, {}
    if not args.no_ragas:
        logger.info("Scoring RAGAS (baseline)...")
        ragas_b = ragas_scores(rows, "baseline", ragas_judge_model)
        logger.info("Scoring RAGAS (lexrag)...")
        ragas_l = ragas_scores(rows, "lexrag", ragas_judge_model)

    print_tables(rows, gates, ragas_b, ragas_l)

    out = {
        "config": {"generator": cfg["generation"]["model"],
                   "gate_judge": gate_judge_model, "ragas_judge": ragas_judge_model},
        "results": [
            {
                "id": r["case"]["id"], "category": r["case"]["category"],
                "source_doc": r["case"].get("source_doc"),
                "question": r["case"]["question"], "ground_truth": r["case"]["ground_truth"],
                "baseline": {**r["baseline"], "gate": gates[r["case"]["id"]]["B"],
                             "ragas": ragas_b.get(r["case"]["id"], {})},
                "lexrag": {**r["lexrag"], "gate": gates[r["case"]["id"]]["L"],
                           "ragas": ragas_l.get(r["case"]["id"], {})},
            }
            for r in rows
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
