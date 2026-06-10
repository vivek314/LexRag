import json
import logging
import os
import sys
sys.path.insert(0, '.')

import yaml
from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval.lexrag import LexRAGRetriever
from src.generation.generator import Generator
from src.data.chunking import Chunk

load_dotenv()
logger = logging.getLogger(__name__)

# Ground truth text answers for all 9 benchmark queries
# Verified against IRS Circular E 2024
RAGAS_QUERIES = [
    # --- Single-page ---
    {
        "query": "What is the lookback period for determining deposit schedule?",
        "category": "single_page",
        "relevant_pages": [33],
        "ground_truth": (
            "The lookback period is the 12-month period ending June 30 of the prior year. "
            "For example, to determine your deposit schedule for 2024, you look at taxes reported "
            "during July 2022 through June 2023. If total taxes during that period were $50,000 or "
            "less you are a monthly depositor; if more than $50,000 you are a semiweekly depositor."
        ),
    },
    {
        "query": "How do you calculate federal income tax withholding using the wage bracket method?",
        "category": "single_page",
        "relevant_pages": [24],
        "ground_truth": (
            "To use the wage bracket method, find the employee's wage range in the appropriate table "
            "for their pay period and filing status. Locate the row matching the employee's wages and "
            "the column matching their withholding allowances from Form W-4. The amount at the "
            "intersection is the federal income tax to withhold."
        ),
    },
    {
        "query": "What is the additional Medicare tax rate for high earners?",
        "category": "single_page",
        "relevant_pages": [30],
        "ground_truth": (
            "The Additional Medicare Tax rate is 0.9%. Employers must withhold this additional tax "
            "on wages paid to an employee in excess of $200,000 in a calendar year, regardless of "
            "the employee's filing status. There is no employer match for the Additional Medicare Tax."
        ),
    },
    # --- Cross-page ---
    {
        "query": "What are the deposit schedules and what happens if you miss a deposit deadline?",
        "category": "cross_page",
        "relevant_pages": [33, 34, 35, 36],
        "ground_truth": (
            "There are two deposit schedules: monthly and semiweekly. Monthly depositors must deposit "
            "taxes for a month by the 15th of the following month. Semiweekly depositors must deposit "
            "taxes for Wednesday-Friday paydays by the following Wednesday, and for Saturday-Tuesday "
            "paydays by the following Friday. If you miss a deposit deadline, a failure-to-deposit "
            "penalty applies. The penalty ranges from 2% for deposits 1-5 days late, 5% for 6-15 days "
            "late, 10% for more than 15 days late, and 15% if not deposited within 10 days of the "
            "first notice from the IRS."
        ),
    },
    {
        "query": "How are social security and Medicare taxes calculated and when must they be deposited?",
        "category": "cross_page",
        "relevant_pages": [30, 32, 33],
        "ground_truth": (
            "Social security tax is 6.2% each for employer and employee (12.4% total) on wages up to "
            "the annual wage base ($168,600 for 2024). Medicare tax is 1.45% each (2.9% total) with "
            "no wage base limit. Employers also withhold 0.9% Additional Medicare Tax on wages over "
            "$200,000. These taxes must be deposited according to the employer's deposit schedule "
            "(monthly or semiweekly) based on the lookback period."
        ),
    },
    {
        "query": "What are the rules for hiring and paying household employees including tax obligations?",
        "category": "cross_page",
        "relevant_pages": [41, 42, 43],
        "ground_truth": (
            "If you pay a household employee cash wages of $2,700 or more in 2024 you must withhold "
            "and pay social security and Medicare taxes. You do not withhold federal income tax unless "
            "the employee requests it. You may owe federal unemployment tax if you paid total cash "
            "wages of $1,000 or more in any calendar quarter. You report household employment taxes "
            "on Schedule H of your Form 1040, not on Form 941."
        ),
    },
    # --- Penalty-rate ---
    {
        "query": "What are all the penalty percentages for late tax deposits?",
        "category": "penalty_rate",
        "relevant_pages": [35, 36],
        "ground_truth": (
            "The failure-to-deposit penalty percentages are: 2% for deposits made 1 to 5 days late; "
            "5% for deposits made 6 to 15 days late; 10% for deposits made more than 15 days late; "
            "10% for amounts subject to electronic deposit requirements but not deposited by EFTPS; "
            "15% for amounts still unpaid more than 10 days after the date of the first IRS notice "
            "or the day you receive a notice requiring immediate payment, whichever is earlier."
        ),
    },
    {
        "query": "What is the trust fund recovery penalty and who is personally liable?",
        "category": "penalty_rate",
        "relevant_pages": [36],
        "ground_truth": (
            "The trust fund recovery penalty equals 100% of the unpaid trust fund taxes (the employee "
            "share of withheld income tax and FICA taxes). It applies to any person who is responsible "
            "for collecting, accounting for, and paying over these taxes and who willfully fails to do "
            "so. This can include officers, partners, employees, or anyone with authority over the "
            "financial decisions of the business."
        ),
    },
    {
        "query": "What penalties apply for failure to file returns and failure to pay taxes?",
        "category": "penalty_rate",
        "relevant_pages": [39],
        "ground_truth": (
            "The failure-to-file penalty is 5% of the unpaid tax for each month or part of a month "
            "the return is late, up to a maximum of 25%. The failure-to-pay penalty is 0.5% of the "
            "unpaid tax per month, also up to 25%. If both penalties apply in the same month, the "
            "failure-to-file penalty is reduced by the failure-to-pay penalty amount. A minimum "
            "penalty applies for returns filed more than 60 days late."
        ),
    },
]


class RAGASEvaluator:
    """
    Evaluates the full LexRAG pipeline using RAGAS-style metrics:
      - Context Precision   : are retrieved chunks relevant to the query?
      - Context Recall      : did retrieval capture all info needed to answer?
      - Faithfulness        : is the answer grounded in the retrieved chunks?
      - Answer Relevance    : does the answer actually address the question?

    All four metrics are scored 0-1 by an LLM judge (gpt-4o-mini at temp=0).
    """

    JUDGE_PROMPT = """\
You are a strict RAG evaluation judge. Score the following on a scale of 0.0 to 1.0.

Query: {query}

Retrieved Contexts:
{contexts}

Generated Answer:
{answer}

Ground Truth Answer:
{ground_truth}

Score each metric and respond ONLY with valid JSON (no markdown, no explanation outside JSON):

{{
  "context_precision": <float 0-1, are the retrieved contexts relevant to the query?>,
  "context_recall": <float 0-1, do the contexts contain all info from the ground truth?>,
  "faithfulness": <float 0-1, is the answer fully grounded in the contexts with no hallucination?>,
  "answer_relevance": <float 0-1, does the answer directly address the query?>
}}"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.retriever = LexRAGRetriever(cfg)
        self.generator = Generator(cfg)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.judge_model = cfg["evaluation"]["llm_judge_model"]
        logger.info("RAGASEvaluator ready")

    def _judge(
        self,
        query: str,
        chunks: list[tuple[Chunk, float]],
        answer: str,
        ground_truth: str,
    ) -> dict:
        contexts = "\n\n".join(
            f"[{i+1}] (Page {chunk.page_number}): {chunk.text[:300]}"
            for i, (chunk, _) in enumerate(chunks)
        )
        prompt = self.JUDGE_PROMPT.format(
            query=query,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
        )
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    def evaluate_query(self, item: dict) -> dict:
        query = item["query"]
        ground_truth = item["ground_truth"]

        chunks = self.retriever.retrieve(query)
        generation = self.generator.generate(query, chunks)
        answer = generation["answer"]

        scores = self._judge(query, chunks, answer, ground_truth)

        return {
            "query": query,
            "category": item["category"],
            "answer": answer,
            "scores": scores,
        }

    def run(self, queries: list[dict] = None) -> None:
        queries = queries or RAGAS_QUERIES
        all_results = []

        print("\n" + "=" * 70)
        print("RAGAS EVALUATION — LexRAG Pipeline")
        print("=" * 70)

        for i, item in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] {item['category']}: {item['query'][:60]}...")
            result = self.evaluate_query(item)
            all_results.append(result)
            s = result["scores"]
            print(
                f"  CtxPrecision={s['context_precision']:.2f}  "
                f"CtxRecall={s['context_recall']:.2f}  "
                f"Faithfulness={s['faithfulness']:.2f}  "
                f"AnsRelevance={s['answer_relevance']:.2f}"
            )

        # --- Summary by category ---
        categories = {}
        for r in all_results:
            categories.setdefault(r["category"], []).append(r["scores"])

        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevance"]

        print("\n" + "=" * 70)
        print(f"{'CATEGORY':<15} {'METRIC':<20} {'AVG SCORE':>10}")
        print("=" * 70)

        for category, score_list in categories.items():
            for metric in metrics:
                avg = sum(s[metric] for s in score_list) / len(score_list)
                print(f"{category:<15} {metric:<20} {avg:>10.3f}")
            print("-" * 70)

        # --- Overall averages ---
        print(f"\n{'OVERALL':<15}", end="")
        for metric in metrics:
            avg = sum(r["scores"][metric] for r in all_results) / len(all_results)
            print(f"  {metric}={avg:.3f}", end="")
        print()


if __name__ == "__main__":
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    evaluator = RAGASEvaluator(cfg)
    evaluator.run()
