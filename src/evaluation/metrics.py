from src.data.chunking import Chunk


def precision_at_k(retrieved: list[tuple[Chunk, float]], relevant_pages: list[int], k: int) -> float:
    """Of the top-K chunks, what fraction came from relevant pages?"""
    top_k = retrieved[:k]
    hits = sum(1 for chunk, _ in top_k if chunk.page_number in relevant_pages)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[tuple[Chunk, float]], relevant_pages: list[int], k: int) -> float:
    """Of all relevant pages, what fraction did we retrieve in top-K?"""
    top_k = retrieved[:k]
    retrieved_pages = set(chunk.page_number for chunk, _ in top_k)
    hits = len(set(relevant_pages) & retrieved_pages)
    return hits / len(relevant_pages) if relevant_pages else 0.0


def mrr(retrieved: list[tuple[Chunk, float]], relevant_pages: list[int]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant result."""
    for rank, (chunk, _) in enumerate(retrieved, start=1):
        if chunk.page_number in relevant_pages:
            return 1.0 / rank
    return 0.0


def page_citation_accuracy(retrieved: list[tuple[Chunk, float]], relevant_pages: list[int], k: int) -> float:
    """What fraction of top-K chunks have a valid (non -1) page citation?"""
    top_k = retrieved[:k]
    valid = sum(1 for chunk, _ in top_k if chunk.page_number != -1)
    return valid / k if k > 0 else 0.0

def llm_judge(query: str, answer: str, ground_truth: str, client, model: str) -> dict:
    prompt = f"""You are evaluating a RAG system answer against ground truth.

Query: {query}

Ground Truth: {ground_truth}

System Answer: {answer}

Score the answer on:
1. Faithfulness (0-10): Is the answer grounded in facts, no hallucination?
2. Completeness (0-10): Does it cover all points in ground truth?
3. Citation Quality (0-10): Are sources cited correctly?

Respond in JSON: {{"faithfulness": X, "completeness": X, "citation_quality": X, "reasoning": "..."}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    import json
    return json.loads(response.choices[0].message.content)