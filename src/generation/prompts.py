from src.data.chunking import Chunk


def format_context(chunks: list[tuple[Chunk, float]]) -> str:
    """
    Format retrieved chunks into a numbered context block with citations.
    Each chunk gets a [SOURCE X] marker so the LLM can cite it.
    """
    parts = []
    for i, (chunk, score) in enumerate(chunks):
        parts.append(
            f"[SOURCE {i+1}] "
            f"(Document: {chunk.doc_id}, Page: {chunk.page_number})\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, chunks: list[tuple[Chunk, float]]) -> list[dict]:
    """
    Build the full message list for GPT-4.
    Returns OpenAI messages format: list of {role, content} dicts.
    """
    context = format_context(chunks)

    system = """You are a legal document assistant. Answer questions using ONLY the provided sources.
Rules:
- Cite sources using [SOURCE X] notation
- If the answer is not in the sources, say "The provided documents do not contain this information"
- Never hallucinate facts not present in the sources
- Be precise and concise"""

    user = f"""Sources:
{context}

Question: {query}

Answer with citations:"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ]