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


def _provenance_label(chunk: Chunk) -> str:
    """Human-readable provenance from D1 tags stamped on the chunk."""
    m = chunk.metadata or {}
    if m.get("authority") == "statute":
        bits = ["AUTHORITATIVE STATUTE"]
        if m.get("in_force_date"):
            bits.append(f"in force {m['in_force_date']}")
        if m.get("as_amended_by"):
            bits.append(f"as amended by {m['as_amended_by']}")
        if m.get("repeals"):
            bits.append(f"repeals {m['repeals']}")
        return " | ".join(bits)
    if m.get("source_type") == "explainer" or m.get("authority") == "none":
        cur = m.get("currency", "older")
        return f"EXPLANATORY ONLY — not authoritative law, dated {cur}"
    return "source authority unknown"


def format_context_currency_aware(chunks: list[tuple[Chunk, float]]) -> str:
    """Like format_context but each source is annotated with its D1 authority/currency."""
    parts = []
    for i, (chunk, score) in enumerate(chunks):
        parts.append(
            f"[SOURCE {i+1}] (Document: {chunk.doc_id} | {_provenance_label(chunk)} "
            f"| Page: {chunk.page_number})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def build_currency_aware_prompt(query: str, chunks: list[tuple[Chunk, float]]) -> list[dict]:
    """
    Currency-aware answer selection (handoff D1): the authoritative + current statute
    governs points of law; explanatory sources never override it, and a repealed
    provision must never be presented as the current basis.
    """
    context = format_context_currency_aware(chunks)

    system = """You are a legal document assistant. Answer using ONLY the provided sources.

Each source is tagged with its authority and currency:
- "AUTHORITATIVE STATUTE" sources are the current, governing law.
- "EXPLANATORY ONLY" sources are plain-language education; they are NOT authoritative
  and may be out of date (e.g. they may cite section numbers from a now-repealed Act).

Currency & authority rules (apply on points of law — which Act/section currently governs,
thresholds, definitions, what is in force):
- The AUTHORITATIVE STATUTE governs. If an EXPLANATORY source conflicts with it, follow the statute.
- NEVER present a repealed Act or a superseded section number as the current basis. If an
  explanatory source cites an outdated section (e.g. an old "Section 80-series" number or the
  repealed Income-tax Act, 1961), correct it using the statute and give the CURRENT section.
- For general / financial-education questions with no statutory conflict, use the explanatory
  sources normally.

Other rules:
- Cite sources using [SOURCE X] notation.
- If the answer is not in the sources, say "The provided documents do not contain this information".
- Never hallucinate facts not present in the sources. Be precise and concise."""

    user = f"""Sources:
{context}

Question: {query}

Answer with citations:"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]