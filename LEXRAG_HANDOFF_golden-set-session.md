# LexRAG — Golden-Set Session Handoff

> Continuation of `LexRAG — Production & Evaluation Handoff` (planning session, 2026-06-17).
> That doc holds the strategy and phased roadmap. **This** doc records what happened in the
> follow-up session where the *real* corpus was inspected and the golden set was drafted —
> the decisions and verified facts that are not in the code.
> Companion artifact: `golden_set.jsonl` (19 candidate cases, produced this session).
> Repo: `C:\Users\Vivek\LexRag`. GitHub: `https://github.com/vivek314/LexRag`.

---

## 0. TL;DR — what changed and the single next step

The prior handoff said "first thing: inspect `input data`, it may have replaced the old corpus."
**It has.** The corpus is now two real documents (details in §1). The golden set has been
drafted and verified against their actual text — it is in `golden_set.jsonl`, 19 cases across
6 categories.

**Immediate next action:** (1) Vivek verifies each golden line against the source (grep the
`source_quote`, confirm section numbers). (2) Reconcile `source_pages` to LexRAG's ingestion
pagination from `data/processed/`. (3) Build the RAGAS harness over the set — but with an
**answer-correctness / `fail_if` gate added alongside the four RAGAS metrics** (see §4, this is
new and non-negotiable). Then run baseline-vs-LexRAG and produce the per-category table.

This completes Phase 0 from the original roadmap.

---

## 1. The real corpus (replaces the old one)

Old corpus (now gone): Copyright Circular 01, IRS Circular E, IT Act 2000.

New corpus — **two documents, deliberately different genres**:

### Doc A — SEBI Financial Education Booklet
- 73 pages, published **November 2020**. Genre: plain-language consumer financial-literacy
  education (savings, planning, securities, insurance, pension, borrowing, govt schemes, a tax
  chapter, Ponzi cautions, grievance redressal).
- **NOT authoritative law.** Its own disclaimer page: "for providing general information to the
  public." Friendly, example-driven prose.
- **It is stale.** Its entire tax chapter is anchored on the *repealed* Income-tax Act, **1961**
  (cites 80C, 80CCD(1B), 80D, 80G, 80TTA, s.24; uses the previous-year/assessment-year framing).

### Doc B — Income-tax Act, 2025 (as amended by Finance Act 2026)
- 686 pages, clean text layer. **This is the authoritative statute.**
- Verified facts (from the official PDF this session; cross-checked against incometaxindia.gov.in):
  - Act 30 of 2025; assented 21 Aug 2025; **in force 1 April 2026**; **repeals the 1961 Act**.
  - 536 sections, 23 chapters (I–XXIII), 16 schedules.
  - Dense legalese, heavy internal cross-references (s.2 definitions resolve via "section 515(3)(b)",
    "section 237(1)", "Chapter XIX-C", "clause (22)(iii)(A)").
- **The uploaded file is the official one** — its filename matches the gov URL
  `incometaxindia.gov.in/.../income_tax_act_2025_as_amended_by_fa_act_2026`.

### Verified 1961→2025 section mappings (drafted the golden set from these)
These were confirmed against the raw statute text this session. `80C`, `80D`, `80TTA`, `80CCD`
return **zero hits** in the 2025 Act.

| Concept (1961 §)            | 2025 Act location                                            |
|----------------------------|--------------------------------------------------------------|
| Life ins. / PF / ELSS (80C)| **s.123** read with **Schedule XV**; cap **₹1,50,000**       |
| Health insurance (80D)     | **s.126**                                                    |
| Donations (80G)            | **s.133** (note: string "80G" has 3 incidental hits elsewhere)|
| Savings-interest (80TTA)   | **s.153** ("Deduction for interest on deposits")             |
| Previous yr / Assessment yr| replaced by single **"tax year"** — **s.3** (12-mo FY from 1 Apr)|
| Residence test             | **s.6(2)**: (a) 182+ days in tax year; OR (b) 60+ days that yr AND 365+ days over preceding 4 yrs. Carve-outs in s.6(3). |
| "accountant" def           | **s.2(1)** → points to **s.515(3)(b)**                       |
| "advance tax" def          | **s.2(4)** → "as per **Chapter XIX-C**"                      |

> Provenance caveat: these were verified against the *raw PDF* text. Reconcile any page numbers
> to LexRAG's `data/processed/` pagination before trusting them. Section numbers and quotes are
> the stable anchors; pages are not.

---

## 2. Decisions locked this session (carry forward)

- **D1 — Conflict policy (the important one).** When the booklet and the statute disagree on a
  point of law, **the statute is authoritative and the booklet must never override it.** The
  booklet is *not* deleted — it stays as a subordinate plain-language explainer, overruled only
  on the things it's unreliable about (current section numbers, thresholds, what's in force).
  **Implementation:** tag every booklet chunk `source_type: explainer, authority: none,
  currency: 2020`. Tag statute chunks with `source: it_act_2025, section, as_amended_by, in_force_date`.
  Retrieval/answer-selection must prefer authoritative+current on points of law.

- **D2 — New golden-set category.** Added `stale_vs_authoritative` to the original five
  (`simple_lookup`, `spanning`, `vocab_mismatch`, `cross_reference`, `refusal`). This corpus
  produces the booklet-vs-statute conflict for free; it's the highest-signal category here.

- **D3 — Anchor statute cases on section + verbatim quote**, not page number. Three page-numbering
  systems exist (raw PDF index, booklet printed footer, LexRAG's ingestion pages) and they don't
  line up. `source_quote` is the real verification handle.

- **D4 — Eval must include an answer-correctness gate, not just RAGAS.** RAGAS faithfulness
  measures grounding, not correctness. A naive system that retrieves the stale booklet and answers
  "80C" scores **high on faithfulness** while being wrong as current law. So the harness must score
  the `fail_if` condition (see §3) per case, alongside the four RAGAS metrics. Without this, the
  headline number flatters a system that confidently cites repealed law.

- **D5 — Report per-doc and per-category, never pooled.** ~10× size asymmetry (73 vs 686 pages)
  means the statute dominates any pooled average; a booklet regression would hide. Emit booklet
  and statute numbers separately, and a per-category breakdown.

---

## 3. The golden set (`golden_set.jsonl`)

19 candidate cases. Distribution: 3 each of the first five categories, 4 of `stale_vs_authoritative`.
All facts traced to real document text — **none written from model memory.**

### Schema (per line)
```jsonc
{
  "id": "sa-01",
  "category": "stale_vs_authoritative",
  "question": "...",
  "ground_truth": "...",                 // the correct answer, in its own words
  "source_doc": "it_act_2025|sebi_booklet|null",  // null for refusal
  "source_section": "123 / Schedule XV", // stable anchor; null for booklet
  "source_pages": [],                    // provisional; [] for statute. Booklet = printed footer pg.
  "source_quote": "...",                 // verbatim snippet = real verification handle
  "tests": "currency_authority",         // which LexRAG lever this case exercises
  // stale_vs_authoritative cases ALSO carry:
  "distractor": "booklet says 80C / 1961 Act",
  "distractor_doc": "sebi_booklet",
  "distractor_pages": [53],
  "fail_if": "answer cites 80C or the 1961 Act as current."  // the correctness gate (D4)
}
```

### `tests` → category map (read the eval as a per-lever scorecard)
- `baseline_floor` (simple_lookup) — both systems should tie; if LexRAG can't match naive here, something's wrong.
- `hyde` (vocab_mismatch) — plain-English query vs statute legalese.
- `neighbor_expansion` (spanning) — answer straddles a chunk/page boundary.
- `statute_dfs` (cross_reference) — answer hinges on a "see section X / Schedule Y" pointer.
- `refusal_gate` (refusal) — answer absent from corpus; correct behavior is to decline.
- `currency_authority` (stale_vs_authoritative) — must prefer current statute over stale booklet.

### Verification still owed (Vivek)
- Grep each `source_quote` against the source; confirm each `source_section`.
- Eyeball **sa-02 (s.126)** — anchored on the heading line, not a full read of the provision.
- Eyeball **sp-03 (agricultural-income exclusions, s.2(5) proviso)** — most paraphrase-heavy ground
  truth in the set (summarizes a multi-clause proviso rather than quoting one clean sentence).

---

## 4. Beat-naive-RAG plan (each lever proven against a category)

"Naive RAG" = single dense retrieval over flat chunks → stuff → generate. No rerank, no hybrid,
no query rewrite, no neighbor expansion, no refusal.

| LexRAG lever                          | Beats naive on        | Why                                                    |
|---------------------------------------|-----------------------|--------------------------------------------------------|
| Hybrid dense+BM25 (RRF)               | cross_reference, numeric lookups | embeddings blur rare tokens ("s.80CCD", "182 days"); BM25 nails them |
| Reranker (BGE-reranker-v2-m3)         | overall context_precision | reorders plausible-but-wrong chunks out of the prompt |
| HyDE                                  | vocab_mismatch        | embeds a hypothetical legalese answer → matches statute |
| ±1 neighbor / hierarchical chunking   | spanning              | grabs the continuation chunk naive truncates           |
| StatuteChunker DFS                    | cross_reference       | follows "subject to section X" to the target           |
| Refusal gate                          | refusal               | naive always retrieves *something* and hallucinates    |
| Currency-aware retrieval (D1 tags)    | stale_vs_authoritative| prefers current authoritative chunk over stale explainer|

Run baseline + LexRAG over the golden set → score 4 RAGAS metrics (faithfulness, answer relevancy,
context precision, context recall) **+ the `fail_if` correctness gate (D4)** → per-category,
per-doc table (D5). The honest headline is that table: tied on `simple_lookup`, LexRAG clearly
ahead on the hard categories. That regenerates the "+X% over naive" claim with real rigor and
replaces the unverified "+34%".

---

## 5. Corpus currency / maintenance (the moat — product work, not model work)

The defensible seam (from the original §1): assemble the authoritative + current sources a layperson
can't. For Indian income tax that's a **three-layer** corpus, refreshed on a cycle:

- **Tier 1 — authoritative (governs answers):**
  - Income-tax Act, 2025 (have it).
  - Income-tax **Rules, 2026** (notified; many "how do I claim / which form" answers live here, not in the Act). **Add for a real product.**
  - **CBDT circulars & notifications** (rolling, interpretive). Add for production; skippable for the demo.
  - Pull canonical text **only** from `incometaxindia.gov.in` / `egazette.gov.in`. Commercial
    reproductions (ClearTax, Taxmann, CA blogs) are fine for cross-checking, never as the canonical chunk.
- **Tier 0 — explanatory (never governs):** the SEBI booklet, tagged per D1.
- **Refresh cadence:** annual **Finance Act** (every Feb budget) → re-ingest the new consolidated
  "as amended by FA <year>" Act PDF; rolling feed for CBDT notifications.
- **Per-chunk provenance metadata** (already implied by D1): `source_doc, section/rule, as_amended_by,
  publication_date, in_force_date`. This is what lets retrieval prefer current law and lets the
  system answer "as of when".

---

## 6. First actions for Claude Code, in order

1. **Verify the golden set** with Vivek (§3 "verification still owed"). Grep `source_quote`s;
   confirm section numbers; reconcile `source_pages` to `data/processed/` pagination.
2. **Implement the source tags (D1)** in ingestion/metadata: booklet = explainer/no-authority/2020;
   statute = authoritative + section + in-force/amended dates.
3. **Build the RAGAS harness** over `golden_set.jsonl`: run `src/retrieval/baseline.py` and
   `src/retrieval/lexrag.py`, score the 4 RAGAS metrics **plus the `fail_if` correctness gate (D4)**,
   emit a **per-category, per-doc** table (D5). Run on the `gpt-4o-mini` generation path first;
   **pin the RAGAS judge model explicitly and to a different family than the generator** (carried
   over from the original handoff's judge-bias warnings).
4. Only then proceed to Phase 1 retrieval upgrades, each measured against the relevant category above.

---

## 7. Carryover gotchas from the original handoff (still live)

- Branch `feat/page-load-ingestion` is polluted with `teacher/` sibling-project commits — clean
  before building.
- `VectorStore(ABC)` abstraction still owed (Phase 0) — mirror the `EmbeddingProvider` pattern;
  keep FAISS, don't migrate to pgvector yet.
- `json.loads()` crash on LLM-judge responses (markdown fences) — wrap parsing, strip fences,
  validate against a schema.
- CI gate flapping risk: judge temp 0, gate on a *margin* not equality, cache judge outputs.
- `context_recall` (RAGAS ≥0.2) decomposes the ground-truth answer into claims — it ignores
  `source_pages`. Keep pages for a separate hit@k metric; don't expect RAGAS to consume them.
- Confirm the harness passes `ground_truth` in the field name your installed RAGAS version expects
  (renamed across 0.1→0.2).
