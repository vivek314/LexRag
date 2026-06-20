# Golden-Set Evaluation — Phase 0 Baseline (2026-06-17)

Honest baseline-vs-LexRAG result over `input data/golden_set.jsonl` (19 cases, 6 categories).
Generator: `gpt-4o-mini`. RAGAS judge: `gpt-4o` (⚠ same family — self-preference caveat;
swap via `RAGAS_JUDGE_MODEL` once a cross-family key exists). Harness: `scripts/eval_golden.py`.
Raw per-case output: `docs/golden_results.json`.

## Per-category (B = baseline, L = LexRAG)

| category               | sys | ans_rel | ctx_recall | faithful | ctx_prec | gate |
|------------------------|-----|---------|------------|----------|----------|------|
| simple_lookup          | B   | 0.59    | 0.67       | 1.00     | 0.59     |  –   |
| simple_lookup          | L   | 0.59    | **1.00**   | –        | 0.60     |  –   |
| vocab_mismatch         | B   | 0.87    | 1.00       | 1.00     | 0.75     |  –   |
| vocab_mismatch         | L   | 0.87    | 1.00       | 0.67     | 0.78     |  –   |
| spanning               | B   | 0.85    | 1.00       | 1.00     | 0.33     |  –   |
| spanning               | L   | 0.86    | 1.00       | 1.00     | 0.28     |  –   |
| cross_reference        | B   | 0.55    | 0.83       | 1.00     | 0.36     |  –   |
| cross_reference        | L   | 0.64    | 0.67       | 0.70     | **0.67** |  –   |
| refusal                | B   | –       | –          | –        | –        | 1.00 |
| refusal                | L   | –       | –          | –        | –        | 1.00 |
| stale_vs_authoritative | B   | 0.79    | 0.12       | 0.92     | 0.25     | 0.25 |
| stale_vs_authoritative | L   | 0.58    | 0.12       | 0.69     | 0.37     | 0.25 |

## Per-doc

| doc          | sys | ans_rel | ctx_recall | faithful | ctx_prec | gate |
|--------------|-----|---------|------------|----------|----------|------|
| it_act_2025  | B   | 0.68    | 0.50       | 0.96     | 0.38     | 0.25 |
| it_act_2025  | L   | 0.63    | 0.44       | 0.69     | 0.44     | 0.25 |
| sebi_booklet | B   | 0.88    | 1.00       | 1.00     | 0.65     |  –   |
| sebi_booklet | L   | 0.89    | 1.00       | 1.00     | **0.79** |  –   |
| refusal      | B/L | –       | –          | –        | –        | 1.00 |

## Headline findings

1. **The "+34% over naive" claim is NOT reproduced.** On this corpus LexRAG is roughly at
   parity with naive baseline, with small edges (simple_lookup context-recall 0.67→1.00;
   cross_reference & booklet context-precision) and a small faithfulness dip in
   cross_reference / stale categories. No category shows a large LexRAG win.

2. **Both systems FAIL the currency gate (0.25).** On `stale_vs_authoritative`, both baseline
   AND LexRAG cite the repealed `80C`/`80D` / previous-year framing from the SEBI booklet
   distractor (only sa-04 passes). This is the D4 gate doing its job: RAGAS faithfulness stays
   high (0.69–0.92) because the answers ARE grounded — in the *stale* source. Faithfulness
   alone would have flattered a confidently-wrong system.

3. **Root cause of #2: the D1 provenance tags are stamped on chunks but unused.** Neither
   retrieval ranking nor the generation prompt prefers authoritative+current. The
   currency-aware lever (handoff §4 last row) is the single highest-value Phase 1 build for
   this corpus, and the gate is now wired to measure it.

4. **Both refuse correctly (gate 1.00)** on the 3 out-of-corpus questions.

## Bugs found & fixed this session
- `indexing.py` never loaded `.env` → silently used OSS (384-dim) embeddings. Added a loader.
- `OpenAIEmbeddingProvider.embed()` sent all chunks in one request → OpenAI 300K-token 400.
  Added token-budget batching.
- **CrossEncoder reranker was silently disabled** (`ragas` install upgraded `huggingface_hub`,
  breaking `sentence-transformers 3.0.0`). LexRAG's fallback sorted by all-zero placeholder
  scores → page order → wrong chunks → refusals on everything. Fixed via `sentence-transformers 5.6.0`.

## Known fragility (not yet fixed)
- **LexRAG is broken on Vercel**: `requirements.txt` is intentionally torch-free, so the reranker
  is always disabled there, and the 0.0-score fallback collapses retrieval. Fix: make the reranker
  fallback score by embedding similarity instead of returning zeros.

## Phase 1 — currency-aware lever (BUILT 2026-06-17)

Implemented the D1-consuming lever as a **LexRAG-only** feature (baseline stays naive for a fair
comparison):
1. **Currency-aware generation prompt** (`build_currency_aware_prompt` in `prompts.py`): each source
   is labelled with its D1 authority/currency tag, and the system prompt adds a conflict rule —
   the authoritative+current statute governs points of law, and a repealed Act / superseded section
   number must never be presented as the current basis.
2. **Authoritative-first ordering** (`LexRAGRetriever._prefer_authoritative`, gated by
   `retrieval.currency_aware: true`): when a result set mixes statute + explainer chunks, the statute
   chunks lead the context. No-op for single-source sets, so booklet-only answers are unaffected.

**Result — `stale_vs_authoritative` gate (Gemini `gemini-2.5-flash` judge, cross-family):**

| system   | gate before | gate after |
|----------|-------------|------------|
| baseline | 0.25        | 0.25       |
| LexRAG   | 0.25        | **0.75**   |

LexRAG now passes sa-01 (→ s.123/2025 Act), sa-02 (→ 2025 Act, no longer 80D/1961), sa-04; baseline
still fails all of sa-01/02/03. Controls unaffected: vm-02 (compounding) and sl-02 (deposit insurance)
still answer correctly from the booklet. This is the first real, measured LexRAG win over naive.

**Remaining gap — sa-03** (previous-year/assessment-year): still fails because the s.3 "tax year"
definition that contradicts the booklet isn't retrieved — a retrieval-RECALL problem, not a
prompt/authority one.

## Phase 1b — hybrid dense+BM25 retrieval, RRF (BUILT 2026-06-17, NOT yet eval'd)

Added to LexRAG's retrieve path (`src/retrieval/lexrag.py`, gated by `retrieval.hybrid_enabled: true`):
a dense sub-chunk ranking and a BM25 keyword ranking are fused via **Reciprocal Rank Fusion**
(`_rrf_fuse`, `rrf_k=60`), then neighbor-expanded and reranked as before. BM25 indexes the same
sub-chunks as the dense `chunk_store`, so the two lists fuse over a shared id space. Bonus: the RRF
order is a *meaningful* fallback when the CrossEncoder reranker is absent (e.g. on Vercel), unlike the
old all-zero placeholder — partially addresses the Vercel fragility above.

Offline BM25 recall check (no API needed) confirms the value and corrects one expectation:
- **sa-01**: BM25 now surfaces "SCHEDULE XV [See section 123]" (the exact answer source dense missed). ✓
- **sa-02**: BM25 surfaces the s.126 health-insurance text (should fix the wrong "s.128" slip). ✓
- **sa-03**: BM25 does NOT surface the s.3 "tax year" definition. The question uses the *old* terms
  "previous year"/"assessment year" (which lexically match the BOOKLET); the 2025 Act uses the *new*
  term "tax year", absent from the query. So neither dense nor BM25 *on the question* finds s.3 — a
  genuine vocabulary gap. HyDE may bridge it (if its hypothetical answer says "tax year"), but this
  is unverified. sa-03 likely needs query expansion / a glossary mapping old→new terms.

**MEASURED 2026-06-18 — hybrid REGRESSED the stale gate, now turned OFF.** With hybrid on, LexRAG's
`stale_vs_authoritative` gate fell **0.75 → 0.50** (gate judged by gpt-4o, comparable to the prior run;
RAGAS judged by gpt-4o-mini this run to conserve credits). Cause: **sa-01 flipped PASS → FAIL** — BM25
over-pulled the booklet (the query "life-insurance / PPF / ELSS" lexically matches the booklet's 80C
investment section), so the final top-5 became 2 statute + 3 booklet, the CrossEncoder reranked the real
Schedule XV / s.123 chunk out, and the model cited 80C from a booklet source. BM25 helps when keywords
point at the statute (sa-02) but backfires when they point at the stale explainer (sa-01).

**Decision: `hybrid_enabled: false`** (reverted to the proven 0.75 currency-lever config). The RRF code
remains behind the flag — it still improves the Vercel no-reranker fallback. Before hybrid can be a net
win it needs **explainer-aware fusion**: down-weight `authority: none` chunks in the fused ranking (and/or
boost `authority: statute`) on points of law, so BM25's booklet hits can't crowd out the statute.

## Phase 1c — statute-aware chunking + cross-reference DFS (BUILT + MEASURED 2026-06-18)

The original `StatuteChunker` (regex tuned for the IT-Act-2000 layout) **could not parse the 2025 Act** —
it produced garbage sections (a date "2026", amendment footnotes as bodies) and missed s.126/133/153.
Added **`Statute2025Chunker`** (`src/data/chunking.py`): detects `^N. (1)` / `^N. <Capital>` section
starts, captures the marginal heading from the line above, rejects amendment/date/schedule noise, and
dedupes by **earliest position** (so the main s.3 "tax year" beats a schedule's "3."). Verified offline:
466 sections, all golden sections correct (3=tax year, 6=residence, 123/126/133/153/515 right), zero noise.

Wired the **cross-reference DFS**:
- `indexing.build_indices` now builds a `lexrag_sections` index (466 statute section chunks = DFS targets).
- `LexRAGRetriever` loads it into `_section_lookup`; `_resolve_references` extracts "section N" mentions
  from each candidate's text and pulls the cited section into the pool, **before** rerank (so resolved
  sections compete for the top-k — fixes the earlier post-rerank truncation). `ref_resolve_depth: 1`.

**Measured (gate=gpt-4o, RAGAS=gpt-4o-mini to conserve credits):**
- **Stale gate held at 0.75** (DFS doesn't disturb the currency lever); baseline 0.25.
- **cross_reference context-recall 0.67 → 0.83** (LexRAG vs baseline); spanning recall/precision also up.
- **DFS proven working:** cr-01 (accountant) pulled **s.515** into context and answered from it; cr-03
  (advance tax) surfaced Chapter XIX-C + s.404/405. **cr-02 still fails** — it references *Schedule XV*,
  not a section, and section-DFS can't resolve schedule/chapter targets (a known limit).

## Phase 1d — explainer-aware fusion + query expansion + section retrieval (2026-06-18)

Built three more levers (config-flagged in `configs/config.yaml`):

1. **Explainer-aware fusion** (`LexRAGRetriever._currency_select`, `explainer_weight: 0.35`): rerank a
   wider pool (`rerank_pool_n: 15`), then down-weight stale-explainer chunks RELATIVE to statute on the
   normalized relevance score before the top-k cut. This let **`hybrid_enabled` go back ON safely** — the
   stale gate held (no 0.50 collapse) and recovered sa-01 (now cites s.123, not 80C). Relevance still
   leads, so booklet-only guidance answers (sl-02 deposit insurance, vm-02 compounding) are unaffected.
2. **Query expansion** (`_expand_query`, old→new glossary): "previous year"/"assessment year" → "tax year"
   so 1961-term queries can match the 2025 Act. Principled (the Act's headline rename), not test-overfit.
3. **Section-index retrieval**: the `lexrag_sections` chunks are now a third RRF source, so whole-section
   *definitions* (distinctive headings like "Definition of tax year") are directly retrievable.

**Outcome (gate=gpt-4o, RAGAS=gpt-4o-mini):**
- **LexRAG stale gate → 1.00** (sa-01/02/04 real correct answers citing the 2025 Act; **sa-03 passes only
  by REFUSING** — see below). Baseline ~0.25 (and noisy: gpt-4o gate-judge flips sa-04 run-to-run — a
  reliability caveat, baseline code is unchanged).
- **cr-01 and cr-02 solved**: cr-01 follows the s.515 reference; **cr-02 was fixed by hybrid retrieval
  surfacing "SCHEDULE XV [See section 123]"** — so the planned schedule-DFS was unnecessary.
- cross_reference & vocab precision up vs the dense-only baseline; guidance controls intact.

**sa-03 — honest limitation (not fully solved).** It now REFUSES instead of asserting the stale
previous/assessment-year framework (safer than the baseline, which asserts it). But it does not yet give
the ideal answer ("No — replaced by the single 'tax year', s.3"). Root cause is fundamental: the
question's terms ("previous/assessment year") share NO vocabulary with the answer's source (s.3 says
"tax year"), so HyDE/BM25/rerank can't surface s.3 from the question, and "tax year" is too common to pin
the definition. The only reliable fix is a concept→section injection (force-inject s.3 when a query
mentions assessment/previous year) — deliberately NOT done, as it overfits this one case.

**Current best config: currency lever ON · hybrid ON · explainer-aware fusion (w=0.35) · statute DFS ON
· query expansion ON · section retrieval ON.** Remaining: sa-03 real answer (concept injection), and the
gate-judge non-determinism (consider voting/caching judge calls).

## Phase 1e — HELD-OUT VALIDATION (2026-06-20) — the overfitting check

Built a 13-case held-out set (`input data/heldout_set.jsonl`) authored from the documents (not from
retrieval behaviour), all quotes/distractors verified against source, distinct from the original 19.
**Four NEW stale traps** the system had never seen: 80G→s.133, 80TTA→s.153, 80CCD→s.124, Section 24→house
property s.22. Ran the FROZEN config once (gate=gpt-4o, RAGAS=gpt-4o-mini); no tuning afterwards.

**Result — the core result GENERALIZES:**

| category | baseline | LexRAG (held-out) |
|---|---|---|
| **stale_vs_authoritative gate** | **0.25** | **1.00** (ha-01..04 — REAL answers citing s.133/153/124, not refusals) |
| cross_reference | recall 0.00, ans 0.21 (missed) | recall 1.00, ans 0.77 |
| vocab_mismatch | ans 0.69, faith 0.79 | ans 0.94, faith 1.00 |
| simple_lookup / spanning / refusal | strong | tie (baseline floor) |

The headline — *LexRAG turns 25%-correct-on-stale-law into 100%, where naive RAG confidently cites
repealed sections* — **held on data it was not built against.** So the currency lever + cross-ref
machinery are real mechanisms, not memorization. Caveats: small n per held-out category (stale n=4 is
the solid one); the `explainer_weight=0.35` knob WAS exercised and held up, but the **query-expansion
glossary was not tested** (no previous/assessment-year case in the held-out set), so its generalization
is still unproven; LexRAG faithfulness dips on stale/cross-ref (model synthesizes the section number,
not always verbatim in-chunk). The non-generalizing last-mile hacks (sa-03 refusal, glossary) are kept
as caveats, not headline claims.

**Judge note:** gate verdicts are identical under gpt-4o and gemini-2.5-flash judges — robust across
families. Judge split (in `eval_golden.py`): `GATE_JUDGE_MODEL` (default gemini-2.5-flash, cross-family,
few calls — fits Gemini's 20-req/day free tier) vs `RAGAS_JUDGE_MODEL` (default gpt-4o; RAGAS needs
hundreds of calls, can't run on the Gemini free tier).

**RAGAS metric refresh — BLOCKED (2026-06-17):** the post-lever RAGAS metric table is incomplete
because BOTH API quotas were exhausted: the OpenAI key now returns `insufficient_quota` (429) and the
Gemini free-tier daily cap was spent on the gate. The lever's headline (the gate) is unaffected and
fully confirmed. The 4 RAGAS metrics for the non-stale categories are unchanged by the lever (it is a
no-op on single-source result sets and only alters generation on statute/explainer conflicts), so the
pre-lever metric table above remains representative. To regenerate the full post-lever metric table:
add OpenAI credit (or wait for the Gemini daily reset) and re-run `scripts/eval_golden.py`.
