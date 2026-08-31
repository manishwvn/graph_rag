# Open Issues

Known problems, deferred work and methodological caveats. Nothing here is a
blocker for the current results — but everything here is a reason to read those
results with a specific caveat in mind.

**What is actually blocking the rest.** The remaining items fall into three
groups, and only the first is ordinary work:

1. *Blocked on measurement budget* (10, 11). The free tier allows 1000
   requests/day on the generation model and a full eval costs ~128. Any change
   to retrieval or generation must be re-measured before it can be published, so
   fixes land at roughly one validated change per day. Issue 11 has a measured,
   ready-to-apply fix waiting on exactly this.
2. *Cannot be fixed without a held-out set* (3, 4, 8, 12). These are all ranking
   or entity-resolution changes. Any variant chosen because it scores better on
   these same 16 queries is fitted to the test set, and the improvement would be
   an artifact. The honest unblock is a larger query set split into dev and
   test — which is issue 9, and which itself costs measurement budget.
3. *Not defects at all* (5, 6, 7). These are the findings. "Graph RAG cannot
   answer aggregate questions that name no entity" is the result, not a bug to
   remove; making it go away means building hybrid retrieval, which is a
   different system and would end the comparison rather than improve it.

Status key: 🔴 affects published numbers · 🟠 real limitation, measured · 🟡 hygiene / deferred · ✅ resolved

---

## ✅ 1. Evaluation re-executed — RESOLVED

`compare/eval/metrics.json` and `compare/comparison_report.md` are now produced
by a full run of the current code. Retrieval metrics came back **byte-identical**
to the previous run, which independently confirms the 32/32 equivalence proof
that stood in for it while the daily quota was exhausted.

## ✅ 2. Judge no longer shares a model with the generator — RESOLVED

`settings.judge_model` defaults to `openai/gpt-oss-120b`, a different family from
the `qwen/qwen3.8-27b` generator, so neither system is graded by its own
producer. Override with `JUDGE_MODEL`.

Worth recording: once issue 2b below was fixed, the independent judge broadly
**agreed** with the original same-family judge (vector 100% correct, graph
50% / 42.9%). The apparent "stricter judge" effect in the first cross-model run
was almost entirely judge *failures*, not judge strictness.

## ✅ 2b. Judge errors were scored as wrong answers — RESOLVED

`judge_answer` returned `Judgement(correct=False, grounded=False)` on any
exception. In one run **9 of 48 judge calls failed** with Groq JSON-validation
errors, and every single "ungrounded" row in that run was one of those failures
rather than an ungrounded answer — a harness artifact presented as a result.

Now: the judge retries (3 attempts), gets a 1200-token budget instead of 400,
and on persistent failure the row is left **unscored** (`None`) and excluded
from the means. `judge_coverage` reports the scored fraction — currently 100%
for all three systems. Pinned by
`test_unscored_judgement_is_excluded_not_counted_wrong`.

## 🟠 3. Entity canonicalization deliberately leaves ambiguous stubs

`store.canonicalize()` merges `Carol` → `Carol Zhang` but leaves `Stanford`
unmerged, because it prefixes both `Stanford University` and
`Stanford Quantum Initiative` — guessing would fabricate edges. 7 of 156 nodes
merged; `Stanford` (degree 28) and `Stanford University` (degree 23) remain
separate entities describing overlapping things.

This splits provenance across duplicate nodes and costs measurable recall.

**Action:** type-aware or embedding-based entity resolution, evaluated against
the same query set so the gain is measured rather than assumed.

## 🟠 4. Hub dilution destroys graph retrieval on factual queries

`factual_single` hit@4 is 25% (hops=1) and 0% (hops=2) versus 100% for vector.
A factual question seeds a single high-degree node — `Acme Corp` has degree 74 —
whose edges spread across many chunks, so reciprocal-rank fusion over the
retrieved triples ranks hub-adjacent chunks above the chunk that actually
contains the answer. The answer *triple* is often retrieved correctly (judge
correct is 75%); it is the chunk ranking that fails.

**Action:** down-weight edges from high-degree nodes in the RRF scoring, or
weight a chunk by how *specific* its supporting triples are.

## 🟠 5. Aggregate questions have no seed entity

`semantic` hit@4 is 33% for both hop settings. "What open-source projects has
Acme Corp released?" links only to the hub; there is no entity whose
neighbourhood is the answer. Graph retrieval has no mechanism for
"everything of type X".

**Action:** this is the clearest argument for the hybrid design — vector
retrieves candidates, graph expands relations over them.

## 🟠 6. hops=2 is worse than hops=1

recall@4 drops 47.6% → 44.0%, judged correctness 50.0% → 42.9%, at +229 Groq
tokens per query. A fixed budget spent further from the seed buys weaker
evidence. The default is
`hops=1` for this reason, but "more hops = better multi-hop reasoning" is
intuitive and wrong here, and the code does not warn anyone who raises it.

**Action:** either scale the budget with hops, or document the tradeoff at the
CLI flag.

## 🟠 7. Graph abstains on 43–50% of answerable queries

Groundedness is 93.8% at hops=1 (one ungrounded row, q8) and 100% at hops=2, so
the dominant failure mode is refusal rather than fabrication. Still, on q4, q9,
q11, q12, q13 and q14 the evidence was present in the retrieved triples and the
model declined anyway. Some of that is the strict `ANSWER_SYSTEM` prompt
interacting with terse triple syntax.

**Action:** test whether rendering triples as sentences ("Charlie Brown designs
AcmeQ-128") rather than arrow syntax reduces refusals, holding retrieval fixed.

## 🟠 8. Known single-query miss: q5

`Carol Zhang --works_at--> Stanford University` exists in the graph, but with
`Dave Kim` as the only seed it does not fit inside the 2200-char context budget
at either hop setting. Vector answers this correctly. Left as-is deliberately:
tuning the ranker until this one query passes would be fitting to the eval.

## 🟡 9. Small sample, no confidence intervals

One corpus, 33 chunks, 16 queries (14 answerable). Per-type cells have n=3 or
n=4, so a single query flipping moves a type's score by 25–33 points. No
variance estimate, no repeated runs, single random seed.

Related: **latency is not reproducible**. Between two runs of identical code the
average swung from 2.08s to 0.37s for vector, purely on provider load. Treat the
ordering as meaningful and the absolute values as not.

**Action:** more queries per type before treating per-type differences as real.

## 🟠 10. Re-running the eval costs most of the daily API budget — PARTIALLY FIXED

`python app_compare.py eval --no-judge` now skips the judge (rows come back
unscored, `judge_coverage` says so), cutting a run from ~128 Groq calls to ~80.

Still open: answers themselves are not cached, so any rerun re-generates them.
At `qwen/qwen3.8-27b`'s free-tier cap of 1000 requests/day, roughly eight full
evals per day is the ceiling — and this is now the **binding constraint on
everything else in this file**, because any change to retrieval or generation
has to be re-measured before it can be published.

**Action:** cache answers keyed by (query, system, context hash) so only
changed configurations cost anything.

## 🟠 11. RRF weighting is measured and wrong — FIX READY, BLOCKED ON QUOTA

`chunk_ids_from_triples` scores a chunk as `sum(1/(rank+1))`. Swept against the
alternatives over all 28 answerable (query × hops) cases with fixed seeds, so
retrieval is deterministic and no API calls are involved:

| weighting | hit@4 | recall@4 | MRR |
|---|---|---|---|
| `1/(rank+1)` — **current** | 0.679 | 0.530 | 0.482 |
| `1/(60+rank)` — standard RRF | **0.821** | **0.625** | 0.625 |
| `1/(10+rank)` | 0.786 | 0.625 | **0.655** |
| uniform count | 0.750 | 0.554 | 0.589 |
| top-10 triples only | 0.714 | 0.595 | 0.616 |
| first citing triple only | 0.643 | 0.476 | 0.351 |

The current constant is the worst except the degenerate one. `1/(rank+1)` is so
top-heavy that a single high-ranked triple outweighs a chunk supported by many
mid-ranked ones — which is exactly the breadth of evidence a hub-seeded query
depends on, so this also bears directly on issue 4.

Switching to the **published** RRF constant (60) is replacing an ad-hoc value
with the literature default, not fitting to this eval set. It is not applied
because it changes retrieval, and the committed metrics cannot be regenerated
until the daily quota resets (issue 10). Shipping it against stale numbers would
reintroduce exactly the defect this work exists to remove.

**Action:** one line in `store.chunk_ids_from_triples` (a `RRF_K = 60`
constant), then `python app_compare.py eval`, then reground both docs.

## 🟡 12. Query-relevance ranking is lexical only

`get_subgraph`'s `rank()` matches question tokens to relation and target tokens
by 4-character prefix. Cheap and deterministic, but it cannot tell that
"where do they work" relates to `is_based_in`, and it treats `advises` and
`works_at` as equally relevant to a question containing both concepts.

**Action:** embedding similarity between question and rendered triple, measured
against the same query set.

## ✅ 13. `compare/` is a real package — RESOLVED

`compare/`, `compare/vector/`, `compare/graph/` and `compare/eval/` each carry
an `__init__.py` describing the subpackage. Imports no longer depend on the
repo root happening to be on `sys.path`.

## ✅ 14. Misleading embedding-cost column — RESOLVED

`Avg embed tokens/query` reported whatever the NVIDIA API billed *this* run, so
a warm cache showed 0. The column is gone from the summary; embedding cost is
reported once, in the build table, where it is not cache-dependent.

## ✅ 15. Lint clean — RESOLVED

The two `ARG002 Unused method argument: run_manager` findings are the
`BaseRetriever` contract, not dead parameters. `pyproject.toml` now carries a
`[tool.ruff.lint.per-file-ignores]` entry naming both files and the reason.
`ruff check --select F,B,ARG` is clean.

## ✅ 16. Compare pipelines are tested offline — RESOLVED

`tests/test_graph_rag.py` (28 tests) covers the store, retriever, ingestion,
metrics and query set, but nothing exercises `run_eval`, `build_vector_store` or
`build_graph_throttled`, because all three need live API calls.

**Action:** a fake embedding client and a stub LLM would make the whole harness
testable offline, and would have caught the double-retrieval bug directly.

## ✅ 17. `uv.lock` — RESOLVED, kept

Validated against `pyproject.toml` with `uv lock --check` (resolves cleanly) and
committed. The documented setup (`uv pip install -e .`) ignores it, but keeping
it costs nothing and pins a reproducible resolution for anyone using
`uv sync`.
