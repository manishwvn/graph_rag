# Open Issues

Known problems, deferred work and methodological caveats. Nothing here is a
blocker for the current results — but everything here is a reason to read those
results with a specific caveat in mind.

**What is actually blocking the rest.** The remaining items fall into three
groups, and only the first is ordinary work:

1. *Blocked on measurement budget* (3, 7, 11). Also the dev set in issue 9,
   which is written and validated but has never been run. The binding limit is tokens, not
   requests: the free tier gives 200k tokens/day per model and one eval spends
   54k on the judge alone, so **three full evals per day**. Any change to
   retrieval or generation must be re-measured before it can be published.
   Issue 11 has a measured, ready-to-apply fix waiting on exactly this.
2. *Needs the dev set to be run* (4, 8, 12). All ranking changes. A variant
   chosen because it scores better on the reported queries is fitted to them.
   The dev/test split in issue 9 removes that objection — tune on dev, report on
   test — but the dev set still has to be evaluated once.

**What is deliberately not in this file.** Findings are not issues. "Graph RAG
cannot answer aggregate questions that name no entity", "hops=2 is worse than
hops=1" and "the graph refuses rather than fabricates" are results, reported in
`README.md` and `LEARNINGS.md`. They were listed here at first, which made the
backlog look longer than it was and buried the one genuine defect hiding among
them: `app_compare.py` defaulted `--hops` to the setting that measures worse.
That is fixed; the observations stay in the docs.

Status key: 🔴 affects published numbers · 🟠 real limitation, measured · 🟡 hygiene / deferred · 🔬 open experiment · ✅ resolved

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

## 🟠 3. Ambiguous entity stubs — FIX MEASURED, ON A BRANCH

`store.canonicalize()` merges `Carol` → `Carol Zhang` but leaves `Stanford`
unmerged, because it prefixes both `Stanford University` and
`Stanford Quantum Initiative` — guessing would fabricate edges. 7 of 156 nodes
merged; `Stanford` (degree 28) and `Stanford University` (degree 23) remain
separate entities describing overlapping things.

This splits provenance across duplicate nodes and costs measurable recall.

**Resolved in principle.** Neighbour overlap is *not* a usable signal — it ranks
`Acme` closer to `Acme Storage` (0.13) than to `Acme Corp` (0.09), so merging on
it would fabricate entities. A stronger signal does separate them: two nodes in
the **same relation to the same third entity** are very likely the same entity.

| stub | candidate | shared (direction, relation, other) |
|---|---|---|
| `Acme` | **`Acme Corp`** | **2** |
| `Acme` | `Acme Storage`, `Acme's toolchain`, … | 0 |
| `Stanford` | **`Stanford University`** | **2** |
| `Stanford` | `Stanford Quantum Initiative`, … | 0 |

Implemented as a third canonicalization pass (149 → 147 nodes, merging exactly
`Acme`→`Acme Corp` and `Stanford`→`Stanford University`, leaving every wrong
candidate intact). On its own it does not move hit@4 or recall@4 and slightly
lowers MRR; combined with the RRF fix in issue 11 it gives the best MRR measured
(0.643).

It sits on the `retrieval-rrf-canonicalization` branch rather than main, because
validating it needs a graph rebuild and the generation model's daily window is
exhausted.

## 🟠 4. Hub dilution destroys graph retrieval on factual queries

`factual_single` hit@4 is 25% (hops=1) and 0% (hops=2) versus 100% for vector.
A factual question seeds a single high-degree node — `Acme Corp` has degree 74 —
whose edges spread across many chunks, so reciprocal-rank fusion over the
retrieved triples ranks hub-adjacent chunks above the chunk that actually
contains the answer. The answer *triple* is often retrieved correctly (judge
correct is 75%); it is the chunk ranking that fails.

**Action:** down-weight edges from high-degree nodes in the RRF scoring, or
weight a chunk by how *specific* its supporting triples are.

## ✅ 6. hops=2 tradeoff is now stated where it is chosen — RESOLVED

The measurement (recall@4 −3.6 points, judged correctness −7.1) is a finding and
lives in `README.md`. What was actually a defect here is fixed: `app.py --hops`
now states the tradeoff in its help text, and `app_compare.py` no longer
defaults `--hops` to 2 — the setting that measures worse was the default for
every `query` and `query-graph` invocation.

## 🔬 7. Untested hypothesis: does triple *rendering* cause the refusals?

That the graph abstains on 43–50% of answerable queries is a finding, and it is
reported in `README.md`. The open item is narrower and is an experiment, not a
defect: on q4, q9, q11, q12, q13 and q14 the evidence **was** present in the
retrieved triples and the model declined anyway. The suspicion is that the
strict `ANSWER_SYSTEM` prompt interacts badly with arrow syntax
(`Charlie Brown --[designs]--> AcmeQ-128`).

**Experiment:** render triples as sentences ("Charlie Brown designs AcmeQ-128"),
hold retrieval completely fixed, and compare abstention. Retrieval being
unchanged means chunk-level metrics must not move — if they do, the experiment
is wrong. Costs one eval; the judge cache will not help, because the context
string changes.

## 🟠 8. Known single-query miss: q5

`Carol Zhang --works_at--> Stanford University` exists in the graph, but with
`Dave Kim` as the only seed it does not fit inside the 2200-char context budget
at either hop setting. Vector answers this correctly. Left as-is deliberately:
tuning the ranker until this one query passes would be fitting to the eval.

## 🟠 9. Small sample — DEV/TEST SPLIT ADDED, evaluation pending

One corpus, 33 chunks. Per-type cells have n=3 or n=4, so a single query
flipping moves a type's score by 25–33 points. No variance estimate, no repeated
runs.

**Half of this is fixed.** The methodological problem was not only the sample
size, it was that there was *one* set, so any ranking variant chosen because it
scored better was fitted to the very queries used to report it. There are now
two disjoint sets over the same corpus:

| file | n | answerable | role |
|---|---|---|---|
| `compare/data_large/queries.json` | 16 | 14 | **test** — held out, what the published metrics measure |
| `compare/data_large/queries_dev.json` | 16 | 14 | **dev** — tune ranking here |

Gold labels for both are derived from chunk contents, capped at 4 chunks each,
and every answer keyword is asserted to appear in a gold chunk. Disjointness of
ids and question text is pinned by `test_dev_and_test_query_sets_are_disjoint`.

Tune with `python app_compare.py eval --queries compare/data_large/queries_dev.json`,
then report on the test set. This is what unblocks issues 4, 8 and 12.

Still open: the dev set has never been run (it costs an eval), and neither set is
large enough for confidence intervals.

Related: **latency is not reproducible**. Between two runs of identical code the
average swung from 2.08s to 0.37s for vector, purely on provider load. Treat the
ordering as meaningful and the absolute values as not.

**Action:** more queries per type before treating per-type differences as real.

## ✅ 10. Eval throughput — RESOLVED

The original diagnosis ("we are hitting requests-per-day") was the symptom, not
the cause. Measured against the real free-tier limits — per model: 30 RPM,
1000 RPD, 8000 TPM, 200k TPD:

| | rate during an unpaced eval | limit | over by |
|---|---|---|---|
| judge tokens | 16,865 / min | 8,000 TPM | **2.1×** |
| generation tokens | 10,000 / min | 8,000 TPM | **1.2×** |
| generation requests | 25 / min | 30 RPM | under |

Requests were never the binding limit. Tokens were. And because a token-limit
429 was simply retried, **each retry still consumed one of the 1000 daily
requests** — which is how a run that nominally needs 128 requests exhausted the
daily allowance. The fixed 2-second sleeps paced the wrong quantity.

Fixed:

* `RateLimiter` in `compare/eval/harness.py` paces on a rolling 60-second
  window of both requests and tokens, one limiter per model, at 85% headroom.
* `run_with_retry` reads the provider's own wait out of the 429 body
  (`"try again in 1m26.4s"`); guessing 3 seconds for a 60-second token window
  just spent another daily request.
* `check-quotas` now reports the token-bound estimate from measured usage
  instead of a request count.

The honest ceiling is **3 full evals per day**, bound by the judge model's
200k tokens/day — not the ~8 the request count suggested. A paced run takes
~11 minutes rather than ~3.

Judge verdicts are now cached (`compare/eval/judge_cache.json`, gitignored),
keyed by model + question + reference answer + context + answer. Judging is 54k
of an eval's 86k tokens and is a pure function of its inputs, so re-running an
unchanged configuration now costs nothing on the judge model. The graph build is
paced too — 33 extractions in ~82s was ~24k tokens/min against the same 8000
TPM cap.

Generated answers are cached too (`compare/eval/answer_cache.json`, gitignored).
An answer depends on the indexes, not just the question, so the key carries a
fingerprint of the graph file, the vector collection's chunk ids and the
generation model; changing any of them invalidates the whole file at once, which
is what stops a stale answer surviving a rebuild. Cached rows are excluded from
the latency mean — a cache hit takes 0.0s and averaging that in would report a
latency the system never achieved.

Re-running an unchanged configuration now costs no Groq tokens at all.

**Note on the daily window.** It is a rolling 24h window, not a calendar day, and
the API's counter is authoritative: the console showed 189 requests used for
`qwen/qwen3.8-27b` while `x-ratelimit-remaining-requests` reported 0 of 1000
with 24h to reset. Read the headers, not the dashboard, when deciding whether a
run will fit.

## 🟠 11. RRF weighting is measured and wrong — FIX ON A BRANCH, BLOCKED ON QUOTA

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

Applied on the `retrieval-rrf-canonicalization` branch together with issue 3.
Combined offline result over the same 28 cases:

| config | hit@4 | recall@4 | MRR |
|---|---|---|---|
| main, as shipped | 0.679 | 0.530 | 0.482 |
| + canonicalization (issue 3) | 0.679 | 0.530 | 0.458 |
| + RRF k=60 | 0.821 | 0.625 | 0.625 |
| **both** | **0.821** | **0.625** | **0.643** |

**Action:** when the generation model's window resets — `git merge
retrieval-rrf-canonicalization`, `python app_compare.py build-graph`,
`python app_compare.py eval`, then reground both docs. The judge cache will not
help here: changed retrieval means changed context, so every verdict is a miss.

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
