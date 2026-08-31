# Vector vs Graph RAG — Comparison Report

Generated from `compare/eval/metrics.json`. Every number below is read from that file.

## Setup

| Setting | Value |
|---|---|
| Corpus chunks | 33 (chunk_size=800, overlap=80) |
| Graph | 149 nodes, 347 edges, provenance over 33 chunks |
| Retrieval budget | k=4 chunks per query, max_triples=60 |
| Generation model | `qwen/qwen3.8-27b` (identical prompt for both systems) |
| Embedding model | `nvidia/nemotron-3-embed-1b` |
| Queries | 16 (14 answerable, 2 negative) |

Both systems return a ranked list of `chunk_uid`s, so retrieval metrics are measured in one id space; both are given the same generation model, the same prompt and the same context budget (see *Avg context chars*). Retrieval metrics average over answerable queries only; the negative queries are scored on abstention instead. Embedding cost is reported in the build table below rather than per query, because query embeddings are cached and a repeat run would bill zero.

## Summary

| Metric | Vector (k=4) | Graph hops=1 | Graph hops=2 | Best |
|---|---|---|---|---|
| Hit@4 | 100.0% | 64.3% | 57.1% | vector |
| Precision@4 | 33.9% | 23.2% | 19.6% | vector |
| Recall@4 | 83.3% | 47.6% | 44.0% | vector |
| MRR | 0.839 | 0.393 | 0.369 | vector |
| Answer keyword recall | 100.0% | 57.1% | 57.1% | vector |
| Judge: correct (answerable) | 100.0% | 50.0% | 42.9% | vector |
| Judge: grounded (all) | 100.0% | 93.8% | 100.0% | vector, graph_hops2 |
| Judge coverage (rows scored) | 100.0% | 100.0% | 100.0% | vector, graph_hops1, graph_hops2 |
| Abstained on answerable | 0.0% | 42.9% | 50.0% | vector |
| Abstained on negative | 100.0% | 100.0% | 100.0% | vector, graph_hops1, graph_hops2 |
| Avg context chars | 2196 | 1474 | 2205 | graph_hops2 |
| Avg latency (s) | 0.37 | 0.57 | 0.67 | vector |
| Avg Groq tokens/query | 552 | 610 | 838 | vector |

## Build cost

| | Vector | Graph |
|---|---|---|
| Chunks indexed | 33/33 | 33/33 |
| API calls | 5 (NVIDIA embed) | 33 (Groq extract) |
| Wall clock | 8.2s | 175.3s |
| Index | 2.25 MB ChromaDB | 149 nodes / 347 edges JSON |
| Canonicalization | — | 7 aliases merged (156 → 149 nodes) |

## By query type

| Type | n | Metric | Vector (k=4) | Graph hops=1 | Graph hops=2 |
|---|---|---|---|---|---|
| factual_single | 4 | hit@4 | 100.0% | 25.0% | 0.0% |
|  |  | judge correct | 100.0% | 75.0% | 75.0% |
|  |  | judge grounded | 100.0% | 100.0% | 100.0% |
| multi_hop | 4 | hit@4 | 100.0% | 100.0% | 100.0% |
|  |  | judge correct | 100.0% | 75.0% | 50.0% |
|  |  | judge grounded | 100.0% | 75.0% | 100.0% |
| relationship | 3 | hit@4 | 100.0% | 100.0% | 100.0% |
|  |  | judge correct | 100.0% | 33.3% | 33.3% |
|  |  | judge grounded | 100.0% | 100.0% | 100.0% |
| semantic | 3 | hit@4 | 100.0% | 33.3% | 33.3% |
|  |  | judge correct | 100.0% | 0.0% | 0.0% |
|  |  | judge grounded | 100.0% | 100.0% | 100.0% |
| negative | 2 | judge correct | 100.0% | 100.0% | 100.0% |
|  |  | judge grounded | 100.0% | 100.0% | 100.0% |

## Per query

| id | type | Vector (k=4) hit@4 / correct | Graph hops=1 hit@4 / correct | Graph hops=2 hit@4 / correct |
|---|---|---|---|---|
| q1 | factual_single | 100.0% / ✅ | 0.0% / ✅ | 0.0% / ✅ |
| q2 | factual_single | 100.0% / ✅ | 100.0% / ✅ | 0.0% / ✅ |
| q3 | factual_single | 100.0% / ✅ | 0.0% / ✅ | 0.0% / ✅ |
| q4 | factual_single | 100.0% / ✅ | 0.0% / ❌ | 0.0% / ❌ |
| q5 | multi_hop | 100.0% / ✅ | 100.0% / ❌ | 100.0% / ❌ |
| q6 | multi_hop | 100.0% / ✅ | 100.0% / ✅ | 100.0% / ✅ |
| q7 | multi_hop | 100.0% / ✅ | 100.0% / ✅ | 100.0% / ❌ |
| q8 | multi_hop | 100.0% / ✅ | 100.0% / ✅ | 100.0% / ✅ |
| q9 | relationship | 100.0% / ✅ | 100.0% / ❌ | 100.0% / ❌ |
| q10 | relationship | 100.0% / ✅ | 100.0% / ✅ | 100.0% / ✅ |
| q11 | relationship | 100.0% / ✅ | 100.0% / ❌ | 100.0% / ❌ |
| q12 | semantic | 100.0% / ✅ | 100.0% / ❌ | 100.0% / ❌ |
| q13 | semantic | 100.0% / ✅ | 0.0% / ❌ | 0.0% / ❌ |
| q14 | semantic | 100.0% / ✅ | 0.0% / ❌ | 0.0% / ❌ |
| q15 | negative | n/a / ✅ | n/a / ✅ | n/a / ✅ |
| q16 | negative | n/a / ✅ | n/a / ✅ | n/a / ✅ |

## Where the systems disagree

- **q4** (factual_single) — correct: vector; wrong: graph_hops1, graph_hops2. Judge on graph_hops1: _The system answer does not provide the reference fact (AcmeQ-128), so it is not correct. It makes no factual claims, thus it is grounded. It explicitly declines to answer, so abstained is true._
- **q5** (multi_hop) — correct: vector; wrong: graph_hops1, graph_hops2. Judge on graph_hops1: _The system correctly identified Carol Zhang as the advisor, which is supported by the context. However, it incorrectly claimed that Carol Zhang's workplace is not specified, whereas the reference answer states she works at Stanford University. All factual statements made by the system are supported by the context, so it is grounded, and it did not abstain._
- **q7** (multi_hop) — correct: vector, graph_hops1; wrong: graph_hops2. Judge on graph_hops2: _The reference answer lists Dave Kim, Frank Liu, Grace Wang, and Henry Zhao as the PhD students advised by Carol Zhang. The system answer ultimately states that only Dave Kim is explicitly identified as a PhD student, contradicting the reference answer. While all factual statements made by the system are supported by the provided context, the answer does not match the reference answer, so it is not correct._
- **q9** (relationship) — correct: vector; wrong: graph_hops1, graph_hops2. Judge on graph_hops1: _The system answer does not provide the reference answer (Alice Chen and Carol Zhang), so it is not correct. It makes no factual claim, thus it is grounded. It explicitly declines to answer, so abstained is true._
- **q11** (relationship) — correct: vector; wrong: graph_hops1, graph_hops2. Judge on graph_hops1: _The system answer does not provide the reference answer (Charlie Brown) and instead says it does not know, so it is not correct. It makes no factual claims, thus it is grounded. It declined to answer, so abstained is true._
- **q12** (semantic) — correct: vector; wrong: graph_hops1, graph_hops2. Judge on graph_hops1: _The system answer does not provide the required information ($25 million over five years), so it is not correct. It makes no factual claim, thus it is grounded. It explicitly declines to answer, so abstained is true._
- **q13** (semantic) — correct: vector; wrong: graph_hops1, graph_hops2. Judge on graph_hops1: _The system answer "I don't know based on the provided context." does not state the reference answer's facts (consensus protocols, fault tolerance, edge computing), so it is not correct. It is a refusal/abstention, which contains no factual claims and is therefore grounded, and it counts as an abstention._
- **q14** (semantic) — correct: vector; wrong: graph_hops1, graph_hops2. Judge on graph_hops1: _The system answer does not provide the listed open‑source projects, so it is not correct. It makes no factual claims, thus it is grounded, and it explicitly declines to answer, so abstained is true._

## Findings

- **Hit@4** — best: vector (Vector (k=4) 100.0%, Graph hops=1 64.3%, Graph hops=2 57.1%)
- **Precision@4** — best: vector (Vector (k=4) 33.9%, Graph hops=1 23.2%, Graph hops=2 19.6%)
- **Recall@4** — best: vector (Vector (k=4) 83.3%, Graph hops=1 47.6%, Graph hops=2 44.0%)
- **MRR** — best: vector (Vector (k=4) 0.839, Graph hops=1 0.393, Graph hops=2 0.369)
- **Answer keyword recall** — best: vector (Vector (k=4) 100.0%, Graph hops=1 57.1%, Graph hops=2 57.1%)
- **Judge: correct (answerable)** — best: vector (Vector (k=4) 100.0%, Graph hops=1 50.0%, Graph hops=2 42.9%)
- **Judge: grounded (all)** — best: vector, graph_hops2 (Vector (k=4) 100.0%, Graph hops=1 93.8%, Graph hops=2 100.0%)
- **Judge coverage (rows scored)** — best: vector, graph_hops1, graph_hops2 (Vector (k=4) 100.0%, Graph hops=1 100.0%, Graph hops=2 100.0%)
- **Abstained on answerable** — best: vector (Vector (k=4) 0.0%, Graph hops=1 42.9%, Graph hops=2 50.0%)
- **Abstained on negative** — best: vector, graph_hops1, graph_hops2 (Vector (k=4) 100.0%, Graph hops=1 100.0%, Graph hops=2 100.0%)
- **Avg context chars** — best: graph_hops2 (Vector (k=4) 2196, Graph hops=1 1474, Graph hops=2 2205)
- **Avg latency (s)** — best: vector (Vector (k=4) 0.37, Graph hops=1 0.57, Graph hops=2 0.67)
- **Avg Groq tokens/query** — best: vector (Vector (k=4) 552, Graph hops=1 610, Graph hops=2 838)

- Going from hops=1 to hops=2 degrades recall@4 by -3.6 points (47.6% → 44.0%) at 0.10s extra latency.
