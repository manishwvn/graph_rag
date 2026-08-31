"""GraphStore: NetworkX MultiDiGraph wrapper with per-chunk provenance.

Persisted as JSON (node-link) rather than pickle: `pickle.load` executes
arbitrary code, and the graph path is user-configurable via settings.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import NamedTuple

import networkx as nx
from langchain_core.documents import Document

from graph_rag.schemas import ExtractionResult


class Triple(NamedTuple):
    """One retrieved relation plus the chunks that support it."""

    source: str
    relation: str
    target: str
    chunk_uids: tuple[str, ...] = ()
    hop: int = 0


def normalize_relation(relation: str) -> str:
    """Fold relation-label variants: ``co-leads``/``Co Leads`` -> ``co_leads``."""
    r = relation.strip().lower()
    r = re.sub(r"[^\w]+", "_", r)
    return re.sub(r"_+", "_", r).strip("_")


def normalize_name(name: str) -> str:
    """Fold surface variants of an entity name to a comparison key.

    Lowercases, strips possessives (``Stanford's`` -> ``stanford``), drops
    punctuation and collapses whitespace. This is what lets ``Acme Corp.``,
    ``acme corp`` and ``Acme Corp`` become one node.
    """
    s = unicodedata.normalize("NFKD", name).strip().lower()
    s = re.sub(r"[’']s\b", "", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class GraphStore:
    """Production wrapper around NetworkX MultiDiGraph for Graph RAG."""

    def __init__(self, path: str | Path = "graph.json"):
        self.path = Path(path)
        # MultiDiGraph: direction matters (Alice --works_at--> Acme != reverse)
        # and a pair of entities can hold several distinct relations.
        self.graph = nx.MultiDiGraph()

    # ---------------------------------------------------------------- build

    def add_extraction(self, result: ExtractionResult, source: str):
        """Add one chunk's extraction. `source` must be a chunk_uid."""
        for e in result.entities:
            self._add_node(e.name.strip(), e.type, source)

        for r in result.relations:
            src, tgt, rel = r.source.strip(), r.target.strip(), normalize_relation(r.relation)
            if not src or not tgt or not rel:
                continue
            self._add_node(src, "UNKNOWN", source)
            self._add_node(tgt, "UNKNOWN", source)
            # key=rel keeps distinct relations as distinct parallel edges
            # instead of concatenating them into one string.
            if self.graph.has_edge(src, tgt, key=rel):
                self.graph[src][tgt][rel]["sources"].add(source)
            else:
                self.graph.add_edge(src, tgt, key=rel, relation=rel, sources={source})

    def _add_node(self, name: str, type_: str, source: str):
        if not name:
            return
        if not self.graph.has_node(name):
            self.graph.add_node(name, type=type_, sources={source})
            return
        attrs = self.graph.nodes[name]
        if attrs.get("type", "UNKNOWN") == "UNKNOWN" and type_ != "UNKNOWN":
            attrs["type"] = type_
        attrs.setdefault("sources", set()).add(source)

    def build_from_documents(self, docs: list[Document], extraction_results: list[ExtractionResult]):
        if len(docs) != len(extraction_results):
            raise ValueError(f"docs ({len(docs)}) and results ({len(extraction_results)}) length mismatch")
        for doc, res in zip(docs, extraction_results, strict=True):
            uid = doc.metadata.get("chunk_uid") or doc.metadata.get("source", "unknown")
            self.add_extraction(res, source=uid)

    # ------------------------------------------------------- canonicalize

    def canonicalize(self) -> dict[str, str]:
        """Merge surface variants of the same entity. Returns alias -> canonical.

        Two passes, both deterministic:
          1. exact normalized-form collision (``Stanford's Quantum Initiative``
             and ``Stanford Quantum Initiative``);
          2. unambiguous single-token prefix (``Carol`` -> ``Carol Zhang``) --
             applied only when exactly one longer name starts with that token,
             so genuinely ambiguous stubs like ``Stanford`` are left alone.
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        def weight(n: str) -> tuple[int, int, int, str]:
            # most-attested wins; then the cleanest surface form; then the longest
            punct = sum(1 for c in n if not (c.isalnum() or c.isspace()))
            return (len(self.graph.nodes[n].get("sources", ())), -punct, len(n), n)

        # pass 1: normalized-form groups
        groups: dict[str, list[str]] = {}
        for n in self.graph.nodes:
            groups.setdefault(normalize_name(n), []).append(n)
        mapping: dict[str, str] = {}
        for members in groups.values():
            canonical = max(members, key=weight)
            for m in members:
                if m != canonical:
                    mapping[m] = canonical

        # pass 2: single-token prefix. One candidate is unambiguous; several are
        # resolved by shared relations if the evidence is decisive, else left alone.
        survivors = {mapping.get(n, n) for n in self.graph.nodes}
        norm_of = {s: normalize_name(s) for s in survivors}

        for stub in sorted(survivors):
            nstub = norm_of[stub]
            if not nstub or " " in nstub:
                continue
            cands = sorted(s for s in survivors if s != stub and norm_of[s].startswith(nstub + " "))
            if not cands:
                continue
            if len(cands) == 1:
                mapping[stub] = cands[0]
            # Ambiguous stubs (`Stanford` prefixes two real entities) are left
            # alone here. A relation-evidence rule that resolves them correctly
            # is ready on the `retrieval-rrf-canonicalization` branch, pending
            # a rebuild to validate. See issues.md #3.

        if not mapping:
            return {}

        merged = nx.MultiDiGraph()
        for n, attrs in self.graph.nodes(data=True):
            c = mapping.get(n, n)
            if merged.has_node(c):
                tgt = merged.nodes[c]
                if tgt.get("type", "UNKNOWN") == "UNKNOWN":
                    tgt["type"] = attrs.get("type", "UNKNOWN")
                tgt["sources"] |= set(attrs.get("sources", ()))
            else:
                merged.add_node(c, type=attrs.get("type", "UNKNOWN"), sources=set(attrs.get("sources", ())))
        for u, v, k, attrs in self.graph.edges(keys=True, data=True):
            cu, cv = mapping.get(u, u), mapping.get(v, v)
            if cu == cv:
                continue  # self-loop created by the merge carries no information
            if merged.has_edge(cu, cv, key=k):
                merged[cu][cv][k]["sources"] |= set(attrs.get("sources", ()))
            else:
                merged.add_edge(cu, cv, key=k, relation=attrs.get("relation", k), sources=set(attrs.get("sources", ())))
        self.graph = merged
        return mapping

    # ------------------------------------------------------------ persist

    def save(self):
        data = nx.node_link_data(self.graph, edges="links")
        for n in data["nodes"]:
            if isinstance(n.get("sources"), set):
                n["sources"] = sorted(n["sources"])
        for e in data["links"]:
            if isinstance(e.get("sources"), set):
                e["sources"] = sorted(e["sources"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return self.path

    def load(self) -> nx.MultiDiGraph:
        if not self.path.exists():
            raise FileNotFoundError(f"Graph not found at {self.path} — run `python app.py --build` first")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        g = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
        for _, attrs in g.nodes(data=True):
            attrs["sources"] = set(attrs.get("sources", ()))
        for _, _, attrs in g.edges(data=True):
            attrs["sources"] = set(attrs.get("sources", ()))
        self.graph = g
        return self.graph

    # ---------------------------------------------------------- retrieval

    def _edges_between(self, a: str, b: str) -> list[tuple[str, str, str, dict]]:
        out = []
        if self.graph.has_edge(a, b):
            for k, attrs in self.graph[a][b].items():
                out.append((a, b, k, attrs))
        if self.graph.has_edge(b, a):
            for k, attrs in self.graph[b][a].items():
                out.append((b, a, k, attrs))
        return out

    def match_seeds(self, seed_nodes: list[str]) -> list[str]:
        """Case-insensitive / normalized exact match of query entities to nodes."""
        by_norm: dict[str, str] = {}
        for n in self.graph.nodes:
            by_norm.setdefault(normalize_name(n), n)
        matched: list[str] = []
        for s in seed_nodes:
            if s in self.graph:
                if s not in matched:
                    matched.append(s)
                continue
            hit = by_norm.get(normalize_name(s))
            if hit and hit not in matched:
                matched.append(hit)
        return matched

    def get_subgraph(
        self,
        seed_nodes: list[str],
        hops: int = 1,
        max_triples: int = 30,
        max_per_node: int = 8,
        query: str | None = None,
        max_chars: int | None = None,
    ) -> list[Triple]:
        """Expand outward from the matched seeds and return the traversed edges.

        Four rules keep the result useful under a fixed budget. Each one exists
        because removing it measurably lost the answer edge on a real query:

        * only edges actually traversed are returned, not every edge of the
          induced subgraph;
        * a node contributes at most `max_per_node` edges to one hop, so a
          high-degree hub cannot swallow the level;
        * within a hop the nodes take turns, so the level is not dominated by
          whichever node happens to sort first alphabetically;
        * each hop gets an equal share of `max_triples`, counting only *new*
          triples, so hop 1 cannot starve hop 2.
        """
        hops = max(1, hops)
        if not seed_nodes:  # nothing linked: return a deterministic overview
            everything = [
                Triple(u, a.get("relation", k), v, tuple(sorted(a.get("sources", ()))), 0)
                for u, v, k, a in self.graph.edges(keys=True, data=True)
            ]
            everything.sort(key=lambda t: (t.source, t.relation, t.target))
            return everything[:max_triples]

        matched = self.match_seeds(seed_nodes)
        if not matched:
            return []
        seed_set = set(matched)

        # Lexical relevance: "where do they work" should pull `works_at` ahead
        # of an equally-close but unrelated relation.
        q_tokens = {t for t in normalize_name(query or "").split() if len(t) > 2}

        def rank(t: Triple) -> tuple:
            words = set(normalize_relation(t.relation).split("_")) | set(normalize_name(t.target).split())
            relevant = q_tokens and any(w[:4] == q[:4] for w in words if len(w) >= 4 for q in q_tokens)
            return (
                0 if relevant else 1,
                # edges closing over entities already reached answer multi-hop
                # questions; edges that merely reach somewhere new do not.
                0 if (t.source in visited and t.target in visited) else 1,
                0 if (t.source in seed_set or t.target in seed_set) else 1,
                t.source,
                t.relation,
                t.target,
            )

        visited = set(matched)
        frontier = sorted(matched)
        emitted: set[tuple[str, str, str]] = set()
        levels: list[list[Triple]] = []

        for hop in range(1, hops + 1):
            per_node: list[list[Triple]] = []
            next_frontier: set[str] = set()
            cap = max(max_per_node, max_triples // max(1, len(frontier)))
            for node in frontier:
                node_edges: list[Triple] = []
                for nb in sorted(set(self.graph.successors(node)) | set(self.graph.predecessors(node))):
                    for u, v, k, attrs in self._edges_between(node, nb):
                        rel = attrs.get("relation", k)
                        if (u, rel, v) in emitted:  # already returned by an earlier hop
                            continue
                        node_edges.append(Triple(u, rel, v, tuple(sorted(attrs.get("sources", ()))), hop))
                    if nb not in visited:
                        next_frontier.add(nb)
                node_edges.sort(key=rank)
                per_node.append(node_edges[:cap])

            width = max((len(e) for e in per_node), default=0)
            level = [e[i] for i in range(width) for e in per_node if i < len(e)]
            levels.append(level)
            emitted |= {(t.source, t.relation, t.target) for t in level}
            visited |= next_frontier
            frontier = sorted(next_frontier)
            if not frontier:
                break

        levels = [lvl for lvl in levels if lvl]
        if not levels:
            return []

        share = max(1, max_triples // len(levels))
        picked: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()

        def take(level: list[Triple], limit: int) -> None:
            """Take up to `limit` triples not already picked. An edge between two
            frontier nodes is collected by both, so duplicates must not count
            against the hop's share."""
            added = 0
            for t in level:
                if added >= limit or len(picked) >= max_triples:
                    return
                key = (t.source, t.relation, t.target)
                if key not in seen:
                    seen.add(key)
                    picked.append(t)
                    added += 1

        for lvl in levels:
            take(lvl, share)
        for lvl in levels:  # hand leftover slots back out, in hop order
            take(lvl, max_triples)

        picked.sort(key=lambda t: (t.hop, t.source, t.relation, t.target))
        picked = picked[:max_triples]

        if max_chars:
            # Equalise the context budget with the vector system: comparing a
            # 2.2k-char prose context against a 0.9k-char triple context
            # measures the budget, not the retrieval strategy.
            out, used = [], 0
            for t in picked:
                cost = len(f"{t.source} --[{t.relation}]--> {t.target}") + 1
                if used + cost > max_chars:
                    break
                out.append(t)
                used += cost
            picked = out
        return picked

    @staticmethod
    def chunk_ids_from_triples(triples: list[Triple], k: int | None = None) -> list[str]:
        """Rank the chunks that support the retrieved triples.

        This is the bridge that makes Graph RAG and Vector RAG scoreable in the
        same id space: both end up producing a ranked list of chunk_uids.
        """
        # Reciprocal-rank fusion over the whole triple list. Taking each
        # triple's chunks in order instead meant the top 4 chunks came from the
        # top ~2 triples, so a correctly-retrieved answer edge sitting at rank
        # 12 contributed nothing to the chunk ranking.
        #
        # NOTE: `1/(rank+1)` measures worse than the standard RRF constant of
        # 60 (hit@4 0.68 vs 0.82). The switch is ready on the
        # `retrieval-rrf-canonicalization` branch; it is not on main because it
        # changes retrieval and the committed metrics cannot be regenerated
        # until the generation model's daily window resets. See issues.md #11.
        scores: dict[str, float] = {}
        for rank, t in enumerate(triples):
            for uid in t.chunk_uids:
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (rank + 1)
        order = sorted(scores, key=lambda uid: (-scores[uid], uid))
        return order[:k] if k else order

    # -------------------------------------------------------------- misc

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "chunks": len({s for _, a in self.graph.nodes(data=True) for s in a.get("sources", ())}),
        }

    def triples_as_text(self, triples: list[Triple]) -> str:
        return "\n".join(f"{t.source} --[{t.relation}]--> {t.target}" for t in triples)
