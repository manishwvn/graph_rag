"""GraphStore: NetworkX DiGraph wrapper. Uses pickle for minimal demo; for prod replace with JSON/GraphML."""

import pickle
from pathlib import Path
import networkx as nx
from graph_rag.schemas import ExtractionResult
from langchain_core.documents import Document


class GraphStore:
    """Production wrapper around NetworkX DiGraph for Graph RAG."""

    def __init__(self, path: str | Path = "graph.gpickle"):
        self.path = Path(path)
        # DiGraph preserves direction: Alice --works_at--> Acme != Acme -> Alice
        self.graph = nx.DiGraph()

    def add_extraction(self, result: ExtractionResult, source: str):
        for e in result.entities:
            name = e.name.strip()
            if not self.graph.has_node(name):
                self.graph.add_node(name, type=e.type, sources={source})
            else:
                if self.graph.nodes[name].get("type") == "UNKNOWN" and e.type != "UNKNOWN":
                    self.graph.nodes[name]["type"] = e.type
                self.graph.nodes[name].setdefault("sources", set()).add(source)

        for r in result.relations:
            src, tgt, rel = r.source.strip(), r.target.strip(), r.relation.strip()
            for n, tp in [(src, "UNKNOWN"), (tgt, "UNKNOWN")]:
                if not self.graph.has_node(n):
                    self.graph.add_node(n, type=tp, sources={source})
            if self.graph.has_edge(src, tgt):
                existing = self.graph[src][tgt].get("relation", "")
                if rel not in existing:
                    self.graph[src][tgt]["relation"] = f"{existing}, {rel}" if existing else rel
            else:
                self.graph.add_edge(src, tgt, relation=rel)

    def build_from_documents(self, docs: list[Document], extraction_results: list[ExtractionResult]):
        if len(docs) != len(extraction_results):
            raise ValueError(f"docs ({len(docs)}) and results ({len(extraction_results)}) length mismatch")
        for doc, res in zip(docs, extraction_results, strict=True):
            self.add_extraction(res, source=doc.metadata.get("source", "unknown"))

    def save(self):
        # pickle needs serializable sources; keep in-memory as set
        g = self.graph.copy()
        for n in g.nodes:
            if isinstance(g.nodes[n].get("sources"), set):
                g.nodes[n]["sources"] = list(g.nodes[n]["sources"])
        with open(self.path, "wb") as f:
            pickle.dump(g, f)
        return self.path

    def load(self) -> nx.DiGraph:
        if not self.path.exists():
            raise FileNotFoundError(f"Graph not found at {self.path} — run `python app.py --build` first")
        with open(self.path, "rb") as f:
            g = pickle.load(f)
        for n in g.nodes:
            if isinstance(g.nodes[n].get("sources"), list):
                g.nodes[n]["sources"] = set(g.nodes[n]["sources"])
        # ensure DiGraph type (old pickle may be Graph)
        if not isinstance(g, nx.DiGraph):
            g = nx.DiGraph(g)
        self.graph = g
        return self.graph

    def get_subgraph(self, seed_nodes: list[str], hops: int = 1, max_triples: int = 30) -> list[tuple[str, str, str]]:
        if not seed_nodes:
            edges = list(self.graph.edges(data=True))[:max_triples]
            return [(u, d["relation"], v) for u, v, d in edges]

        lower_map = {n.lower(): n for n in self.graph.nodes}
        matched: list[str] = []
        for s in seed_nodes:
            if s in self.graph:
                matched.append(s)
            elif s.lower() in lower_map:
                matched.append(lower_map[s.lower()])

        if not matched:
            return []

        visited = set(matched)
        frontier = set(matched)
        for _ in range(hops):
            nxt: set[str] = set()
            for n in frontier:
                # DiGraph: successors + predecessors for undirected BFS feel, but keep direction in triples
                nxt |= set(self.graph.successors(n)) | set(self.graph.predecessors(n))
                # filter already visited
            nxt -= visited
            visited |= nxt
            frontier = nxt

        sub = self.graph.subgraph(visited)
        triples = [(u, d["relation"], v) for u, v, d in sub.edges(data=True)]
        # Rank by distance from matched nodes (0=incident, 1=1-hop neighbor, etc.) then deterministic
        # Compute BFS distance from matched set
        from collections import deque

        dist: dict[str, int] = {n: 0 for n in matched}
        q = deque(matched)
        while q:
            cur = q.popleft()
            for nb in set(self.graph.successors(cur)) | set(self.graph.predecessors(cur)):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        def rank(t: tuple[str, str, str]) -> tuple[int, str]:
            u, _, v = t
            d = min(dist.get(u, 99), dist.get(v, 99))
            return (d, f"{u} {v}")
        triples.sort(key=rank)
        return triples[:max_triples]

    def stats(self) -> dict:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}

    def triples_as_text(self, triples: list[tuple[str, str, str]]) -> str:
        return "\n".join([f"{s} --[{r}]--> {t}" for s, r, t in triples])
