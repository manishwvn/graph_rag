"""NVIDIA Nemotron-3-Embed-1B embeddings via integrate.api.nvidia.com v1/embeddings.

Free-tier safe: batch 8, sleep 1.5s, retry 429 with Retry-After/backoff.
Cache key is sha256(model + input_type + text) and the cache is JSON, not
pickle -- unpickling a cache file executes arbitrary code.
"""

import hashlib
import json
import time
from pathlib import Path

import httpx

from graph_rag.config import settings

API_URL = "https://integrate.api.nvidia.com/v1/embeddings"


class NVIDIAEmbeddings:
    """Client for NVIDIA embedding API with free-tier hardening."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        batch_size: int = 8,
        sleep_s: float = 1.5,
        cache_path: str | None = None,
    ):
        self.api_key = api_key or settings.nvidia_api_key
        self.model = model or settings.nvidia_embed_model
        self.dim = dim or settings.nvidia_embed_dim
        self.batch_size = batch_size
        self.sleep_s = sleep_s
        self.cache_path = Path(cache_path or settings.embedding_cache_path)
        self.api_calls = 0
        self.tokens = 0
        self._cache: dict[str, list[float]] = {}
        self._load_cache()
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not set in settings or env")

    def _cache_key(self, text: str, input_type: str) -> str:
        # model is part of the key: switching NVIDIA_EMBED_MODEL must not reuse
        # vectors produced by the previous model.
        return hashlib.sha256(f"{self.model}:{input_type}:{text}".encode()).hexdigest()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                print(f"[embed] loaded cache {len(self._cache)} entries from {self.cache_path}")
            except Exception as e:
                print(f"[embed] cache load failed: {e}")
                self._cache = {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception as e:
            print(f"[embed] cache save failed: {e}")

    def _post(self, texts: list[str], input_type: str) -> list[list[float]]:
        """POST /embeddings with retry on 429."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        retries = 0
        while True:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(API_URL, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                embeddings = [d["embedding"] for d in data["data"]]
                if embeddings and len(embeddings[0]) != self.dim:
                    raise RuntimeError(
                        f"embedding dim {len(embeddings[0])} != expected {self.dim} "
                        f"(set nvidia_embed_dim for model {self.model})"
                    )
                usage = data.get("usage", {}) or {}
                self.api_calls += 1
                self.tokens += int(usage.get("total_tokens", 0))
                print(f"[embed] {input_type} {len(texts)} texts ok, usage {usage}")
                return embeddings
            if resp.status_code == 429:
                retries += 1
                if retries > 3:
                    raise RuntimeError(f"NVIDIA API 429 after 3 retries: {resp.text}")
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2**retries, 8)
                print(f"[embed] 429 rate limit, retry {retries}/3 wait {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                raise RuntimeError("NVIDIA API 401 unauthorized: check NVIDIA_API_KEY")
            raise RuntimeError(f"NVIDIA API {resp.status_code}: {resp.text}")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple passages (indexing). Caches per text."""
        return self._embed_batch(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        """Embed single query."""
        return self._embed_batch([text], "query")[0]

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        results: list[list[float]] = []
        dirty = False
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results: list[list[float] | None] = [None] * len(batch)
            to_embed: list[str] = []
            to_embed_idx: list[int] = []
            for j, t in enumerate(batch):
                key = self._cache_key(t, input_type)
                if key in self._cache:
                    batch_results[j] = self._cache[key]
                else:
                    to_embed.append(t)
                    to_embed_idx.append(j)
            if to_embed:
                for idx, vec in zip(to_embed_idx, self._post(to_embed, input_type), strict=True):
                    batch_results[idx] = vec
                    self._cache[self._cache_key(batch[idx], input_type)] = vec
                dirty = True
                if i + self.batch_size < len(texts):
                    time.sleep(self.sleep_s)
            results.extend(batch_results)  # type: ignore[arg-type]
        if dirty:  # write once per call, not once per batch
            self._save_cache()
        return results  # type: ignore[return-value]
