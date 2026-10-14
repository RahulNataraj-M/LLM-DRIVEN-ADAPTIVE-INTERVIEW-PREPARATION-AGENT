import hashlib
import math
import re
import requests

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
NG_DIM = 4096
NG_MIN = 2
NG_MAX = 4


def _normalize(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def _hash_ngrams(text: str) -> dict:
    clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = clean.split()
    vec: dict[int, float] = {}
    for token in tokens:
        for n in range(NG_MIN, min(NG_MAX, len(token)) + 1):
            for i in range(len(token) - n + 1):
                gram = token[i : i + n]
                idx = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % NG_DIM
                vec[idx] = vec.get(idx, 0.0) + 1.0
    return vec


def _dense_to_dict(vals: list) -> dict:
    return {i: float(v) for i, v in enumerate(vals)}


def ollama_available(model: str = EMBED_MODEL) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(n.startswith(model) for n in names)
    except Exception:
        return False


class Embedder:

    def __init__(self, prefer_ollama: bool = True):
        self.mode = "local"
        self.dim = NG_DIM
        self.ollama = False
        if prefer_ollama and ollama_available():
            try:
                self.embed(["probe"])[0]
                self.mode = "ollama"
                self.ollama = True
            except Exception:
                self.mode = "local"

    def embed(self, texts: list[str]) -> list[dict]:
        if self.mode == "ollama":
            try:
                payload = {"model": EMBED_MODEL, "input": texts}
                resp = requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=60)
                resp.raise_for_status()
                raw = resp.json().get("embeddings", [])
                if raw:
                    return [_normalize(_dense_to_dict(v)) for v in raw]
            except Exception:
                self.ollama = False
                self.mode = "local"
        return [_normalize(_hash_ngrams(t)) for t in texts]

    def embed_one(self, text: str) -> dict:
        return self.embed([text])[0]

    def cosine(self, a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        return sum(a.get(i, 0.0) * b.get(i, 0.0) for i in a if i in b)

    def top_k(self, query: str, items: list[dict], k: int = 3) -> list[tuple[float, int]]:
        if not items:
            return []
        q = self.embed_one(query)
        scored = [(self.cosine(q, item.get("embedding") or {}), i) for i, item in enumerate(items)]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:k]
