import json
import os
import time

import config as C
from embeddings import Embedder


class CandidateMemoryStore:
    def __init__(self, candidate_id: str = "guest", top_k: int = 3):
        self.candidate_id = candidate_id
        self.top_k = top_k
        self.entries: list[dict] = []
        self.embedder = Embedder()

    def path(self) -> str:
        safe = self.candidate_id.replace("/", "_").replace("\\", "_")
        return os.path.join(C.VECTORS_DIR, f"{safe}.json")

    def add_memory(self, text: str, topic: str, turn: int,
                   kind: str, mistake: bool = False):
        emb = self.embedder.embed_one(text)
        self.entries.append({
            "text": text,
            "embedding": emb,
            "topic": topic,
            "turn": turn,
            "kind": kind,            # question | answer | evaluation | summary
            "mistake": bool(mistake),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self.trim()

    def add_answer_memory(self, topic: str, turn: int, question: str,
                          answer: str, eval_data: dict | None):
        """Store a compact, retrievable record of this Q/A exchange."""
        correctness = float((eval_data or {}).get("correctness", 0.0))
        mistake = correctness < 0.35
        gist = f"Topic {topic}: candidate answered a question about '{question[:100]}' with quality {correctness:.2f}."
        if mistake:
            gist += " This was a mistake / weak area."
        self.add_memory(gist, topic, turn, "evaluation", mistake=mistake)

    def retrieve(self, query: str, k: int | None = None, topic: str | None = None) -> list[dict]:
        """k-NN: return the top-k entries most similar to `query` (cosine)."""
        k = k or self.top_k
        items = [e for e in self.entries if topic is None or e.get("topic") == topic]
        if not items:
            return []
        scored = self.embedder.top_k(query, items, k=k)
        return [items[i] for _, i in scored]

    def retrieve_mistakes(self, k: int = 2, topic: str | None = None) -> list[dict]:
        items = [e for e in self.entries
                 if e.get("mistake") and (topic is None or e.get("topic") == topic)]
        return items[-k:]

    def to_context(self, memories: list[dict]) -> str:
        if not memories:
            return ""
        lines = ["RELEVANT PAST MEMORIES:"]
        for m in memories:
            tag = "[MISTAKE]" if m.get("mistake") else "[MEMORY]"
            lines.append(f"{tag} {m.get('text', '')}")
        return "\n".join(lines)

    def trim(self, limit: int = 300):
        if len(self.entries) > limit:
            self.entries = self.entries[-limit:]

    def save(self):
        os.makedirs(C.VECTORS_DIR, exist_ok=True)
        payload = {"candidate_id": self.candidate_id, "entries": self.entries}
        with open(self.path(), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, candidate_id: str = "guest") -> "CandidateMemoryStore":
        store = cls(candidate_id)
        if os.path.exists(store.path()):
            with open(store.path(), encoding="utf-8") as f:
                payload = json.load(f)
            store.entries = payload.get("entries", [])
        return store
