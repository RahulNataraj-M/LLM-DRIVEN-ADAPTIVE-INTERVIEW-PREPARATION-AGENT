import json
import os
import time

import config as C


class SessionRecorder:
    def __init__(self, candidate_id: str = "guest", mode: str = "practice",
                 persona_name: str = "friendly_mentor", model: str = C.CHAT_MODEL,
                 topics: list[str] | None = None):
        self.candidate_id = candidate_id
        self.mode = mode
        self.persona_name = persona_name
        self.model = model
        self.topics = list(topics or [])
        self.turns: list[dict] = []
        self.start_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.session_id = self.start_ts.replace(":", "").replace("-", "") + f"_{candidate_id}"[:24]

    def add_turn(self, turn: int, topic: str, difficulty: int, action: str,
                 question: str, answer: str, eval_data: dict, hint_policy: str,
                 question_validation: dict | None = None):
        self.turns.append({
            "turn": turn, "topic": topic, "difficulty": difficulty,
            "action": action, "question": question, "answer": answer,
            "evaluation": eval_data, "hint_policy": hint_policy,
            "question_validation": question_validation,
        })

    def summary(self) -> dict:
        n = len(self.turns)
        if n == 0:
            return {"turns": 0}
        by_topic: dict[str, float] = {}
        mistakes: list[str] = []
        diffs = []
        for t in self.turns:
            corr = float(t["evaluation"].get("correctness", 0.0))
            by_topic.setdefault(t["topic"], []).append(corr)
            diffs.append(int(t["difficulty"]))
            if corr < 0.35:
                mistakes.append(t["topic"])
        per_topic = {k: round(sum(v) / len(v), 3) for k, v in by_topic.items()}
        avg = round(sum(diffs) / len(diffs), 2) if diffs else 0
        return {
            "turns": n,
            "avg_correctness": round(sum(
                float(t["evaluation"].get("correctness", 0.0)) for t in self.turns) / n, 3),
            "per_topic_correctness": per_topic,
            "mistake_topics": mistakes,
            "difficulty_trend": "up" if len(diffs) >= 2 and diffs[-1] > diffs[0] else
                                ("down" if len(diffs) >= 2 and diffs[-1] < diffs[0] else "stable"),
            "recommended_focus": min(per_topic, key=per_topic.get) if per_topic else "",
        }

    def save(self) -> str:
        os.makedirs(C.SESSION_DIR, exist_ok=True)
        recap = {
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "persona": self.persona_name,
            "model": self.model,
            "topics": self.topics,
            "started": self.start_ts,
            "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": self.summary(),
            "turns": self.turns,
        }
        path = os.path.join(C.SESSION_DIR, f"{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recap, f, indent=2)
        self._append_episodes()
        return path

    def _append_episodes(self):
        os.makedirs(C.LOGS_DIR, exist_ok=True)
        with open(os.path.join(C.LOGS_DIR, "episodes.jsonl"), "a", encoding="utf-8") as f:
            for t in self.turns:
                f.write(json.dumps({
                    "candidate_id": self.candidate_id,
                    "session_id": self.session_id,
                    "topic": t["topic"], "difficulty": t["difficulty"],
                    "action": t["action"],
                    "correctness": t["evaluation"].get("correctness", 0.0),
                    "conceptual": t["evaluation"].get("conceptual_understanding", 0.0),
                    "answer_valid": t["evaluation"].get("is_valid_answer", True),
                    "hint_policy": t["hint_policy"],
                }) + "\n")

    @classmethod
    def load_latest(cls, candidate_id: str = "guest") -> dict | None:
        if not os.path.isdir(C.SESSION_DIR):
            return None
        files = [f for f in os.listdir(C.SESSION_DIR) if f.endswith(".json")]
        if not files:
            return None
        latest = max(files)
        with open(os.path.join(C.SESSION_DIR, latest), encoding="utf-8") as f:
            return json.load(f)
