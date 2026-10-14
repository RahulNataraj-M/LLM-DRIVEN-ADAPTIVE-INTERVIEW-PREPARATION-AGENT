import json
import os
import random

from config import (RL_ACTIONS, RL_ALPHA, RL_GAMMA,
                    RL_EPSILON_START, RL_EPSILON_MIN, RL_EPSILON_DECAY,
                    RL_REWARD_MASTERY, SEED, RL_TABLES_DIR)

AC = list(RL_ACTIONS)


def _default_qpath() -> str:
    return os.path.join(RL_TABLES_DIR, "default.json")


class QLearner:
    """Tabular Q-learning over (skill_band, difficulty, topic) x RL_ACTIONS."""

    def __init__(self, table_path: str | None = None):
        self.alpha = RL_ALPHA
        self.gamma = RL_GAMMA
        self.epsilon = RL_EPSILON_START
        self.epsilon_min = RL_EPSILON_MIN
        self.epsilon_decay = RL_EPSILON_DECAY
        self.q: dict[tuple, dict[str, float]] = {}
        self.visit: dict[tuple, int] = {}
        self.table_path = table_path or _default_qpath()
        self.rng = random.Random(SEED)

    def _state_key(self, skill_band: int, difficulty: int, topic_idx: int):
        return (max(0, min(2, int(skill_band))),
                max(1, min(5, int(difficulty))),
                max(0, int(topic_idx)))

    def _row(self, state) -> dict[str, float]:
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in AC}
        return self.q[state]

    def choose(self, skill_band: int, difficulty: int, topic_idx: int) -> str:
        state = self._state_key(skill_band, difficulty, topic_idx)
        row = self._row(state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(row))
        best = max(row, key=row.get)
        return best

    def update(self, skill_band: int, difficulty: int, topic_idx: int,
               action: str, reward: float, next_difficulty: int):
        s = self._state_key(skill_band, difficulty, topic_idx)
        s2 = self._state_key(skill_band, next_difficulty, topic_idx)
        row, row2 = self._row(s), self._row(s2)
        self.visit[s] = self.visit.get(s, 0) + 1
        target = reward + self.gamma * max(row2.values())
        row[action] = row.get(action, 0.0) + self.alpha * (target - row[action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | None = None):
        path = path or self.table_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "q": {json.dumps(list(k)): v for k, v in self.q.items()},
            "visits": {json.dumps(list(k)): v for k, v in self.visit.items()},
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "decay": self.epsilon_decay,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | None = None, table_id: str | None = None) -> "QLearner":
        if path is None:
            path = os.path.join(RL_TABLES_DIR, f"{table_id or 'default'}.json")
        q = cls(path)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            q.q = {tuple(json.loads(k)): v for k, v in payload.get("q", {}).items()}
            q.visit = {tuple(json.loads(k)): v for k, v in payload.get("visits", {}).items()}
            q.epsilon = float(payload.get("epsilon", q.epsilon))
        return q


def reward_for(mastery: float, prev_mastery: float, prev_action: str) -> float:
    delta = mastery - prev_mastery
    r = 0.0
    if delta >= 0:
        r += 0.5 * delta
        if mastery >= 0.8:
            r += RL_REWARD_MASTERY
        elif mastery >= 0.6:
            r += 0.15
    else:
        r += 0.5 * delta
        if delta < -0.3:
            r -= 0.1
    if mastery >= 0.75 and prev_action == "ASK_HARDER":
        r += 0.2
    if mastery < 0.45 and prev_action in ("ASK_EASIER", "GIVE_HINT"):
        r += 0.1
    return round(r, 4)
