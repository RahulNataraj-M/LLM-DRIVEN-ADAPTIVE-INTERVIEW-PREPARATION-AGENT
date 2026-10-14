import json
import os
import statistics
import time

import config as C

# Configurable thresholds (see also config.TOPIC_SUBTOPICS)
SKILL_UPDATE_RATE = 0.15        # EWMA learning rate per answered question
WEAK_THRESHOLD = 0.45           # topic skill below this counts as a weakness
WEAKNESS_THRESHOLD = WEAK_THRESHOLD   # alias used by the weakness profile
STRONG_THRESHOLD = 0.65         # legacy fallback classification floor
STRENGTH_THRESHOLD = 0.75       # a topic/subtopic is a strength only at/above this
STRENGTH_MIN_EVIDENCE = 3       # and only after enough repeated performance
WEAKNESS_MIN_EVIDENCE = 2       # never classify from a single failed question
INITIAL_SKILL = 0.50

# Severity bands for weaknesses (score = EWMA skill of the unit).
SEVERITY_HIGH_AT = 0.25
SEVERITY_MEDIUM_AT = 0.35

# A correctness >= CORRECT_AT counts as "correct" for history statistics.
CORRECT_AT = 0.5

MISCONCEPTION_CONFIDENCE_INIT = 0.70
MISCONCEPTION_CONFIDENCE_STEP = 0.15
MISCONCEPTION_CONFIDENCE_MAX = 0.95
MISCONCEPTION_CONFIDENCE_RESOLVED_STEP = -0.20

TOPIC_NAMES = list(C.TOPICS)
TOPIC_SUBTOPICS = dict(getattr(C, "TOPIC_SUBTOPICS", {}) or {})


def subtopics_for(topic: str) -> list:
    """Known subtopics for a topic, from the configurable mapping."""
    return list(TOPIC_SUBTOPICS.get(topic, []) or [])


def resolve_subtopic(topic: str, text: str = "", explicit=None):
    subs = subtopics_for(topic)
    if not subs:
        return None
    if explicit:
        e = (explicit or "").strip()
        hit = next((s for s in subs if e.lower() == s.lower()), None)
        if hit is not None:
            return hit
    if not text:
        return None
    low = text.lower()
    best, best_score = None, 0
    for s in subs:
        sl = s.lower()
        score = 2 if (len(sl) >= 3 and sl in low) else 0
        if score == 0:
            score = sum(1 for tok in sl.split() if len(tok) >= 3 and tok in low)
        if score > best_score:
            best, best_score = s, score
    return best if best_score >= 1 else None


def _normalize_mistakes(mistakes):
    """Ensure legacy mistake records carry the new optional fields."""
    out = []
    for m in (mistakes or []):
        if not isinstance(m, dict):
            continue
        m.setdefault("subtopic", None)
        m.setdefault("resolved", False)
        out.append(m)
    return out


def _merge_subtopics(defaults: dict, loaded: dict, default=INITIAL_SKILL):
    merged = {t: dict(subs) for t, subs in defaults.items()}
    for topic, subs in (loaded or {}).items():
        if topic not in merged:
            continue
        for sub, val in (subs or {}).items():
            if sub in merged[topic]:
                merged[topic][sub] = default if default == 0 else float(val)
    return merged


class CandidateProfile:
    def __init__(self, candidate_id: str = "guest", name: str | None = None):
        self.candidate_id = candidate_id
        # --- A. basic candidate information (optional, never required) -----
        self.name = name
        self.education: str | None = None
        self.branch: str | None = None
        self.target_role: str | None = None
        self.experience_level: str | None = None
        self.preferred_topics: list = []

        # --- B. topic-level (existing) --------------------------------------
        self.skills: dict[str, float] = {t: INITIAL_SKILL for t in TOPIC_NAMES}
        self.attempts: dict[str, int] = {t: 0 for t in TOPIC_NAMES}
        self.hints_used: dict[str, int] = {t: 0 for t in TOPIC_NAMES}
        self.recent_outcomes: list[float] = []
        self.mistakes: list[dict] = []     # [{topic, subtopic, question, gist, ts, resolved}]
        self.last_seen: str = ""

        # --- 2. subtopic-level skills ---------------------------------------
        self.subtopic_skills: dict[str, dict[str, float]] = {
            t: {s: INITIAL_SKILL for s in subtopics_for(t)} for t in TOPIC_NAMES}
        self.subtopic_attempts: dict[str, dict[str, int]] = {
            t: {s: 0 for s in subtopics_for(t)} for t in TOPIC_NAMES}

        # --- 3/4. strengths & weaknesses (derived, recomputed) --------------
        self.strengths: dict[str, dict] = {}
        self.weaknesses: dict[str, dict] = {}

        # --- 5. misconceptions (evaluator-reported only) --------------------
        self.misconceptions: dict[str, list[dict]] = {}

        # --- 8/9. history ----------------------------------------------------
        self.topic_history: dict[str, dict] = {}
        self.difficulty_history: dict[str, dict] = {}

        # --- 10. observable behaviour ----------------------------------------
        self.behavior: dict = {
            "average_response_time": None,   # None until real timing data exists
            "hints_per_question": 0.0,
            "followups_per_question": 0.0,
            "consistency": 1.0,
            "_followups": 0,                 # internal counter (persisted)
        }

        # --- 11. learning progress (per completed session) -------------------
        self.progress: dict = {
            "sessions_completed": 0,
            "score_history": [],
            "improvement": 0.0,
        }

    # Persistence
    def path(self) -> str:
        safe = self.candidate_id.replace("/", "_").replace("\\", "_")
        return os.path.join(C.CANDIDATE_DIR, f"{safe}.json")

    def save(self):
        os.makedirs(C.CANDIDATE_DIR, exist_ok=True)
        self._recompute_classifications()
        data = {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "education": self.education,
            "branch": self.branch,
            "target_role": self.target_role,
            "experience_level": self.experience_level,
            "preferred_topics": list(self.preferred_topics or []),
            "skills": self.skills,
            "subtopic_skills": self.subtopic_skills,
            "subtopic_attempts": self.subtopic_attempts,
            "attempts": self.attempts,
            "hints_used": self.hints_used,
            "recent_outcomes": self.recent_outcomes[-50:],
            "mistakes": self.mistakes[-40:],
            "misconceptions": self.misconceptions,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "topic_history": self.topic_history,
            "difficulty_history": self.difficulty_history,
            "behavior": self.behavior,
            "progress": self.progress,
            "last_seen": self.last_seen,
        }
        with open(self.path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def create(cls, candidate_id: str = "guest", name: str | None = None) -> "CandidateProfile":
        """A complete, fully-defaulted profile for a new candidate."""
        return cls(candidate_id, name=name)

    @classmethod
    def load(cls, candidate_id: str = "guest", name: str | None = None) -> "CandidateProfile":
        """Restore a profile, migrating legacy files (missing fields -> defaults)."""
        p = cls(candidate_id, name=name)
        path = p.path()
        if not os.path.exists(path):
            return p
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        p.name = data.get("name") or p.name or candidate_id
        p.education = data.get("education")
        p.branch = data.get("branch")
        p.target_role = data.get("target_role")
        p.experience_level = data.get("experience_level")
        p.preferred_topics = list(data.get("preferred_topics") or [])

        p.skills = {t: float(data.get("skills", {}).get(t, INITIAL_SKILL)) for t in TOPIC_NAMES}
        p.attempts = {t: int(data.get("attempts", {}).get(t, 0)) for t in TOPIC_NAMES}
        p.hints_used = {t: int(data.get("hints_used", {}).get(t, 0)) for t in TOPIC_NAMES}
        p.recent_outcomes = [float(x) for x in data.get("recent_outcomes", [])]
        p.mistakes = _normalize_mistakes(data.get("mistakes", []))
        p.last_seen = data.get("last_seen", "")

        p.subtopic_skills = _merge_subtopics(p.subtopic_skills, data.get("subtopic_skills", {}))
        p.subtopic_attempts = _merge_subtopics(p.subtopic_attempts,
                                               data.get("subtopic_attempts", {}), default=0)
        p.misconceptions = dict(data.get("misconceptions") or {})
        p.topic_history = dict(data.get("topic_history") or {})
        p.difficulty_history = {str(k): dict(v) for k, v in (data.get("difficulty_history") or {}).items()}
        saved_behavior = dict(data.get("behavior") or {})
        p.behavior.update({k: v for k, v in saved_behavior.items() if not k.startswith("_")})
        p.behavior["_followups"] = int(saved_behavior.get("_followups", 0))
        for key, dflt in (("average_response_time", None),
                          ("hints_per_question", 0.0),
                          ("followups_per_question", 0.0),
                          ("consistency", 1.0)):
            if p.behavior.get(key) is None or key not in saved_behavior:
                p.behavior[key] = dflt
        p.progress = dict(data.get("progress") or p.progress)

        p._recompute_classifications()
        return p

    # Core update pipeline (extends the existing record())
    def record(self, topic: str, correctness: float, hinted: bool = False,
               question: str = "", gist: str = "",
               subtopic=None, difficulty=None, misconceptions=None,
               followup: bool = False, response_time=None):
        correctness = float(correctness)
        self.last_seen = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.attempts[topic] = self.attempts.get(topic, 0) + 1
        if hinted:
            self.hints_used[topic] = self.hints_used.get(topic, 0) + 1

        # -- topic EWMA skill (unchanged mechanism) ------------------------
        prior = self.skills.get(topic, INITIAL_SKILL)
        self.skills[topic] = round(prior + SKILL_UPDATE_RATE * (correctness - prior), 4)
        self.recent_outcomes.append(correctness)

        # -- subtopic skill (only when a known subtopic is provided) --------
        unit = topic
        known_sub = None
        if subtopic and subtopic in subtopics_for(topic):
            known_sub = subtopic
            unit = subtopic
            prior_sub = self.subtopic_skills[topic].get(subtopic, INITIAL_SKILL)
            self.subtopic_skills[topic][subtopic] = round(
                prior_sub + SKILL_UPDATE_RATE * (correctness - prior_sub), 4)
            self.subtopic_attempts[topic][subtopic] = (
                self.subtopic_attempts[topic].get(subtopic, 0) + 1)

        # -- topic history ---------------------------------------------------
        th = self.topic_history.setdefault(topic, {"attempts": 0, "correct": 0, "accuracy": 0.0})
        th["attempts"] += 1
        if correctness >= CORRECT_AT:
            th["correct"] += 1
        th["accuracy"] = round(th["correct"] / th["attempts"], 4)

        # -- difficulty history (uses the app's own difficulty ints 1..5) -----
        if difficulty is not None:
            d = str(int(difficulty))
            dh = self.difficulty_history.setdefault(
                d, {"attempts": 0, "correct": 0, "accuracy": 0.0})
            dh["attempts"] += 1
            if correctness >= CORRECT_AT:
                dh["correct"] += 1
            dh["accuracy"] = round(dh["correct"] / dh["attempts"], 4)

        # -- mistakes (existing behaviour + richer fields) -------------------
        if correctness < 0.35 and question:
            self.mistakes.append({
                "topic": topic,
                "subtopic": known_sub,
                "question": question,
                "gist": gist or question[:120],
                "ts": self.last_seen,
                "resolved": False,
            })
            self.mistakes = self.mistakes[-40:]

        if correctness >= 0.6 and unit:
            for m in self.mistakes:
                if m.get("resolved"):
                    continue
                msub = m.get("subtopic")
                related = (msub and msub == unit) or (not msub and m.get("topic") == unit)
                if related:
                    m["resolved"] = True
                    m.setdefault("resolved_at", self.last_seen)

        # -- misconceptions: only from an explicit evaluator signal ----------
        for text in (misconceptions or []):
            text = (text or "").strip()
            if not text:
                continue
            bucket = self.misconceptions.setdefault(topic, [])
            existing = next((e for e in bucket if e.get("misconception") == text), None)
            if existing:
                existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
                existing["confidence"] = round(min(
                    MISCONCEPTION_CONFIDENCE_MAX,
                    float(existing.get("confidence", MISCONCEPTION_CONFIDENCE_INIT))
                    + MISCONCEPTION_CONFIDENCE_STEP), 3)
            else:
                bucket.append({
                    "concept": unit,
                    "misconception": text,
                    "confidence": MISCONCEPTION_CONFIDENCE_INIT,
                    "occurrences": 1,
                    "resolved": False,
                })
        if correctness >= 0.6 and unit:
            for e in self.misconceptions.get(topic, []):
                if not e.get("resolved") and e.get("concept") == unit:
                    e["resolved"] = True
                    e["confidence"] = round(max(0.1, float(e.get(
                        "confidence", MISCONCEPTION_CONFIDENCE_INIT))
                        + MISCONCEPTION_CONFIDENCE_RESOLVED_STEP), 3)

        # -- observable behaviour --------------------------------------------
        self._update_behavior(correctness, followup, response_time)

        # -- refresh derived classifications ---------------------------------
        self._recompute_classifications()

    def record_session(self, score):
        """Record one completed interview session into learning progress."""
        if score is None:
            return
        hist = list(self.progress.get("score_history", []))
        hist.append(round(float(score), 4))
        self.progress["score_history"] = hist[-50:]
        self.progress["sessions_completed"] = int(self.progress.get("sessions_completed", 0)) + 1
        if len(hist) >= 2:
            self.progress["improvement"] = round(hist[-1] - hist[0], 4)

    # Derived classifiers
    def _update_behavior(self, correctness, followup=False, response_time=None):
        total = sum(self.attempts.values()) or 1
        self.behavior["_followups"] = int(self.behavior.get("_followups", 0)) + (1 if followup else 0)
        self.behavior["hints_per_question"] = round(sum(self.hints_used.values()) / total, 4)
        self.behavior["followups_per_question"] = round(
            int(self.behavior.get("_followups", 0)) / total, 4)
        outs = self.recent_outcomes
        if len(outs) >= 2:
            self.behavior["consistency"] = round(max(0.0, 1.0 - statistics.pstdev(outs)), 4)
        if response_time is not None:
            prev = self.behavior.get("average_response_time")
            if prev is None:
                self.behavior["average_response_time"] = round(float(response_time), 2)
            else:
                self.behavior["average_response_time"] = round(
                    0.8 * float(prev) + 0.2 * float(response_time), 2)

    @staticmethod
    def _severity(score: float) -> str:
        if score <= SEVERITY_HIGH_AT:
            return "high"
        if score <= SEVERITY_MEDIUM_AT:
            return "medium"
        return "low"

    def _unit_label(self, topic: str, subtopic: str) -> str:
        if subtopic in TOPIC_NAMES:
            return f"{topic}::{subtopic}"
        return subtopic

    def _recompute_classifications(self):
        strengths: dict[str, dict] = {}
        weaknesses: dict[str, dict] = {}

        def classify(label: str, score: float, evidence: int):
            if evidence >= STRENGTH_MIN_EVIDENCE and score >= STRENGTH_THRESHOLD:
                strengths[label] = {"score": round(score, 4), "evidence_count": int(evidence)}
            if evidence >= WEAKNESS_MIN_EVIDENCE and score <= WEAKNESS_THRESHOLD:
                weaknesses[label] = {
                    "score": round(score, 4),
                    "severity": self._severity(score),
                    "evidence_count": int(evidence),
                }

        for topic in TOPIC_NAMES:
            classify(topic, self.skills.get(topic, INITIAL_SKILL), self.attempts.get(topic, 0))
            for sub in subtopics_for(topic):
                classify(self._unit_label(topic, sub),
                         self.subtopic_skills.get(topic, {}).get(sub, INITIAL_SKILL),
                         self.subtopic_attempts.get(topic, {}).get(sub, 0))

        self.strengths = strengths
        self.weaknesses = weaknesses
        return strengths, weaknesses

    # Queries used by the adaptive stack (existing interface preserved)
    def weakness(self, topic: str | None = None, topics=None) -> str:
        pool = [t for t in (topics or TOPIC_NAMES) if t in TOPIC_NAMES]
        weak_topics = {k: v for k, v in self.weaknesses.items() if k in pool}
        if weak_topics:
            return min(weak_topics, key=lambda t: (weak_topics[t].get("score", 1.0),))
        usable = [t for t in pool if self.attempts.get(t, 0) > 0] or pool
        return min(usable, key=lambda t: self.skills.get(t, 1.0))

    def strengths(self) -> list[str]:
        rich = [k for k, v in self.strengths.items()
                if k in TOPIC_NAMES and v.get("evidence_count", 0) > 0]
        if rich:
            return rich
        return [t for t in TOPIC_NAMES
                if self.attempts.get(t, 0) > 0 and self.skills.get(t, 0) >= STRONG_THRESHOLD]

    def weaknesses(self) -> list[str]:
        rich = [k for k in self.weaknesses if k in TOPIC_NAMES]
        if rich:
            return rich
        return [t for t in TOPIC_NAMES
                if self.attempts.get(t, 0) > 0 and self.skills.get(t, 0) <= WEAK_THRESHOLD]

    def recent_accuracy(self, limit: int | None = None) -> float:
        if limit is not None:
            tail = self.recent_outcomes[-limit:]
        else:
            tail = self.recent_outcomes
        if not tail:
            return INITIAL_SKILL
        return round(sum(tail) / len(tail), 4)

    def trend(self, width: int = 5) -> int:
        """Rolling trend over recent outcomes: +1 improving, 0 stable, -1 declining."""
        outs = self.recent_outcomes
        if len(outs) < 2 * width:
            return 0
        recent = outs[-width:]
        older = outs[-2 * width:-width]
        dr = sum(recent) / width - sum(older) / width
        if dr >= 0.05:
            return 1
        if dr <= -0.05:
            return -1
        return 0

    def skill_band(self) -> int:  # 0 low / 1 mid / 2 high (RL state dimension)
        composite = self.recent_accuracy()
        composite += 0.05 if self.trend() > 0 else (-0.05 if self.trend() < 0 else 0.0)
        if any(k in TOPIC_NAMES for k in self.weaknesses):
            composite -= 0.05
        if composite >= 0.68:
            return 2
        if composite >= 0.45:
            return 1
        return 0

    def rl_state(self, topic: str = "DSA", difficulty: int = 3) -> dict:
        acc = self.recent_accuracy()
        n_weak = sum(1 for k in self.weaknesses if k in TOPIC_NAMES)
        return {
            "skill_band": self.skill_band(),
            "recent_accuracy_band": 2 if acc >= 0.68 else (1 if acc >= 0.45 else 0),
            "difficulty": int(max(1, min(5, int(difficulty)))),
            "weakness_level": 0 if n_weak == 0 else (1 if n_weak <= 1 else 2),
            "topic_idx": _topic_index(topic),
            "trend": self.trend(),
        }

    # Computed views (spec: prefer derivation over redundant persistence)
    def performance(self) -> dict:
        """Overall performance, computed from already-persisted counters."""
        n = sum(self.attempts.values()) or 0
        correct = sum(int(v.get("correct", 0)) for v in self.topic_history.values())
        accuracy = round(correct / n, 4) if n else 0.0
        outs = self.recent_outcomes
        avg_score = round(sum(outs) / len(outs), 4) if outs else INITIAL_SKILL
        tot_d, wsum = 0, 0
        for d, v in self.difficulty_history.items():
            try:
                dd = int(d)
            except (TypeError, ValueError):
                continue
            tot_d += int(v.get("attempts", 0))
            wsum += dd * int(v.get("attempts", 0))
        avg_difficulty = round(wsum / tot_d, 2) if tot_d else 0.0
        return {
            "total_questions": n,
            "correct_answers": correct,
            "accuracy": accuracy,
            "average_score": avg_score,
            "average_difficulty": avg_difficulty,
            "last_5_accuracy": self.recent_accuracy(5),
            "last_10_accuracy": self.recent_accuracy(10),
        }

    def to_dict(self) -> dict:
        """Complete serializable view (for inspection / debugging)."""
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "education": self.education,
            "branch": self.branch,
            "target_role": self.target_role,
            "experience_level": self.experience_level,
            "preferred_topics": list(self.preferred_topics or []),
            "skills": self.skills,
            "subtopic_skills": self.subtopic_skills,
            "attempts": self.attempts,
            "subtopic_attempts": self.subtopic_attempts,
            "hints_used": self.hints_used,
            "recent_outcomes": self.recent_outcomes,
            "mistakes": self.mistakes,
            "misconceptions": self.misconceptions,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "topic_history": self.topic_history,
            "difficulty_history": self.difficulty_history,
            "behavior": {k: v for k, v in self.behavior.items() if not k.startswith("_")},
            "progress": self.progress,
            "performance": self.performance(),
            "last_seen": self.last_seen,
        }

    def summary(self) -> dict:
        """Human/audit-friendly summary of the profile."""
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recent_accuracy": self.recent_accuracy(),
            "attempts": self.attempts,
            "mistakes": self.mistakes[-5:],
            "misconceptions": self.misconceptions,
            "difficulty_performance": self.difficulty_history,
            "progress": self.progress,
            "last_seen": self.last_seen,
        }

    def prompt_context(self, topic: str | None = None) -> str:
        """Compact profile context for Llama 3.2 (never the raw JSON dump)."""
        strong = sorted(set(self.strengths) & set(TOPIC_NAMES))
        weak = sorted(set(self.weaknesses) & set(TOPIC_NAMES))
        lines = []
        if strong:
            lines.append("Strong topics: " + ", ".join(strong))
        if weak:
            lines.append("Weak topics: " + ", ".join(weak))
        lines.append(f"Recent accuracy: {self.recent_accuracy() * 100:.0f}%")
        recent = [m for m in self.mistakes[-3:] if not m.get("resolved")]
        if recent:
            gists = ", ".join((m.get("gist") or m.get("question") or "mistake")[:60]
                              for m in recent)
            lines.append("Recent mistakes: " + gists)
        lines.append(f"Current target topic: {topic or ''}")
        return "\n".join(lines)


def _topic_index(topic: str) -> int:
    try:
        return TOPIC_NAMES.index(topic)
    except ValueError:
        return 0
