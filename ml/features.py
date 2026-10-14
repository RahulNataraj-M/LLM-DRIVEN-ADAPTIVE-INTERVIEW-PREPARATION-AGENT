from candidate.profile import CandidateProfile, WEAK_THRESHOLD

FEATURES = ["topic_idx", "difficulty", "skill", "accuracy",
            "attempts", "hints", "is_weak", "prev5"]


def build_features(profile: CandidateProfile, topic: str,
                   difficulty: int, hinted: bool) -> list[float]:
    topic_idx = _topic_index(topic)
    prev5 = profile.recent_outcomes[-5:]
    prev5_mean = (sum(prev5) / len(prev5)) if prev5 else 0.5
    return [
        float(topic_idx),
        float(max(1, min(5, int(difficulty)))),
        float(profile.skills.get(topic, 0.5)),
        float(profile.recent_accuracy()),
        float(min(20.0, profile.attempts.get(topic, 0))),
        float(1 if hinted else 0),
        float(1 if profile.skills.get(topic, 0.5) <= WEAK_THRESHOLD else 0),
        float(prev5_mean),
    ]


def _topic_index(topic: str) -> int:
    try:
        from config import TOPICS
        return TOPICS.index(topic)
    except ValueError:
        return 0
