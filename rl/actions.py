import random

from config import (DIFFICULTY_MIN, DIFFICULTY_MAX, TOPICS, SEED, RL_ACTIONS)

if not RL_ACTIONS:  # defensive: never run with an empty action space
    RL_ACTIONS = ["ASK_SAME_DIFFICULTY"]


class ActionPlan:
    def __init__(self, topic: str, difficulty: int, hint_policy: str = "mid",
                 prompt_directive: str = "", revisit_text: str = "",
                 action: str = ""):
        self.topic = topic
        self.difficulty = difficulty
        self.hint_policy = hint_policy   # "none" | "mid" | "full"
        self.prompt_directive = prompt_directive
        self.revisit_text = revisit_text
        self.action = action

    def __repr__(self):  # pragma: no cover
        return f"ActionPlan(action={self.action}, topic={self.topic}, diff={self.difficulty}, hint={self.hint_policy})"


class ActionEngine:
    def __init__(self, seed: int = SEED):
        self.rng = random.Random(seed)

    def apply(self, action: str, profile, memory_store=None,
              current_topic: str | None = None, difficulty: int = 3,
              topic_pool=None) -> ActionPlan:
        a = (action or "ASK_SAME_DIFFICULTY").upper()
        pool = [t for t in (topic_pool or TOPICS) if t in TOPICS]
        if not pool:
            pool = list(TOPICS)
        topic = current_topic if current_topic in pool else pool[0]
        new_diff = int(difficulty)
        hint = "mid"
        directive = ""
        revisit = ""

        if a == "ASK_EASIER":
            new_diff = max(DIFFICULTY_MIN, difficulty - 1)
        elif a == "ASK_HARDER":
            new_diff = min(DIFFICULTY_MAX, difficulty + 1)
        elif a == "ASK_WEAK_TOPIC":
            topic = profile.weakness(topic, topics=pool)
            new_diff = max(DIFFICULTY_MIN, difficulty - 1)
            directive = "Target this heavy weakness area; the candidate is shaky here, keep it concrete."
        elif a == "ASK_NEW_TOPIC":
            candidates = [t for t in pool if t != topic]
            if candidates:
                topic = self.rng.choice(candidates)
                directive = "Switch to a fresh topic to diversify the interview."
            else:
                directive = "Only one topic selected for this session; staying on it."
        elif a == "REVISIT_MISTAKE":
            mem = (memory_store.retrieve_mistakes(k=1, topic=topic)
                   if memory_store else [])
            if not mem and set(pool) == set(TOPICS) and memory_store:
                # unrestricted session: any past mistake is fair game
                mem = memory_store.retrieve_mistakes(k=1)
            if mem:
                revisit = mem[0].get("text", "")
                topic = mem[0].get("topic", topic)
                new_diff = max(DIFFICULTY_MIN, difficulty - 1)
                directive = ("This candidate previously missed a question like this. "
                             "Re-ask with a slightly simpler variant, check understanding, don't repeat the first question verbatim.")
            else:
                topic = profile.weakness(topic, topics=pool)
                directive = "No recorded mistake yet; probe this candidate's weakest topic instead."
        elif a == "GIVE_HINT":
            hint = "full"
            directive = "This candidate earned generous scaffolding: offer a clear hint to keep them moving."
        elif a == "ASK_FOLLOWUP":
            directive = "Follow up on the candidate's previous answer; drill one level deeper on the same idea."
        elif a == "ASK_SAME_DIFFICULTY":
            pass

        return ActionPlan(topic=topic, difficulty=int(new_diff), hint_policy=hint,
                          prompt_directive=directive, revisit_text=revisit, action=a)
