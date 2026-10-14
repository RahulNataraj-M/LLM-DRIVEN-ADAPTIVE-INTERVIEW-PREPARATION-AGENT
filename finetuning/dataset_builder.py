import json
import os
import random

import config as C

FINETUNE_DIR = os.path.join(C.DATA_DIR, "finetune")
OUT = os.path.join(FINETUNE_DIR, "eval_dataset.jsonl")

SYSTEM = (
    "You are an expert technical interviewer scoring a candidate's answer. Reply "
    "with ONLY JSON: {\"correctness\", \"conceptual_understanding\", "
    "\"communication\", \"missing_concepts\"[], \"misconceptions\"[], "
    "\"feedback\", \"recommended_action\"}."
)

_VARIANT_PREFIXES = [
    "",
    "Let me be brief: ",
    "Let me walk through this step by step. First I lay out the question in my own "
    "words, then I examine the relevant options and their trade-offs before "
    "committing to the approach. So, specifically, ",
]

_ANSWER_TEXT = {
    ("DSA", "good"): ("I would use two pointers moving at different speeds; if they collide "
                      "a cycle exists. This runs in O(n) time and O(1) space, which is the "
                      "key advantage over a visited set."),
    ("DSA", "bad"): ("I would mark every visited node in a set and reuse the original list; "
                     "when a node repeats there is a cycle. Space is O(1) because I reuse "
                     "the pointer field."),
    ("DBMS", "good"): ("Clustered indexes physically order rows by the key so range scans "
                       "are fast; non-clustered keep a separate key-to-pointer map."),
    ("DBMS", "bad"): ("Just put an index on the hot column and it will be fast."),
    ("Operating Systems", "good"): ("A process owns an address space; threads within it "
                                    "share memory and file descriptors but keep separate "
                                    "registers, stacks and program counters."),
    ("Operating Systems", "bad"): ("A thread is a lightweight process with no real "
                                   "differences other than being faster."),
    ("OOP", "good"): ("Overloading is resolved at compile time by signature; overriding "
                      "is resolved at run time through dynamic dispatch."),
    ("OOP", "bad"): ("Overloading and overriding pick the closest match at compile time by "
                     "return type—they are basically the same mechanism."),
    ("Computer Networks", "good"): ("TCP gives ordered, reliable delivery with congestion "
                                    "control; UDP trades that for low latency, which suits "
                                    "real-time and loss-tolerant traffic."),
    ("Computer Networks", "bad"): ("UDP is always faster than TCP because it has no "
                                   "handshake, so it is strictly better."),
    ("Programming", "good"): ("A pure function has no side effects and always returns the "
                              "same output for the same input, which makes it trivially "
                              "testable and cacheable."),
    ("Programming", "bad"): ("A pure function is one that never throws exceptions; if it "
                             "is quick to run it is also pure."),
}


def _seeded_answers(seed: dict, label: str) -> list[tuple[str, dict]]:
    base = seed[label]
    text = _ANSWER_TEXT.get((seed["topic"], label)) or seed["question"]
    answers = []
    for pre in _VARIANT_PREFIXES:
        answers.append(((pre + text).strip(), base))
    return answers


def build() -> int:
    from finetuning.dataset_seeds import SEEDS

    rng = random.Random(C.SEED)
    os.makedirs(FINETUNE_DIR, exist_ok=True)
    rows = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            topic = seed["topic"]
            q = seed["question"]
            for label in ("good", "bad"):
                for answer, ev in _seeded_answers(seed, label):
                    user = (
                        f"Evaluate this candidate answer.\nTopic: {topic}\n"
                        f"Question: {q}\nAnswer: {answer}"
                    )
                    assistant = json.dumps({
                        "correctness": ev["correctness"],
                        "conceptual_understanding": ev["conceptual"],
                        "communication": ev["communication"],
                        "missing_concepts": ev["missing_concepts"],
                        "misconceptions": ev["misconceptions"],
                        "feedback": ev["feedback"],
                        "recommended_action": ev["rec"],
                    })
                    for role, content in (("system", SYSTEM), ("user", user),
                                          ("assistant", assistant)):
                        f.write(json.dumps({"role": role, "content": content}) + "\n")
                        rows += 1
    print(f"Built {rows} lines -> {OUT}")
    return rows


if __name__ == "__main__":
    build()
