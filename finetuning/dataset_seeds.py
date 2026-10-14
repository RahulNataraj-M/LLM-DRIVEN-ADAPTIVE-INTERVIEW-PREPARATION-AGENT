GOOD_ANS = {
    "DSA": ("I would use two pointers moving at different speeds; if they collide a cycle "
            "exists. This is O(n) time and O(1) space, which is the key advantage over a "
            "visited set."),
    "DBMS": ("A clustered index physically orders rows by the key, so range scans are fast. "
             "A non-clustered index keeps a separate structure mapping keys to row pointers, "
             "useful when the hot column differs from physical order."),
    "Operating Systems": ("A process owns an address space; threads inside it share that "
                          "space and its file descriptors but keep separate registers, stacks "
                          "and program counters."),
    "OOP": ("Overloading is resolved at compile time by signature; overriding is resolved at "
            "run time through dynamic dispatch on the object type."),
    "Computer Networks": ("TCP gives ordered, reliable delivery with congestion control at "
                          "the cost of latency; UDP is connectionless and lossy but fast, "
                          "so it suits real-time media."),
    "Programming": ("A pure function has no side effects and always returns the same output "
                    "for the same input, which makes it trivial to test in isolation."),
}

BAD_ANS = {
    "DSA": ("I would mark every visited node in a set and reuse the original list; when a "
            "node repeats, there is a cycle. Space is O(1) because I reuse the pointer field."),
    "DBMS": ("Just put an index on the column and it will be fast; the whole table is "
             "optimized automatically once the key is created."),
    "Operating Systems": ("A thread is just a lightweight process; the only real difference "
                          "is that threads are faster because they skip the scheduler."),
    "OOP": ("Overloading and overriding are basically the same idea — the compiler picks the "
            "closest match at run time. Return type is the key difference."),
    "Computer Networks": ("UDP is always faster than TCP because it has smaller headers, so "
                          "it is strictly better unless you need to check packets arrived."),
    "Programming": ("A pure function is one that never throws exceptions; if it is fast it "
                    "is also free of side effects."),
}

EVAL_GOOD = {
    "correctness": 0.85, "conceptual": 0.82, "communication": 0.86,
    "missing_concepts": [], "misconceptions": [],
    "feedback": "Covers the mechanism, the trade-offs and the complexity implications with "
                "clear structure.",
    "rec": "ASK_HARDER",
}
EVAL_BAD = {
    "correctness": 0.3, "conceptual": 0.28, "communication": 0.4,
    "missing_concepts": ["cycle detection", "space complexity", "index scan mechanics"],
    "misconceptions": ["claims O(1) space for a non-destructive algorithm"],
    "feedback": "Heard the keywords but the mechanism and complexity claims are inverted.",
    "rec": "GIVE_HINT",
}

QUESTIONS = {
    "DSA": "Detect a cycle in a linked list. How do you approach it and what is the "
           "space-time trade-off?",
    "DBMS": "How do clustered and non-clustered indexes differ, and when does each "
            "speed up a range scan?",
    "Operating Systems": "What is the difference between a process and a thread?",
    "OOP": "How do method overloading and method overriding differ, and when is each "
           "resolved?",
    "Computer Networks": "Compare TCP and UDP for real-time voice traffic.",
    "Programming": "What makes a function pure, and why does that help testing?",
}

SEEDS = [
    {"topic": topic, "question": question,
     "good": dict(EVAL_GOOD), "bad": dict(EVAL_BAD)}
    for topic, question in QUESTIONS.items()
]
