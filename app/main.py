import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as C
import llm_client as LLM
from rl import q_learning as Q
from rl.actions import ActionEngine
from sentiment import get_classifier
from candidate.profile import CandidateProfile, resolve_subtopic, subtopics_for
from memory.memory_knn import CandidateMemoryStore
from ml.performance import PerformancePredictor
from evaluation.evaluator import evaluate_answer
from evaluation.question_validator import (validate_question, question_similarity,
                                           RECENT_WINDOW)
from app.session import SessionRecorder
from app import registry

KIND_BANK = {
    "DSA": "I'd start with a hash map for expected O(1) lookups and confirm the access pattern before choosing.",
    "DBMS": "I'd reach for an index on the hot column and explain the trade-off in I/O.",
    "Operating Systems": "I'd think about the scheduler and context-switch cost, then pick appropriately.",
    "OOP": "I'd encapsulate the behaviour behind an interface and depend on the abstraction.",
    "Computer Networks": "I'd walk the TCP handshake and reason about retransmission under loss.",
    "Programming": "I'd write small testable functions and lead with the failing test.",
}
RUDE_BANK = {
    "DSA": "Hand the hash map over, obviously. Don't waste my time with the rest.",
    "DBMS": "Just use an index. This is basic, why are you even asking.",
    "Operating Systems": "The scheduler handles it, end of story.",
    "OOP": "Interfaces. That's all you need to know.",
    "Computer Networks": "TCP, obviously. Stop prattling on.",
    "Programming": "Pointless. Just make it work and move on.",
}
CURATED_BANK = {
    "DSA": [
        {"difficulty": 2, "subtopic": "Linked Lists", "answer_type": "explanation",
         "question": "Implement an algorithm to detect a cycle in a singly linked list. "
                     "What is the time and space complexity?",
         "hint": "Consider Floyd's tortoise-and-hare: two pointers moving at different speeds.",
         "expected_concepts": ["cycle detection", "two pointers", "time/space complexity"]},
        {"difficulty": 1, "subtopic": "Arrays", "answer_type": "comparison",
         "question": "Given an unsorted array, what is a hash-map-based way to check if a "
                     "pair sums to a target? Contrast it with a nested-loop approach.",
         "hint": "Store complements as you scan; a nested loop would cost O(n^2).",
         "expected_concepts": ["hash map", "complement", "time complexity"]},
        {"difficulty": 2, "subtopic": "Strings", "answer_type": "explanation",
         "question": "How would you find the length of the longest substring without "
                     "repeating characters? Walk through the algorithm and its complexity.",
         "hint": "A sliding window with two indices is O(n); remember where each character last appeared.",
         "expected_concepts": ["sliding window", "hash set/map", "two pointers"]},
        {"difficulty": 3, "subtopic": "Arrays", "answer_type": "explanation",
         "question": "Explain binary search on a sorted array: the invariant, the loop "
                     "bounds, and the time complexity. What goes wrong if the array has duplicates?",
         "hint": "Narrow [lo, hi] by one index per probe; a duplicate can break the monotonic assumption.",
         "expected_concepts": ["binary search", "sorted invariant", "log n complexity"]},
        {"difficulty": 2, "subtopic": "Recursion", "answer_type": "code",
         "question": "Write code to compute n! recursively. Then derive the time and space "
                     "complexity and explain what a base case is for.",
         "hint": "fact(n) = n * fact(n-1) with fact(0) = 1; count the nested calls.",
         "expected_concepts": ["recursion", "base case", "call stack space"]},
    ],
    "DBMS": [
        {"difficulty": 2, "subtopic": "Indexing", "answer_type": "comparison",
         "question": "What is the difference between a clustered and a non-clustered index? "
                     "When would you use each?",
         "hint": "Think about how rows are physically ordered on disk.",
         "expected_concepts": ["physical row order", "key lookup", "secondary index"]},
        {"difficulty": 2, "subtopic": "Transactions", "answer_type": "definition",
         "question": "Explain the ACID properties of a database transaction. Why is each one needed?",
         "hint": "Atomicity, consistency, isolation, durability - one per letter.",
         "expected_concepts": ["atomicity", "isolation", "durability", "consistency"]},
        {"difficulty": 2, "subtopic": "Normalization", "answer_type": "explanation",
         "question": "What problem does normalization solve? Contrast 1NF, 2NF and 3NF "
                     "with one example of a violation for each.",
         "hint": "Redundancy and update anomalies; every normal form removes a specific dependency kind.",
         "expected_concepts": ["redundancy", "functional dependency", "anomalies"]},
        {"difficulty": 1, "subtopic": "SQL Queries", "answer_type": "comparison",
         "question": "What is the difference between an INNER JOIN and a LEFT JOIN? "
                     "Give a query where they produce different row counts.",
         "hint": "Left join keeps every row from the left table even with no match.",
         "expected_concepts": ["join", "null paddings", "row preservation"]},
        {"difficulty": 3, "subtopic": "Indexing", "answer_type": "comparison",
         "question": "Compare a B-tree index with a hash index: which query types is each "
                     "best for, and what are the trade-offs?",
         "hint": "Range scans favour sorted trees; exact-key equality favours hashing.",
         "expected_concepts": ["range queries", "equality lookup", "ordered keys"]},
    ],
    "Operating Systems": [
        {"difficulty": 2, "subtopic": "Processes", "answer_type": "comparison",
         "question": "Explain the difference between a process and a thread, and what "
                     "resources each one shares.",
         "hint": "Consider memory address space and scheduling granularity.",
         "expected_concepts": ["address space", "shared resources", "scheduling unit"]},
        {"difficulty": 3, "subtopic": "Deadlocks", "answer_type": "definition",
         "question": "What is a deadlock? List the four necessary conditions and one "
                     "practical way to break each.",
         "hint": "Mutual exclusion, hold-and-wait, no preemption, circular wait.",
         "expected_concepts": ["circular wait", "hold and wait", "preemption"]},
        {"difficulty": 2, "subtopic": "Scheduling", "answer_type": "explanation",
         "question": "How does round-robin scheduling work? What is the effect of choosing "
                     "a very small versus a very large time quantum?",
         "hint": "Each ready process gets a fixed CPU slice in rotation; the quantum trades "
                 "responsiveness against context-switch overhead.",
         "expected_concepts": ["time quantum", "context switch", "fairness"]},
        {"difficulty": 3, "subtopic": "Memory Management", "answer_type": "explanation",
         "question": "Explain virtual memory and paging. How does a translation look-aside "
                     "buffer (TLB) speed up address translation?",
         "hint": "Pages map virtual to physical; a TLB is a hardware cache of recent mappings.",
         "expected_concepts": ["page tables", "virtual to physical", "tlb cache"]},
        {"difficulty": 3, "subtopic": "Threads", "answer_type": "explanation",
         "question": "What is a race condition? Give an example with two threads on a shared "
                     "counter, and explain one way to prevent it.",
         "hint": "Interleaving of read-modify-write; a lock or atomic operation serializes it.",
         "expected_concepts": ["interleaving", "shared state", "mutex/atomicity"]},
    ],
    "OOP": [
        {"difficulty": 2, "subtopic": "Polymorphism", "answer_type": "comparison",
         "question": "Explain the difference between method overloading and method overriding "
                     "in Java-like languages.",
         "hint": "One is resolved at compile time, the other at run time.",
         "expected_concepts": ["compile-time binding", "run-time dispatch", "signature"]},
        {"difficulty": 2, "subtopic": "Encapsulation", "answer_type": "comparison",
         "question": "What is the difference between encapsulation and abstraction? Give one "
                     "example of each.",
         "hint": "Encapsulation hides what changes (state); abstraction hides how (details).",
         "expected_concepts": ["data hiding", "information hiding", "interfaces"]},
        {"difficulty": 3, "subtopic": "Abstraction", "answer_type": "comparison",
         "question": "When would you choose an interface over an abstract class? Explain the "
                     "semantic difference between the two.",
         "hint": "An interface is a contract of capability; an abstract class carries shared "
                 "state or partial implementation.",
         "expected_concepts": ["contract vs implementation", "multiple inheritance",
                               "shared state"]},
        {"difficulty": 3, "subtopic": "Inheritance", "answer_type": "comparison",
         "question": "Compare inheritance with composition. When does composition become the "
                     "better design, and why?",
         "hint": "Prefer has-a over is-a when coupling or reuse breaks down; delegation overrides inheritance subtly.",
         "expected_concepts": ["has-a vs is-a", "coupling", "fragile base class"]},
        {"difficulty": 2, "subtopic": "Polymorphism", "answer_type": "explanation",
         "question": "What does it mean for a language to dispatch methods dynamically? "
                     "Why does that matter for polymorphism?",
         "hint": "The runtime chooses which implementation to call based on the object's actual type.",
         "expected_concepts": ["dynamic dispatch", "vtable", "subtype"]},
    ],
    "Computer Networks": [
        {"difficulty": 2, "subtopic": "Transport Layer", "answer_type": "explanation",
         "question": "Walk me through what happens when a client requests a webpage over "
                     "TCP: DNS, handshake, transfer.",
         "hint": "Order the steps: DNS resolution, TCP three-way handshake, HTTP, teardown.",
         "expected_concepts": ["dns resolution", "syn-ack handshake", "http request"]},
        {"difficulty": 1, "subtopic": "Transport Layer", "answer_type": "comparison",
         "question": "Compare TCP and UDP: reliability, ordering, and typical use cases "
                     "for each.",
         "hint": "One guarantees delivery and order with handshakes and retransmission; "
                 "the other is connectionless and best-effort.",
         "expected_concepts": ["reliability", "ordering", "connection"], },
        {"difficulty": 2, "subtopic": "OSI & TCP/IP Model", "answer_type": "definition",
         "question": "List the layers of the OSI model and say what the transport and "
                     "network layers are responsible for.",
         "hint": "Seven layers from physical to application; remember where ports and "
                 "IP addresses fit.",
         "expected_concepts": ["seven layers", "ports-transport", "ip-network"]},
        {"difficulty": 2, "subtopic": "Application Layer", "answer_type": "comparison",
         "question": "What is the difference between HTTP and HTTPS? How does the latter "
                     "protect a request?",
         "hint": "HTTPS wraps HTTP in TLS; think about encryption and certificates.",
         "expected_concepts": ["tls encryption", "plaintext vs ciphertext", "certificates"]},
        {"difficulty": 1, "subtopic": "Application Layer", "answer_type": "explanation",
         "question": "How does DNS resolve a hostname like example.com into an address? "
                     "Walk through the lookup chain.",
         "hint": "Root -> TLD -> authoritative nameserver; responses may be cached.",
         "expected_concepts": ["hierarchy", "caching", "authoritative server"]},
    ],
    "Programming": [
        {"difficulty": 2, "subtopic": "Functions", "answer_type": "definition",
         "question": "Explain what it means for a function to be 'pure' and why pure "
                     "functions make code easier to test.",
         "hint": "Pure functions have no side effects and always return the same output "
                 "for the same input.",
         "expected_concepts": ["no side effects", "determinism", "testability"]},
        {"difficulty": 2, "subtopic": "Data Types", "answer_type": "comparison",
         "question": "Explain the difference between pass-by-value and pass-by-reference, "
                     "and the mutation surprises each can cause.",
         "hint": "One copies the argument; the other aliases the original and lets the "
                 "callee mutate it.",
         "expected_concepts": ["copy vs alias", "mutation", "call semantics"]},
        {"difficulty": 2, "subtopic": "Debugging & Testing", "answer_type": "explanation",
         "question": "Describe your debugging workflow for a program that works locally "
                     "but fails only in production.",
         "hint": "Reproduce first, then isolate via logs, binary search of changes, and "
                 "a stack trace before fixing.",
         "expected_concepts": ["reproduce", "isolation", "observability"]},
        {"difficulty": 1, "subtopic": "Error Handling", "answer_type": "explanation",
         "question": "What is the difference between exceptions and error codes as "
                     "failure-handling styles? When would you prefer each?",
         "hint": "Exceptions unwind the stack automatically; error codes force explicit "
                 "per-call handling.",
         "expected_concepts": ["unwinding", "explicit handling", "recoverable errors"]},
        {"difficulty": 2, "subtopic": "Functions", "answer_type": "explanation",
         "question": "What is a closure? Give a small example and say what state it captures.",
         "hint": "A function that remembers variables from the scope where it was defined.",
         "expected_concepts": ["captured scope", "lexical environment", "state retention"]},
    ],
}
MODE = {}


def _topic_index(topic: str) -> int:
    try:
        return C.TOPICS.index(topic)
    except ValueError:
        return 0


def _persona_block(persona_name: str) -> str:
    p = C.PERSONAS.get(persona_name) or C.PERSONAS["faang"]
    return (f"Persona: {p['name']}. Traits: {p['traits']}. "
            f"Hint style: {p['hint_style']}. Feedback style: {p['feedback_style']}.")


def _mode_directive(mode: str, hint_policy: str) -> str:
    if mode == "mock":
        return ("This is a mock interview: no scaffolded hints, time pressure, realistic pacing, "
                "ask exactly one question per turn." if hint_policy != "full" else
                "Mock interview but the policy explicitly allows one pointed hint.")
    if mode == "weakness":
        return "Weakness-training mode: probe the candidate's weak topic, allow hints, verify understanding."
    return ("Practice mode: warm tone, allow hints, one focused question per turn.")


def _ask_question(plan, persona_name: str, mode: str, memories_text: str,
                  topic: str, profile_context: str = "",
                  recent_questions=()) -> tuple[dict, dict]:
    attempts = 3
    repair: list[str] = []
    for _ in range(attempts):
        draft = _generate_question(plan, persona_name, mode, memories_text,
                                   topic, profile_context, repair=repair)
        if draft is None:
            break  # LLM unreachable -> curated bank
        vq = validate_question(draft, plan=plan, recent_questions=recent_questions,
                               use_llm_judge=True)
        if vq["ok"]:
            return vq["normalized"], vq
        repair = vq["failures"]
        print(f"# question rejected ({', '.join(repair)}); regenerating...")

    return _fallback_bank(plan, persona_name, mode, memories_text,
                          topic, profile_context, recent_questions)


def _curated_draft(entry: dict, plan) -> dict:
    """Turn a CURATED_BANK entry into a validator-ready draft for this plan."""
    return {
        "topic": plan.topic,
        "difficulty": int(entry.get("difficulty", plan.difficulty)),
        "question": entry["question"],
        "hint": entry.get("hint", "") if plan.hint_policy != "none" else "",
        "subtopic": entry.get("subtopic", ""),
        "answer_type": entry.get("answer_type", "free_text"),
        "expected_concepts": entry.get("expected_concepts", []),
    }


def _fallback_bank(plan, persona_name: str, mode: str, memories_text: str,
                   topic: str, profile_context: str,
                   recent_questions) -> tuple[dict, dict]:
    entries = CURATED_BANK.get(topic) or CURATED_BANK["Programming"]
    ordered = sorted(entries, key=lambda e: (abs(int(e.get("difficulty", 3))
                                                 - plan.difficulty),))
    rejected: list[str] = []

    # (a) next-best fully-valid candidate
    for e in ordered:
        draft = _curated_draft(e, plan)
        vq = validate_question(draft, plan=plan, recent_questions=recent_questions,
                               use_llm_judge=False, trusted=True)
        if vq["ok"]:
            return vq["normalized"], vq
        rejected.extend(vq["failures"])

    # (b) generate a fresh distinct question at the same topic/difficulty
    stale = [r for r in recent_questions if r][-5:]
    fresh = _generate_question(
        plan, persona_name, mode, memories_text, topic, profile_context,
        repair=["Create a NEW question that is NOT a reworded repeat of any of "
                "these recently served questions: " + " || ".join(stale)])
    if fresh:
        vq = validate_question(fresh, plan=plan, recent_questions=recent_questions,
                               use_llm_judge=False)
        if vq["ok"]:
            return vq["normalized"], vq

    # (c) widen sub-topic pool: least-similar entry (whole-bank exhausted)
    best, best_sim = None, 2.0
    for e in entries:
        sim = max((question_similarity(e["question"], r) for r in stale if r),
                  default=0.0)
        if sim < best_sim:
            best, best_sim = e, sim
    draft = _curated_draft(best, plan)
    vq = validate_question(draft, plan=plan, recent_questions=recent_questions,
                           use_llm_judge=False, trusted=True)
    print(f"# warning: whole curated bank for {topic} (diff {plan.difficulty}) "
          f"collides with recent questions; widened to least-similar candidate "
          f"(sim={best_sim:.2f}). Rejected along the way: "
          f"({' | '.join(sorted(set(rejected)))})")
    return vq["normalized"], vq


def _generate_question(plan, persona_name: str, mode: str, memories_text: str,
                       topic: str, profile_context: str = "",
                       repair=None) -> dict | None:
    """Ask the LLM for ONE question plus self-describing metadata. None offline."""
    persona = _persona_block(persona_name)
    modal = _mode_directive(mode, plan.hint_policy)
    sub_list = ", ".join(subtopics_for(topic) or [])
    validation_rule = (
        "Before answering, self-check: the question must be complete and "
        f"solvable as-is; difficulty MUST be exactly {plan.difficulty}/5; "
        f"subtopic MUST be one of [{sub_list}] (or empty if truly none applies); "
        "the hint MUST point at this exact problem; answer_type MUST be one of "
        "[code, pseudocode, complexity, definition, explanation, comparison, "
        "scenario, free_text]."
    )
    system = (
        "You are a technical interviewer generating ONE interview question. "
        + persona + "\n" + modal + "\n"
        + (profile_context + "\n" if profile_context else "")
        + (memories_text + "\n" if memories_text else "")
        + (("A previous draft was rejected because: " + "; ".join(repair)
            + ". Fix every listed issue.\n") if repair else "")
        + "If hints are allowed, propose one pointed hint. Return a JSON object with "
          "keys: topic, difficulty (1-5), question (single question, no preamble, no lists), "
          "hint (string, empty if no hints), expected_concepts (array of up to 4 strings), "
          "subtopic (string name of the focus area within the topic, or empty if not applicable), "
          "answer_type (string from the allowed set).\n"
        + validation_rule
    )
    user = f"Ask one {topic} question at difficulty {plan.difficulty}/5, then a hint."
    try:
        out = LLM.chat_json(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.4, retries=2)
    except Exception:
        return None
    if not isinstance(out, dict):
        return None
    out.setdefault("topic", topic)
    out.setdefault("difficulty", plan.difficulty)
    if not str(out.get("question") or "").strip():
        return None
    return out


def _parse_topics(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return list(C.TOPICS)
    out: list[str] = []
    for name in raw.split(","):
        name = name.strip()
        if not name:
            continue
        canon = next((t for t in C.TOPICS if t.lower() == name.lower()), None)
        if canon is None:
            print(f"# warning: unknown topic {name!r} ignored "
                  f"(known: {', '.join(C.TOPICS)})")
        elif canon not in out:
            out.append(canon)
    return out or list(C.TOPICS)


def _candidate_id(args) -> str:
    if args.id:
        return args.id  # raw state key; registry entry is created only after an

    if args.candidate:
        target = args.candidate.strip().lower()
        hit = registry.find(target) or registry.find_by_name(target)
        if hit is None:
            print(f"# no registered profile yet for {target!r} - attending as a new "
                  f"candidate; they are registered only after this interview is saved.")
            return target
        return hit["id"]

    return registry.select()


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="LLM-DRIVEN ADAPTIVE INTERVIEW PREPARATION AGENT",
        description="Adaptive technical interview practice (RL + ML + memory).")
    ap.add_argument("--candidate", default="",
                    help="registered candidate id or name (interactive pick if omitted)")
    ap.add_argument("--id", default="",
                    help="force a raw state key (bypasses registry; Q-table/profile/memory "
                         "keyed exactly on this string)")
    ap.add_argument("--style", choices=["kind", "rude"], default="kind",
                    help="interviewer demeanour used for NEW candidates created interactively")
    ap.add_argument("--create", action="store_true",
                    help="with --candidate: register as a new candidate instead of resolving")

    ap.add_argument("--list", dest="show_list", action="store_true",
                    help="show the candidate registry and exit")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--topics", default=None,
                    help="comma-separated topics for this session (e.g. DSA,DBMS); "
                         "an interactive multi-select menu is shown when omitted")
    ap.add_argument("--mode", choices=C.MODE_NAMES, default="practice")
    ap.add_argument("--persona", choices=list(C.PERSONAS.keys()), default="faang")
    ap.add_argument("--answers", choices=["typed", "kind", "rude"], default="typed",
                    help="how answers are produced: typed (interactive, default) or a "
                         "scripted bank (kind/rude) for unattended runs")
    args = ap.parse_args()

    typed = args.answers == "typed"

    if args.show_list:
        entries = registry.list_all()
        print("# Candidate registry:")
        if not entries:
            print("#   (empty - first interview will register you)")
        for e in entries:
            print(f"#   {e['id']:<14} {e['name']:<20} style={e['style']}  "
                  f"sessions={e['sessions']}  created={e['created_at']}  "
                  f"last={e['last_active']}")
        return 0

    if args.create and args.candidate:
        candidate_id = args.candidate.strip().lower()
        print(f"# will register {args.candidate!r} as id={candidate_id} "
              f"(style={args.style}) after their first attended session")
        create_flag = True
    else:
        candidate_id = _candidate_id(args)
        create_flag = False

    style = args.style
    if args.id:
        base = registry.find(args.id)
        style = base["style"] if base else args.style
    else:
        entry = registry.find(candidate_id)
        style = entry["style"] if entry else args.style

    topics = _parse_topics(args.topics)

    clf = get_classifier()
    profile = CandidateProfile.load(candidate_id)
    memory = CandidateMemoryStore.load(candidate_id)
    predictor = PerformancePredictor.load()
    learner = Q.QLearner.load(table_id=candidate_id)
    engine = ActionEngine()
    recorder = SessionRecorder(candidate_id=candidate_id, mode=args.mode,
                               persona_name=args.persona, model=C.CHAT_MODEL,
                               topics=topics)

    print(f"# LLM-DRIVEN ADAPTIVE INTERVIEW PREPARATION AGENT — live session  candidate={candidate_id!r}  style={style}  "
          f"mode={args.mode}  persona={args.persona}  topics={','.join(topics)}  "
          f"answers={'typed' if typed else args.answers}")
    print(f"# sentiment={'ML' if clf.uses_ml() else 'rules'}  "
          f"performance={predictor.uses_ml()}  llm={C.CHAT_MODEL}")

    recorder = SessionRecorder(candidate_id=candidate_id, mode=args.mode,
                               persona_name=args.persona, model=C.CHAT_MODEL,
                               topics=topics)

    difficulty = plan_diff = 2
    prev_topic = profile.weakness(topics=topics)
    prev_mastery = 0.40
    recent_questions = _recent_questions(profile, candidate_id)

    for turn in range(1, args.turns + 1):
        skill_band = profile.skill_band()
        topic_idx = _topic_index(prev_topic)

        action = learner.choose(skill_band, difficulty, topic_idx)
        plan = engine.apply(action, profile, memory,
                            current_topic=prev_topic, difficulty=difficulty,
                            topic_pool=topics)
        if args.mode == "weakness" and plan.action != "REVISIT_MISTAKE" \
                and plan.topic != profile.weakness(prev_topic, topics=topics):
            plan.topic = profile.weakness(prev_topic, topics=topics)
            plan.prompt_directive = ("Weakness training: force this session's focus onto the "
                                     "candidate's weakest topic.")
        plan_diff = plan.difficulty

        memory_text = memory.to_context(
            memory.retrieve(query=f"{plan.topic} {action} interview answer", k=C.MEM_TOP_K))

        profile_context = profile.prompt_context(plan.topic)
        draft, qval = _ask_question(
            plan, args.persona, args.mode, memory_text, plan.topic,
            profile_context, recent_questions=recent_questions)
        question = draft["question"]
        hint = draft["hint"]
        q_subtopic = draft["subtopic"]
        recent_questions.append(question)
        recent_questions = recent_questions[-RECENT_WINDOW:]

        if typed:
            print(f"Q : {question}")
            if hint:
                print(f"   hint: {hint}")
            answer = _read_answer()
            print()
        else:
            answer = _bank(args.answers).get(plan.topic, KIND_BANK["Programming"])

        ev = evaluate_answer(question, answer, plan.topic, plan_diff,
                             persona_traits=_persona_block(args.persona),
                             memories_context=memory_text)
        mastery = float(ev["correctness"])

        subtopic = resolve_subtopic(plan.topic, question, q_subtopic)
        profile.record(plan.topic, mastery, hinted=plan.hint_policy == "full"
                       or bool(hint), question=question, gist=hint or "",
                       subtopic=subtopic, difficulty=plan_diff,
                       misconceptions=ev.get("misconceptions") or [],
                       followup=(action == "ASK_FOLLOWUP"))
        memory.add_answer_memory(plan.topic, turn, question, answer, ev)

        reward = Q.reward_for(mastery, prev_mastery, action)
        learner.update(skill_band, difficulty, topic_idx, action, reward, plan_diff)

        from ml.features import build_features
        feat = build_features(profile, plan.topic, plan_diff, bool(hint))
        p_ok, used_ml = predictor.predict_accuracy(feat)

        recorder.add_turn(turn, plan.topic, plan_diff, action, question, answer,
                          ev, hint_policy=plan.hint_policy,
                          question_validation=qval)

        qflag = "ok" if qval["ok"] else "FAIL"
        aflag = "valid" if ev.get("is_valid_answer", True) else "invalid"
        print(f"\n--- turn {turn} | action={action} -> {plan} | eval={'LLM' if ev['used_llm'] else 'rules'} "
              f"| qvalid={qflag} avalid={aflag} | P(correct)={p_ok:.2f}{'(ML)' if used_ml else '(rules)'} "
              f"| answers={'typed' if typed else 'scripted'}")
        if not typed:
            print(f"Q : {question}")
            if hint:
                print(f"   hint: {hint}")
        print(f"A ({'you' if typed else args.answers}): {answer}")
        print(f"   correctness={mastery:.2f}  reward={reward:+.2f}  "
              f"skill[{plan.topic}]={profile.skills.get(plan.topic, 0):.2f}  "
              f"{('sub=' + subtopic + '  ') if subtopic else ''}"
              f"memory={len(memory.entries)} entries  next_diff={plan_diff}")
        if ev.get("feedback"):
            print(f"   feedback: {ev['feedback']}")

        prev_topic, prev_mastery, difficulty = plan.topic, mastery, plan_diff
        time.sleep(0.2)

    learner.save()
    memory.save()

    summary = recorder.summary()
    if summary.get("turns", 0) > 0:
        profile.record_session(summary.get("avg_correctness"))
    profile.save()
    path = recorder.save()

    if create_flag:
        # The candidate ATTENDED (session persisted) -> now it is safe to register.
        registered_id, _ = registry.register(candidate_id, style=style)
        print(f"\n# registered {registered_id!r} (style={style}) -- came, attended, saved "
              f"sessions=1")

    if not registry.find(candidate_id):
        registry.register(candidate_id, style=style)
        print(f"# registered {candidate_id!r} after attending session (style={style})")

    print("\n# saved: q_table ->", learner.table_path)
    print("# saved: memory ->", memory.path())
    print("# saved: profile ->", profile.path())
    print("# saved: session ->", path)
    return 0


def _bank(style: str) -> dict:
    return RUDE_BANK if style == "rude" else KIND_BANK


def _recent_questions(profile, candidate_id: str) -> list[str]:
    """Seed the near-duplicate gate from past mistakes + the latest session."""
    seen: set[str] = set()
    out: list[str] = []
    for q in ([m.get("question") or "" for m in profile.mistakes]
              + _latest_session_questions(candidate_id)):
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[-RECENT_WINDOW:]


def _latest_session_questions(candidate_id: str) -> list[str]:
    latest = SessionRecorder.load_latest(candidate_id)
    if not latest:
        return []
    return [t.get("question") or "" for t in latest.get("turns", [])]


def _read_answer() -> str:
    """Prompt the human candidate for a free-form answer (single line)."""
    answer = input("YOUR ANSWER: ").strip()
    while not answer:
        answer = input("Please type an answer (empty is not allowed): ").strip()
    return answer


if __name__ == "__main__":
    sys.exit(main())
