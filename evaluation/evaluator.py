import re
import json

from llm_client import chat_json
from sentiment import get_classifier
from config import TEMP_STRUCTURED, CHAT_MODEL, RL_ACTIONS

try:
    from finetuning import infer_lora as _FINETUNE
except Exception:
    _FINETUNE = None

ALLOWED_ACTIONS = set(RL_ACTIONS)

_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid_answer": {"type": "boolean"},
        "correctness": {"type": "number"},
        "score": {"type": "number"},
        "conceptual_understanding": {"type": "number"},
        "communication": {"type": "number"},
        "missing_concepts": {"type": "array", "items": {"type": "string"}},
        "mistakes": {"type": "array", "items": {"type": "string"}},
        "misconceptions": {"type": "array", "items": {"type": "string"}},
        "feedback": {"type": "string"},
        "recommended_action": {"type": "string"},
    },
    "required": ["is_valid_answer", "correctness", "score",
                 "conceptual_understanding", "communication",
                 "missing_concepts", "mistakes", "misconceptions",
                 "feedback", "recommended_action"],
}

RULES_CONCEPT_KEYWORDS = {
    "linked list / two pointers / cycle": [
        "linked list", "singly", "cycle", "floyd", "tortoise", "hare",
        "fast pointer", "slow pointer", "two pointers", "node", "dot reference"],
    "hash map / lookup": ["hash", "hashmap", "hash map", "dictionary", "bucket",
                          "key-value", "o(1"],
    "algorithmic complexity": ["o(", "time complexity", "space complexity",
                               "big o", "linear", "logarithmic", "quadratic",
                               "constant time"],
    "search / binary search": ["binary search", "sorted", "midpoint", "divide"],
    "recursion": ["recursion", "recursive", "base case", "call stack"],
    "indexing": ["index", "clustered", "non-clustered", "b-tree", "btree",
                 "primary key", "secondary"],
    "normalization": ["normal", "redundancy", "dependency", "anomaly", "1nf",
                      "2nf", "3nf"],
    "transactions / ACID": ["acid", "atomic", "commit", "rollback", "isolation",
                            "durability", "transaction", "consistency"],
    "sql joins": ["join", "inner", "left", "right", "outer", "cartesian"],
    "process vs thread": ["process", "thread", "address space", "stack",
                          "scheduler", "cpu", "multithread"],
    "scheduling": ["schedul", "round robin", "priority", "quantum", "preempt",
                   "fifo", "sjf"],
    "deadlock": ["deadlock", "mutual exclusion", "hold and wait", "no preempt",
                 "circular wait"],
    "memory / paging": ["virtual memory", "page", "paging", "tlb", "swap",
                        "segment", "mmu"],
    "overloading vs overriding": ["overload", "override", "compile time",
                                  "run time", "runtime", "signature", "static",
                                  "dynamic dispatch"],
    "encapsulation": ["encapsul", "private", "data hiding", "access modifier"],
    "abstraction / interfaces": ["abstraction", "interface", "abstract class",
                                 "implementation details"],
    "polymorphism": ["polymorph", "virtual", "overrid", "subclass"],
    "inheritance / composition": ["inheritance", "composition", "is-a", "has-a",
                                  "extends", "derived"],
    "tcp handshake": ["tcp", "handshake", "syn", "ack", "three-way", "segment",
                      "connection"],
    "tcp vs udp": ["tcp", "udp", "connection-oriented", "connectionless",
                   "reliable", "datagram"],
    "osi model": ["osi", "layer", "presentation", "network layer", "data link",
                  "encapsulation", "protocol stack"],
    "dns": ["dns", "domain", "resolve", "nameserver"],
    "http": ["http", "https", "request", "response", "method", "status code"],
    "pure function": ["pure", "side effect", "deterministic", "same output",
                      "no side effects"],
    "pass by value / reference": ["pass by value", "pass by reference", "by ref",
                                  "copy", "aliasing"],
    "null vs undefined": ["null", "undefined", "none", "missing value"],
    "testing": ["test", "unit", "assert", "mock", "coverage", "tdd"],
    "debugging": ["debug", "breakpoint", "stack trace", "logs", "reproduce"],
}


def _extract_concepts(question: str) -> list[str]:
    """The 2-4 concept families this question is built on (deterministic)."""
    low = (question or "").lower()
    out: list[str] = []
    for concept, keywords in RULES_CONCEPT_KEYWORDS.items():
        if any(k in low for k in keywords):
            out.append(concept)
        if len(out) == 4:
            break
    return out or ["core_technical_answer"]


def _clamp01(v, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _strict_num(v) -> float:
    """Numeric score or TypeError (bools and junk are rejected)."""
    if isinstance(v, bool) or v is None:
        raise TypeError("not a numeric score")
    return float(v)


def _str_list(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [s for s in (str(x).strip() for x in v) if s]


def normalize_evaluation(ev: dict, answer_valid: bool) -> tuple[dict, list[str]]:
    failures: list[str] = []

    try:
        correctness = _strict_num(ev.get("correctness"))
    except (TypeError, ValueError):
        failures.append("correctness_missing_or_not_numeric")
        correctness = 0.0
    correctness = max(0.0, min(1.0, correctness))

    conceptual = _clamp01(ev.get("conceptual_understanding"), correctness)
    communication = _clamp01(ev.get("communication"), 0.5)

    try:
        score = max(0.0, min(1.0, _strict_num(ev.get("score"))))
    except (TypeError, ValueError):
        score = round(0.7 * correctness + 0.2 * conceptual + 0.1 * communication, 3)
        failures_score = "score_missing_or_not_numeric_recomputed"
        # repaired, not fatal - correctness (the number that matters) was valid
        ev.setdefault("_repaired", []).append(failures_score)

    is_valid = bool(ev.get("is_valid_answer", True)) and bool(answer_valid)

    action = str(ev.get("recommended_action") or "").strip().upper()
    if action not in ALLOWED_ACTIONS:
        action = "ASK_FOLLOWUP"
        ev.setdefault("_repaired", []).append("recommended_action_normalized")

    feedback = str(ev.get("feedback") or "").strip()
    if not feedback:
        feedback = "No detailed feedback returned."
        ev.setdefault("_repaired", []).append("feedback_defaulted")

    out = dict(ev)
    out["is_valid_answer"] = is_valid
    out["correctness"] = round(correctness, 3)
    out["score"] = round(score, 3)
    out["conceptual_understanding"] = round(conceptual, 3)
    out["communication"] = round(communication, 3)
    out["missing_concepts"] = _str_list(ev.get("missing_concepts"))
    out["mistakes"] = _str_list(ev.get("mistakes"))
    out["misconceptions"] = _str_list(ev.get("misconceptions"))
    out["feedback"] = feedback
    out["recommended_action"] = action
    if not is_valid:
        # unscorable answer: never let it move skill / reward upward
        out["correctness"] = 0.0
        out["score"] = 0.0
    return out, failures


def _rule_evaluate(question: str, answer: str) -> dict:
    sentiment, stance = get_classifier().predict(answer)
    from sentiment import rapport_delta
    rd = rapport_delta(sentiment, stance)
    low = answer.lower()
    words = len(re.findall(r"\S+", answer))
    length_e = 0.02 if words >= 40 else (0.0 if words >= 15 else -0.05)

    scores: list[float] = []
    covered = 0
    for concept in _extract_concepts(question):
        keywords = RULES_CONCEPT_KEYWORDS.get(concept, ())
        hits = sum(1 for k in keywords if k in low)
        ratio = hits / len(keywords) if keywords else 0.0
        s = 1.0 if ratio >= 0.5 else (0.5 if hits >= 1 else 0.0)
        if words < 4 and s == 1.0 and ratio < 1.0:
            s = 0.5  # too terse to claim full coverage
        scores.append(s)
        if s > 0.0:
            covered += 1

    correctness = (sum(scores) / len(scores)) if scores else 0.0
    if any(s == 1.0 for s in scores):
        correctness = max(correctness, 0.4)   # rubric sanity gate
    correctness = round(min(1.0, correctness), 3)

    if correctness < 0.4:
        rec = "GIVE_HINT"
    elif correctness < 0.6:
        rec = "ASK_EASIER"
    elif correctness < 0.8:
        rec = "ASK_FOLLOWUP"
    else:
        rec = "ASK_HARDER"

    n_concepts = len(scores)
    feedback = (
        f"Fallback verifier (sentiment={sentiment}, stance={stance}): "
        f"correctness {correctness:.2f}, {covered}/{n_concepts} core concepts "
        "addressed (1.0=covered / 0.5=gestured / 0.0=missing). "
        + ("Strong conceptual coverage of the question's core ideas."
           if correctness >= 0.6 else
           "Several core concepts were missed or only gestured at; restructure "
           "the answer around them.")
    )
    conceptual = max(0.05, min(0.95, correctness + 0.05))
    communication = max(0.05, min(0.95, 0.5 + 0.2 * length_e))
    score = round(0.7 * correctness + 0.2 * conceptual + 0.1 * communication, 3)
    return {
        "is_valid_answer": True,
        "correctness": correctness,
        "score": score,
        "conceptual_understanding": round(conceptual, 3),
        "communication": round(communication, 3),
        "missing_concepts": [],
        "mistakes": [],
        "misconceptions": [],
        "feedback": feedback,
        "recommended_action": rec if rec in ALLOWED_ACTIONS else "ASK_FOLLOWUP",
        "used_llm": False,
        "sentiment": sentiment,
        "stance": stance,
        "rapport_delta": rd,
    }


def _llm_evaluate(question: str, answer: str, topic: str, difficulty: int,
                  persona_traits: str, memories_context: str) -> dict | None:
    system = (
        "You are the ANSWER VERIFIER for an adaptive interview system. You grade "
        "SEMANTIC meaning - never string or keyword matching. Follow this rubric:\n"
        "1. Identify the 2-4 core concepts this question is actually testing.\n"
        "2. Check the answer against EACH core concept independently. Ignore "
        "grammar, typos, word-order and OCR-style garble UNLESS the noise changes "
        "meaning. Per concept score: 1.0 = clearly correct, 0.5 = gestured at but "
        "vague/incomplete, 0.0 = missing or wrong. correctness = the average of the "
        "per-concept scores, NOT a keyword match and NOT a binary.\n"
        "3. Two failure modes, never conflated: 'right idea, bad expression' -> "
        "correctness 0.6-0.85; 'wrong or absent understanding' (e.g. 'abc', "
        "off-topic, contradicts the concept) -> correctness 0.0-0.15.\n"
        "4. Sanity check before finalizing: if correctness < 0.2 then the answer "
        "must contain NO correct core concept; if it contains at least one clearly "
        "correct concept, correctness cannot be below 0.4. If you were about to "
        "output a score violating this, rescore the per-concept values.\n"
        "5. reward/skill deltas downstream scale with correctness - scoring an "
        "answer above 0.6 must never yield a near-zero reward, so be honest with "
        "the number.\n"
        "Return ONLY JSON matching this exact schema (no extra, renamed or removed "
        "fields):\n"
        f"{_EVAL_SCHEMA}\n"
        f"Interviewer persona: {persona_traits}\n"
        f"Topic: {topic}, difficulty {difficulty}/5.\n" +
        (memories_context + "\n" if memories_context else "")
    )
    prompt = (
        f"Question: {question}\n\nCandidate answer:\n{answer}\n\n"
        "Apply the rubric: name the core concepts, score each, average them into "
        "correctness. is_valid_answer: false ONLY if the answer is "
        "empty/unscorable gibberish or a refusal to attempt; otherwise true.\n"
        "score: overall 0..1 (holistic; may differ from correctness). "
        "missing_concepts: concepts a strong answer would include. "
        "mistakes: concrete errors made (short phrases). "
        "misconceptions: underlying wrong beliefs revealed (short phrases). "
        "feedback: 1-2 sentences. recommended_action: one of "
        "ASK_EASIER, ASK_SAME_DIFFICULTY, ASK_HARDER, ASK_WEAK_TOPIC, ASK_NEW_TOPIC, "
        "REVISIT_MISTAKE, GIVE_HINT, ASK_FOLLOWUP."
    )
    try:
        result = chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMP_STRUCTURED, retries=2)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    result["used_llm"] = True
    _, stance = get_classifier().predict(answer)
    result["stance"] = stance
    result["sentiment"] = "neutral"
    result["rapport_delta"] = 0
    return result


def evaluate_answer(question: str, answer: str, topic: str, difficulty: int,
                     persona_traits: str = "strict, precise, fair",
                     memories_context: str = "") -> dict:
    answer = answer or ""
    answer_present = bool(answer.strip())

    if not answer_present:
        ev = _rule_evaluate(question, ".")
        ev, _ = normalize_evaluation(ev, answer_valid=False)
        ev["validation"] = {"ok": True, "failures": ["answer_empty"],
                            "source": "rules"}
        return ev

    # Try finetuned LoRA evaluator first if available
    lora_raw = None
    if _FINETUNE is not None:
        try:
            if getattr(_FINETUNE, "available", lambda: False)():
                lr = _FINETUNE.evaluate_structured(question, answer, topic)
                if isinstance(lr, dict):
                    if ("available" not in lr) or (lr.get("available") and not lr.get("parse_error")):
                        if "available" not in lr:
                            lora_raw = lr
                        else:
                            r = lr.get("raw", "")
                            m = re.search(r'\{.*\}', str(r), re.DOTALL)
                            if m:
                                try:
                                    lora_raw = json.loads(m.group(0))
                                except Exception:
                                    pass
                            if lora_raw is None:
                                obj = {}
                                for line in str(r).splitlines():
                                    mm = re.match(r'^\s*"([^"]+)"\s*:\s*(.*)$', line)
                                    if not mm:
                                        continue
                                    k = mm.group(1)
                                    v = mm.group(2).rstrip()
                                    if v.endswith(','):
                                        v = v[:-1]
                                    vl = v.lower()
                                    if vl == 'true':
                                        obj[k] = True
                                        continue
                                    if vl == 'false':
                                        obj[k] = False
                                        continue
                                    if re.match(r'^-?\d+(\.\d+)?$', v):
                                        obj[k] = float(v) if '.' in v else int(v)
                                        continue
                                    if v.startswith('[') and v.endswith(']'):
                                        try:
                                            obj[k] = json.loads(v)
                                        except Exception:
                                            obj[k] = v
                                        continue
                                    if v.startswith('"') and v.endswith('"'):
                                        obj[k] = v[1:-1]
                                        continue
                                    obj[k] = v
                                if obj:
                                    lora_raw = obj
                    else:
                        try:
                            lora_raw = json.loads(m.group(0))
                        except Exception:
                            # try to fix common cases
                            s = m.group(0)
                            # remove trailing garbage after last } if inside string? just parse what we have
                            try:
                                lora_raw = json.loads(s)
                            except Exception:
                                lora_raw = None
        except Exception:
            lora_raw = None

    raw = lora_raw if lora_raw is not None else _llm_evaluate(question, answer, topic, difficulty,
                        persona_traits, memories_context)
    if raw is None:
        ev = _rule_evaluate(question, answer)
        ev, _ = normalize_evaluation(ev, answer_valid=True)
        ev["validation"] = {"ok": True, "failures": [], "source": "rules"}
        return ev

    ev, failures = normalize_evaluation(raw, answer_valid=True)
    if failures:
        # the grader broke the contract - do NOT trust its numbers
        ev = _rule_evaluate(question, answer)
        ev, _ = normalize_evaluation(ev, answer_valid=True)
        ev["validation"] = {"ok": False, "failures": failures,
                            "source": "rules_fallback"}
        return ev

    src = "lora" if lora_raw is not None else "llm"
    ev["validation"] = {"ok": True, "failures": [], "source": src,
                        "repaired": ev.pop("_repaired", [])}
    return ev
