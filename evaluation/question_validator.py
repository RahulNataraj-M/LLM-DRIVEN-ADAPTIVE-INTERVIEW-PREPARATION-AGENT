import difflib
import re

import config as C
from candidate.profile import resolve_subtopic

ANSWER_TYPES = {"code", "pseudocode", "complexity", "definition", "explanation",
                "comparison", "scenario", "free_text"}

ASK_MARKERS = ("what ", "why ", "how ", "when ", "where ", "which ", "explain",
               "describe", "walk ", "implement", "write ", "derive", "compare",
               "contrast", "analyze", "prove", "design", "define", "list ",
               "show ", "trace", "calculate", "difference", "walk me")

BROKEN_MARKERS = ("todo", "fixme", "{{", "}}", "<insert", "[insert",
                  "lorem ipsum", "\x00")

NEAR_DUPLICATE_AT = 0.82
RECENT_WINDOW = 12
DIFFICULTY_TOLERANCE = 1
MIN_Q_CHARS = 20
MAX_Q_CHARS = 700
HINT_COPY_AT = 0.95

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def question_similarity(a: str, b: str) -> float:
    """Max of sequence ratio and token Jaccard - catches reworded copies."""
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    ta, tb = _tokens(a), _tokens(b)
    jac = (len(ta & tb) / len(ta | tb)) if ta and tb else 0.0
    return max(ratio, jac)


def _hint_related(question: str, hint: str, concepts: list[str]) -> bool:
    """Hint must share content with the question (or its expected concepts)."""
    if not hint:
        return True
    qt, ht = _tokens(question), _tokens(hint)
    if not ht:
        return False
    if {t for t in ht & qt if len(t) >= 4}:
        return True
    cl = " ".join(concepts or []).lower()
    return any(t in cl for t in ht if len(t) >= 4)


def _looks_solvable(question: str) -> tuple[bool, str]:
    low = (question or "").lower()
    if not (MIN_Q_CHARS <= len(question) <= MAX_Q_CHARS):
        return False, f"length {len(question)} outside [{MIN_Q_CHARS}, {MAX_Q_CHARS}]"
    bad = [m for m in BROKEN_MARKERS if m in low]
    if bad:
        return False, f"broken marker(s): {', '.join(bad)}"
    if question.rstrip().endswith("?"):
        return True, ""
    if any(m in low for m in ASK_MARKERS):
        return True, ""
    return False, "no recognizable ask (no question mark, no ask verb)"


def _llm_judge(question: str, hint: str, topic: str,
               concepts: list[str]) -> dict | None:
    """Independent solvability / hint-correspondence judge (None if offline)."""
    from llm_client import chat_json
    system = (
        "You rigorously vet technical interview questions before they are shown. "
        "Return ONLY JSON: {\"solvable\": boolean, \"hint_matches\": boolean, "
        "\"reason\": string}. solvable = the question is well-posed and a "
        "competent candidate can actually answer it (no missing inputs, no "
        "contradictions). hint_matches = the hint points at THIS question's "
        "problem, or true when there is no hint."
    )
    user = (
        f"Topic: {topic}\n"
        f"Question: {question}\n"
        f"Hint: {hint or '(none)'}\n"
        f"Expected concepts: {', '.join(concepts) or '(none)'}"
    )
    try:
        out = chat_json(messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        temperature=0.0, retries=1)
    except Exception:
        return None
    return out if isinstance(out, dict) else None


def validate_question(draft: dict, plan=None, recent_questions=(),
                      use_llm_judge: bool = False,
                      trusted: bool = False,
                      recent_window: int = RECENT_WINDOW,
                      near_duplicate_at: float = NEAR_DUPLICATE_AT) -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict = {}

    q = str(draft.get("question") or "").strip()
    raw_topic = str(draft.get("topic") or "").strip()
    hint = str(draft.get("hint") or "").strip()
    raw_sub = str(draft.get("subtopic") or "").strip()
    atype = str(draft.get("answer_type") or draft.get("expected_answer_type")
                or "").strip().lower()
    concepts = [str(c) for c in (draft.get("expected_concepts") or [])
                if str(c).strip()]

    # -- plan policy: hints disallowed -> hint is cleared before checks ----
    hint_policy = getattr(plan, "hint_policy", "") if plan else ""
    if hint_policy == "none":
        hint = ""

    # -- 1. required input data -------------------------------------------
    missing = [k for k, v in (("question", q), ("topic", raw_topic)) if not v]
    plan_diff = getattr(plan, "difficulty", None) if plan else None
    if plan_diff is None and draft.get("difficulty") is None:
        missing.append("difficulty")
    checks["required_data"] = not missing
    if missing:
        failures.append(f"missing_required_data: {', '.join(missing)}")

    # -- 6a. known topic (checked early so later checks have a real topic) --
    if not raw_topic and plan is not None:
        warnings.append("topic_missing: falling back to plan topic")
        raw_topic = getattr(plan, "topic", "")
    topic = next((t for t in C.TOPICS if t.lower() == raw_topic.lower()), None)
    checks["known_topic"] = topic is not None
    if raw_topic and not topic:
        failures.append(f"unknown_topic: {raw_topic!r} not in {C.TOPICS}")
    elif topic and plan is not None and topic != getattr(plan, "topic", topic):
        failures.append(f"topic_mismatch: question is {topic!r} but plan asks "
                        f"for {plan.topic!r}")
        topic = plan.topic
    if topic is None and plan is not None:
        topic = getattr(plan, "topic", "")  # best-effort for downstream checks

    # -- 4. difficulty appropriate ----------------------------------------
    claimed = draft.get("difficulty")
    diff_ok = True
    if plan_diff is not None:
        target = int(plan_diff)
        if claimed is None:
            warnings.append("difficulty_missing: using plan difficulty")
        else:
            try:
                d = int(claimed)
                if d < C.DIFFICULTY_MIN or d > C.DIFFICULTY_MAX:
                    diff_ok = False
                    failures.append(f"difficulty_out_of_range: {d}")
                elif abs(d - target) > DIFFICULTY_TOLERANCE:
                    diff_ok = False
                    failures.append(
                        f"difficulty_inappropriate: {d} vs plan {target} "
                        f"(tolerance +-{DIFFICULTY_TOLERANCE})")
            except (TypeError, ValueError):
                diff_ok = False
                failures.append(f"difficulty_not_an_integer: {claimed!r}")
        difficulty = target
    else:
        try:
            difficulty = max(C.DIFFICULTY_MIN, min(C.DIFFICULTY_MAX, int(claimed)))
        except (TypeError, ValueError):
            difficulty = 2
            diff_ok = False
            failures.append(f"difficulty_not_an_integer: {claimed!r}")
    checks["difficulty_appropriate"] = diff_ok

    # -- 3. expected answer type exists ------------------------------------
    answer_type_ok = bool(atype) and atype in ANSWER_TYPES
    checks["answer_type_exists"] = answer_type_ok
    if not atype:
        failures.append("missing_answer_type")
    elif atype not in ANSWER_TYPES:
        failures.append(f"unknown_answer_type: {atype!r} not in "
                        f"{sorted(ANSWER_TYPES)}")

    # -- 2. solvable (deterministic heuristics) -----------------------------
    solvable, why = _looks_solvable(q) if q else (False, "empty question")
    checks["solvable"] = solvable
    if not solvable:
        msg = f"not_solvable: {why}"
        (warnings if trusted else failures).append(msg)

    # -- 5. hint corresponds to the actual problem --------------------------
    hint_ok = True
    if hint:
        if q and question_similarity(q, hint) >= HINT_COPY_AT:
            hint_ok = False
            failures.append("hint_duplicates_question")
        elif not _hint_related(q, hint, concepts):
            hint_ok = False
            msg = "hint_unrelated_to_question"
            (warnings if trusted else failures).append(msg)
    checks["hint_corresponds"] = hint_ok

    # -- 6b. known subtopic --------------------------------------------------
    sub_resolved = None
    if raw_sub:
        sub_resolved = resolve_subtopic(topic or "", q, raw_sub)
        if sub_resolved is None:
            failures.append(
                f"unknown_subtopic: {raw_sub!r} is not a subtopic of "
                f"{topic or raw_topic!r}")
            checks["known_subtopic"] = False
        elif sub_resolved.lower() != raw_sub.lower():
            warnings.append(f"subtopic_corrected: {raw_sub!r} -> {sub_resolved!r}")
            checks["known_subtopic"] = True
        else:
            checks["known_subtopic"] = True
    else:
        sub_resolved = resolve_subtopic(topic or "", q, None)
        if sub_resolved is None:
            failures.append(f"missing_subtopic: no known subtopic for "
                            f"{topic or raw_topic!r} could be resolved")
            checks["known_subtopic"] = False
        else:
            checks["known_subtopic"] = True

    # -- 7. sufficiently different from recent questions ---------------------
    recent = [r for r in (recent_questions or []) if r][-recent_window:]
    dup = None
    worst = 0.0
    for r in recent:
        sim = question_similarity(q, r)
        if sim > worst:
            worst = sim
        if sim >= near_duplicate_at:
            dup = r
            break
    checks["different_from_recent"] = dup is None
    checks["max_recent_similarity"] = round(worst, 3)
    if dup is not None:
        failures.append(f"near_duplicate: {worst:.2f} similar to a recent "
                        f"question ({dup[:60]!r})")

    # -- optional LLM judge (only when structural checks already passed) -----
    if use_llm_judge and not failures:
        verdict = _llm_judge(q, hint, topic or raw_topic, concepts)
        if verdict is None:
            warnings.append("llm_judge_unavailable: solvability not "
                            "independently confirmed (offline)")
        else:
            solvable_j = verdict.get("solvable")
            hint_j = verdict.get("hint_matches")
            reason = str(verdict.get("reason") or "").strip()
            if solvable_j is False:
                failures.append(f"llm_judge_unsolvable: {reason or 'no reason'}")
            if hint and hint_j is False:
                failures.append(f"llm_judge_hint_mismatch: {reason or 'no reason'}")
            if not isinstance(solvable_j, bool) or (
                    hint and not isinstance(hint_j, bool)):
                warnings.append("llm_judge_malformed: unexpected judge payload")
            checks["llm_judge"] = {"solvable": solvable_j,
                                   "hint_matches": hint_j, "reason": reason}

    normalized = {
        "topic": topic or raw_topic,
        "difficulty": int(difficulty),
        "question": q,
        "hint": hint,
        "subtopic": sub_resolved or "",
        "answer_type": atype,
        "expected_concepts": concepts,
    }
    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "normalized": normalized,
    }
