import json
import os
import re
import time

import config as C

REGISTRY_PATH = os.path.join(C.CANDIDATE_DIR, "_registry.json")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _load() -> list[dict]:
    try:
        if not os.path.exists(REGISTRY_PATH):
            return []
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "candidate"


def register(name: str, style: str = "kind") -> tuple[str, str]:
    """Add a new candidate to the registry. Returns (id, display_name)."""
    entries = _load()
    base = _slug(name)
    slug = base
    n = 2
    while any(e["id"] == slug for e in entries):
        slug = f"{base}_{n}"
        n += 1
    display = name.strip() or f"candidate {len(entries) + 1}"
    entry = {
        "id": slug, "name": display.split("_")[0].title(),
        "created_at": _now(), "last_active": _now(),
        "sessions": 0, "style": style,
    }
    entries.append(entry)
    _save(entries)
    return slug, display


def find(cid: str) -> dict | None:
    return next((e for e in _load() if e["id"] == cid), None)


def find_by_name(name: str) -> dict | None:
    slug = _slug(name)
    return next((e for e in _load() if e["id"] == slug), None)


def touch(cid: str) -> None:
    """Bump last_active + session counter for a candidate after a session."""
    entries = _load()
    for e in entries:
        if e["id"] == cid:
            e["last_active"] = _now()
            e["sessions"] = e.get("sessions", 0) + 1
    _save(entries)


def list_all() -> list[dict]:
    return _load()


def select() -> str:
    entries = _load()
    try:
        return _select_interactive(entries)
    except EOFError:
        if entries:
            return max(entries, key=lambda e: e["last_active"])["id"]
        cid, _ = register("candidate 1")
        return cid


def _select_interactive(entries: list[dict]) -> str:
    if not entries:
        print("# LLM-DRIVEN ADAPTIVE INTERVIEW PREPARATION AGENT: no candidates registered yet.")
        name = input("#   Register a name (Enter = candidate 1): ").strip()
        cid, _ = register(name or "candidate 1")
        print(f"#   Registered as {cid}.")
        return cid

    print("# Registered candidates:")
    for i, e in enumerate(entries, 1):
        print(f"  [{i}] {e['name']}  (id={e['id']}, sessions={e.get('sessions', 0)}, "
              f"last={e['last_active']})")
    print("  [n] create a new candidate")

    while True:
        choice = input("# Pick a candidate (1..N, n=new, Enter=last used): ").strip().lower()
        if not choice:
            return max(entries, key=lambda e: e["last_active"])["id"]
        if choice in ("n", "new"):
            name = input("#   Name for the new candidate (Enter = default): ").strip()
            cid, _ = register(name or f"candidate {len(entries) + 1}")
            print(f"#   Registered as {cid}.")
            return cid
        try:
            idx = int(choice)
            if 1 <= idx <= len(entries):
                return entries[idx - 1]["id"]
        except ValueError:
            pass
        print("#   Try a number 1..N, 'n' for new, or just Enter.")
