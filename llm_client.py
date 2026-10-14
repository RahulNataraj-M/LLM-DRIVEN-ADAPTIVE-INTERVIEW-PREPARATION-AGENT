import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def chat_json(messages: list[dict], temperature: float = 0.4, retries: int = 3) -> dict:
    for attempt in range(retries):
        raw = chat(messages, temperature=temperature)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                return {
                    "dialogue": raw,
                    "action": "none",
                    "rapport_delta": 0,
                    "difficulty_delta": 0,
                }
    return {}
