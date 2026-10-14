import os

# LLM / Ollama
OLLAMA_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_EMBED_URL = f"{OLLAMA_URL}/api/embed"
OLLAMA_TAGS_URL = f"{OLLAMA_URL}/api/tags"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"   # embedding model pulled at setup (fallback available)
CHAT_TIMEOUT = 240
EMBED_TIMEOUT = 60
MAX_RETRIES = 2
TEMPERATURE = 0.7
TEMP_STRUCTURED = 0.25
MAX_TOKENS = 4096
EMBED_DIM = 768

# Paths (all persistence under this project)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MEMORY_DIR = os.path.join(DATA_DIR, "memories")
CANDIDATE_DIR = os.path.join(DATA_DIR, "candidates")
SESSION_DIR = os.path.join(DATA_DIR, "sessions")
MODEL_DIR = os.path.join(DATA_DIR, "models")
SUMMARIES_DIR = os.path.join(MEMORY_DIR, "summaries")
VECTORS_DIR = os.path.join(DATA_DIR, "vectors")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
RL_TABLES_DIR = os.path.join(DATA_DIR, "rl", "tables")

for _d in (DATA_DIR, MEMORY_DIR, CANDIDATE_DIR, SESSION_DIR, MODEL_DIR,
           SUMMARIES_DIR, VECTORS_DIR, LOGS_DIR, RL_TABLES_DIR):
    os.makedirs(_d, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "performance_model.pkl")
VECTOR_FILE = os.path.join(VECTORS_DIR, "memory_vectors.pkl")
PROFILE_FILE = os.path.join(CANDIDATE_DIR, "profile.json")
QTABLE_FILE = os.path.join(DATA_DIR, "q_table.json")
SKILL_FILE = os.path.join(CANDIDATE_DIR, "skills.json")

# Topics (spec section 20) + a bank-fallback flag
TOPICS = ["DSA", "DBMS", "Operating Systems", "OOP", "Computer Networks", "Programming"]

TOPIC_SUBTOPICS = {
    "DSA": ["Arrays", "Strings", "Linked Lists", "Stacks", "Queues",
            "Trees", "Graphs", "Recursion", "Dynamic Programming"],
    "DBMS": ["ER Model", "Normalization", "Indexing", "Transactions", "SQL Queries"],
    "Operating Systems": ["Processes", "Threads", "Scheduling",
                          "Memory Management", "File Systems", "Deadlocks"],
    "OOP": ["Classes & Objects", "Inheritance", "Polymorphism",
            "Encapsulation", "Abstraction"],
    "Computer Networks": ["OSI & TCP/IP Model", "Application Layer",
                          "Transport Layer", "Network Layer", "Data Link Layer"],
    "Programming": ["Syntax & Semantics", "Data Types", "Functions",
                    "Error Handling", "Debugging & Testing"],
}

DIFFICULTY_MIN = 1
DIFFICULTY_MAX = 5
DIFFICULTY_NAMES = ["Basic", "Easy", "Medium", "Hard", "Advanced"]

# Modes (spec section 19)
MODE_NAMES = ["practice", "mock", "weakness"]
MODE_LABELS = {
    "practice": "Practice — full hints, explanations, warm tone",
    "mock": "Mock Interview — minimal hints, strict timing, realistic",
    "weakness": "Weakness Training — targets weak topics, hints allowed",
}

# Interviewer personas (spec section 4A) — tone/style only, never scoring
PERSONAS = {
    "friendly_mentor": {
        "name": "Friendly Mentor",
        "traits": "warm, patient, encouraging, praises effort, gives generous scaffolded hints",
        "hint_style": "generous, step-by-step",
        "feedback_style": "gentle, net-positive, starts with what went well",
        "urgency": 0.2,
    },
    "strict_technical": {
        "name": "Strict Technical Interviewer",
        "traits": "formal, terse, precise, values correctness over kindness, calls gaps directly",
        "hint_style": "rare, single pointed hint only when stuck",
        "feedback_style": "direct, critique-first, ties to measurable criterion",
        "urgency": 0.6,
    },
    "faang": {
        "name": "FAANG-style Interviewer",
        "traits": "fast-paced, benchmark-focused, asks complexity/edge cases, minimal warmth",
        "hint_style": "almost none, expects independent reasoning",
        "feedback_style": "score-first, performance-oriented, brief",
        "urgency": 0.8,
    },
    "system_design": {
        "name": "System Design Interviewer",
        "traits": "architectural, probes trade-offs, scale, bottlenecks; collaborative around a whiteboard",
        "hint_style": "occasional gentle scaffold on structure",
        "feedback_style": "framework-based, evaluates structure/trade-offs",
        "urgency": 0.4,
    },
    "dsa": {
        "name": "DSA Interviewer",
        "traits": "algorithmic, complexity-focused, probes edge cases and proof of correctness",
        "hint_style": "pointed hints toward the right data structure",
        "feedback_style": "algorithm-correctness-and-complexity framed",
        "urgency": 0.55,
    },
    "hr": {
        "name": "HR / Behavioral Interviewer",
        "traits": "warm, structured around STAR, assesses soft skills, culture fit, communication",
        "hint_style": "coaches toward structured (STAR) answers",
        "feedback_style": "communication-and-behavior focused",
        "urgency": 0.3,
    },
}

# Memory system (spec section 5-7)
MEM_SUMMARIZE_EVERY = 4    # summarize short-term buffer every N turns
MEM_RECENT = 6             # inject last N turns verbatim into the prompt
MEM_TOP_K = 3              # k-NN retrievals injected into the prompt
MEM_EMBED_FALLBACK_BITS = 4096   # hashed-ngram fallback embedding size when Ollama embed is absent

# Performance predictor (spec section 8) — Gradient Boosting
ML_TOPIC_RANGE = 3                 # one-hot hot-encoded topic buckets kept small
ML_RANDOM_STATE = 42
ML_CONFIDENCE_FLOOR = 0.55         # below this the model defers to rules/profile
ML_MIN_SAMPLES = 20

# Q-learning (spec sections 10-13)
RL_ALPHA = 0.15            # learning rate
RL_GAMMA = 0.9             # discount
RL_EPSILON_START = 0.35
RL_EPSILON_MIN = 0.05
RL_EPSILON_DECAY = 0.9995
RL_STATES = [
    # (skill_band, difficulty_band) -- discretized performance bands
]
RL_ACTIONS = ["ASK_EASIER", "ASK_SAME_DIFFICULTY", "ASK_HARDER", "ASK_WEAK_TOPIC",
              "ASK_NEW_TOPIC", "REVISIT_MISTAKE", "GIVE_HINT", "ASK_FOLLOWUP"]
RL_REWARD_MASTERY = 1.0

# Seeds
SEED = 20260922
