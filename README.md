# LLM-DRIVEN ADAPTIVE INTERVIEW PREPARATION AGENT

The **LLM-Driven Adaptive Interview Preparation Agent** is a fully local, adaptive **technical-interview practice agent**. A candidate sits a mock interview one question at a time, an LLM interviewer grades each answer on its technical content, and the *next* question is chosen by a Q-learning policy that learns, per candidate, which question shape works best. Everything runs on one machine — no cloud, no API keys, no database server.

## Highlights

- **Adaptive question selection** — a tabular Q-learning policy picks the next action (easier, harder, weak topic, revisit mistake, hint, follow-up, …) from the candidate's live skill state.
- **Persistent candidate profile** — per-topic and per-subtopic skill tracking with EWMA, evidence-gated strengths/weaknesses, mistake resolution and misconceptions.
- **Long-term memory** — k-nearest-neighbour retrieval over past interactions, injected into the interviewer prompt.
- **Validated questions** — every generated question passes a multi-check gate (topic, difficulty, solvability, hint match, near-duplicate detection) before it is shown, with automatic repair and a curated fallback bank.
- **Trust-first grading** — an LLM verifier plus a deterministic keyword-rules fallback; a malformed model verdict is discarded rather than trusted.
- **Fine-tuned evaluator** — a LoRA adapter trained on a hand-authored corpus provides a fast, offline, structured grader.
- **Runs offline** — curated questions, rule grading, hashed-n-gram embeddings and a linear fallback keep the app working with no model available.

## How it works

```
candidate answer -> evaluate (LoRA -> LLM -> rules) -> update profile
      ^                                                      |
      |                                                      v
 next question <- question validator <- ActionEngine <- Q-learning policy
```

Each turn: read state -> RL chooses an action -> action becomes a plan (topic, difficulty, hint policy) -> retrieve memories -> generate + validate a question -> collect answer -> grade -> update profile -> reward + Q-update -> record.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the `llama3.2` chat model:
  ```powershell
  ollama serve
  ollama pull llama3.2
  ollama pull nomic-embed-text   # optional; without it embeddings fall back to hashed n-grams
  ```
- Python packages in `requirements.txt`. `torch`, `transformers`, `peft` and `accelerate` are only needed for fine-tuning; the live interview path needs just `requests`, `scikit-learn` and `numpy`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python app/main.py                                   # interactive candidate picker
python app/main.py --list                             # show the candidate registry
python app/main.py --candidate test --turns 4         # interview a candidate
python app/main.py --candidate test --mode weakness   # drill weak topics
python app/main.py --candidate test --topics DSA,DBMS --turns 4
python app/main.py --candidate newperson --create --style rude
python app/main.py --id scratch --answers kind --turns 5   # unattended run
```

| Flag | Default | Meaning |
|---|---|---|
| `--candidate` | `""` | registry id or name; interactive picker when omitted |
| `--id` | `""` | force a raw state key, bypassing the registry |
| `--style` | `kind` | `kind` \| `rude` — demeanour for scripted answers |
| `--create` | off | register a new candidate |
| `--list` | off | print the registry and exit |
| `--turns` | `3` | number of questions |
| `--topics` | all | comma-separated syllabus subset for this session |
| `--mode` | `practice` | `practice` \| `mock` \| `weakness` |
| `--persona` | `faang` | interviewer persona (see `config.PERSONAS`) |
| `--answers` | `typed` | `typed` (interactive) \| `kind` \| `rude` (scripted) |

## Training

All models are trained locally and are **not committed** (see `.gitignore`); regenerate them with these scripts.

```powershell
python train_sentiment.py            # Random Forest sentiment/stance classifier -> sentiment_model.pkl
python -m ml.train_performance       # Gradient Boosting performance predictor -> data/models/performance_model.pkl
```

### Fine-tuning the answer evaluator (LoRA)

```powershell
python -m finetuning.dataset_builder     # -> data/finetune/eval_dataset.jsonl
python -m finetuning.train_lora --model HuggingFaceTB/SmolLM2-135M-Instruct --out finetuning/adapter
python -m finetuning.infer_lora --topic DSA --answer "..." --question "..."
```

The adapter is picked up automatically by `evaluation/evaluator.py` when present and used as the first-choice structured grader; otherwise the app falls back to the Ollama LLM and then to deterministic rules.

## Testing

```powershell
python test_profile.py     # 15 unit tests for the candidate profile
```

## Project structure

```
app/            entry point, turn loop, registry, session recorder
candidate/      candidate profile (skills, mistakes, strengths/weaknesses)
evaluation/     answer evaluator + question validator
ml/             performance predictor + its training script
rl/             Q-learning policy + action engine
memory/         k-NN long-term memory store
finetuning/     LoRA dataset builder, trainer, inference
data/           runtime JSON artifacts (gitignored; folders kept via .gitkeep)
config.py       single source of truth for topics, personas, modes, paths, hyperparameters
```

## Data & persistence

Everything is plain JSON on disk under `data/`, keyed by candidate id. Runtime artifacts — profiles, sessions, vectors, Q-tables, episode logs and trained binaries — are generated locally and excluded from git.

## Notes

- Personas change tone and hint style only — never scoring.
- The system degrades gracefully: if Ollama is unreachable it uses the curated question bank, keyword grading, fallback embeddings and a linear probability estimate, and still completes a session.
