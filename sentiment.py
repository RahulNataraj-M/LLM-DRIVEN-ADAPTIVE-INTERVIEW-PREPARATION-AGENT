import os
import pickle

MODEL_FILE = "sentiment_model.pkl"

SENTIMENTS = ["negative", "neutral", "positive"]
STANCES = ["greet", "smalltalk", "ask", "goodwill", "hostile", "threat", "exit"]

# Coarse stance each fine-grained dataset intent is folded into.
INTENT_TO_STANCE = {
    "greet": "greet",
    "smalltalk": "smalltalk",
    "share": "smalltalk",
    "neutral_statement": "smalltalk",
    "question": "ask",
    "request_hint": "ask",
    "engage": "goodwill",
    "accept_challenge": "goodwill",
    "appreciate": "goodwill",
    "flattery": "goodwill",
    "insult": "hostile",
    "demand": "hostile",
    "threat": "threat",
    "farewell": "exit",
}

SENTIMENT_DELTA = {"negative": -1, "neutral": 0, "positive": 1}
STANCE_DELTA = {
    "greet": 0, "smalltalk": 0, "ask": 0,
    "goodwill": 1, "hostile": -1, "threat": -2, "exit": 0,
}


def rapport_delta(sentiment: str, stance: str) -> int:
    d = SENTIMENT_DELTA.get(sentiment, 0) + STANCE_DELTA.get(stance, 0)
    return max(-3, min(3, d))


KEYWORD_FEATURES = [
    # polarity
    "thank", "appreciate", "great", "good", "happy", "gladly", "please", "sure",
    "absolutely", "love", "nice", "helpful", "useful", "clear", "makes sense",
    "really", "solid", "better", "enjoy", "grateful", "welcome", "precise", "kind",
    # hostility
    "boring", "waste", "pointless", "useless", "stupid", "annoying", "shut up",
    "rambling", "prattling", "nonsense", "ridiculous", "fool", "stalling", "liar",
    "mock", "quit ", "cease", "despise", "fraud", "grudge", "cut this", "worst",
    # threats
    "walk out", "or else", "regret", "had enough", "threat", "leave", "done here",
    # greetings / farewell
    "hello", "hi ", "greetings", "good day", "good evening", "good morning",
    "thanks for", "goodbye", "bye", "farewell", "take my leave", "leaving",
    "on my way", "going",
    # hints / help
    "hint", "clarify", "clarification", "rephrase", "pointer", "guidance",
    "assistance", "example", "repeat", "elaborate",
    # engage
    "happy to", "let me", "i can", "i will", "walk through", "reason",
    "approach", "structure", "trade-offs", "complexity", "edge case",
    # questions
    "who ", "what ", "where ", "when ", "how ", "why ", "?", "could you",
    "can you", "may i", "tell me", "are you",
]
_NEG_FEATURE_IDX = [i for i, k in enumerate(KEYWORD_FEATURES) if k in {
    "boring", "waste", "pointless", "useless", "stupid", "annoying", "shut up",
    "rambling", "prattling", "nonsense", "ridiculous", "fool", "stalling", "liar",
    "mock", "quit ", "cease", "despise", "fraud", "grudge", "cut this", "worst",
    "walk out", "or else", "regret", "had enough", "threat", "done here"}]
_POS_FEATURE_IDX = [i for i, k in enumerate(KEYWORD_FEATURES) if k in {
    "thank", "appreciate", "great", "good", "happy", "gladly", "please", "sure",
    "absolutely", "love", "nice", "helpful", "useful", "clear", "makes sense",
    "really", "solid", "better", "enjoy", "grateful", "welcome", "precise", "kind"}]


def _build_features(text: str) -> list[int]:
    low = text.lower()
    feats = [1 if k in low else 0 for k in KEYWORD_FEATURES]
    feats.append(1 if "!" in low else 0)
    feats.append(min(1, len(low.split())))  # very short inputs
    return feats


class _RuleClassifier:
    """Deterministic fallback: explicit keyword dictionary lookup."""

    def __init__(self):
        self._greet = {"hello", "hi", "greetings", "good day", "good evening",
                       "good morning", "thanks for fitting", "thanks for having"}
        self._exit = {"goodbye", "bye", "leaving", "on my way", "farewell",
                       "take my leave", "going", "done"}
        self._hint = {"hint", "clarify", "clarification", "rephrase", "pointer",
                       "guidance", "example", "repeat", "elaborate"}
        self._engage = {"i will", "i can", "let me", "i could", "i would",
                        "happy to", "sure", "absolutely", "walk through",
                        "reason through"}
        self._ask = {"who ", "what ", "where ", "when ", "how ", "why ", "?",
                     "could you", "can you", "may i", "tell me", "are you"}
        self._threat = {"walk out", "or else", "regret", "had enough",
                        "will leave", "threat", "done here"}
        self._hostile = {"boring", "waste", "pointless", "useless", "stupid",
                         "annoying", "shut up", "rambling", "prattling",
                         "nonsense", "ridiculous", "fool", "stalling", "liar",
                         "mock", "quit", "cease", "despise", "fraud", "grudge"}

    def predict(self, text: str) -> tuple[str, str]:
        t = text.lower().strip()

        neg_hits = sum(1 for k in self._hostile | self._threat if k in t)
        pos_hits = sum(1 for k in ("thank", "appreciate", "great", "good", "nice",
                                   "happy", "gladly", "please", "sure",
                                   "absolutely", "helpful", "useful", "clear",
                                   "makes sense", "solid", "grateful", "love")
                       if k in t)
        if neg_hits > pos_hits:
            sentiment = "negative"
        elif pos_hits > neg_hits:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        if any(k in t for k in self._greet):
            stance = "greet"
        elif any(k in t for k in self._exit):
            stance = "exit"
        elif any(k in t for k in self._hostile):
            stance = "hostile"
        elif any(k in t for k in self._threat):
            stance = "threat"
        elif any(k in t for k in self._engage):
            stance = "goodwill" if sentiment != "negative" else "hostile"
        elif any(k in t for k in self._hint):
            stance = "ask" if sentiment == "neutral" else ("goodwill" if sentiment == "positive" else "hostile")
        elif any(k in t for k in self._ask):
            stance = "ask"
        else:
            stance = "smalltalk"
        return sentiment, stance


class SentimentClassifier:
    def __init__(self):
        self.sentiment_model = None
        self.stance_model = None
        self.sentiment_classes = list(SENTIMENTS)
        self.stance_classes = list(STANCES)
        self.rules = _RuleClassifier()

    def uses_ml(self) -> bool:
        return self.sentiment_model is not None

    def fit(self, entries: list[dict]):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [e["text"] for e in entries]
        sent = [e["sentiment"] for e in entries]
        stance = [e.get("stance") or INTENT_TO_STANCE.get(e["intent"], "smalltalk")
                  for e in entries]

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                          max_features=20000, sublinear_tf=True)
        X = self.vectorizer.fit_transform(texts)

        self.sentiment_model = RandomForestClassifier(
            n_estimators=220, class_weight="balanced", random_state=7, n_jobs=-1)
        self.sentiment_model.fit(X, sent)
        self.stance_model = RandomForestClassifier(
            n_estimators=220, class_weight="balanced", random_state=7, n_jobs=-1)
        self.stance_model.fit(X, stance)

    def train(self, entries: list[dict] | None = None):
        entries = entries if entries is not None else _entries_for_training()
        self.fit(entries)
        self.save()

    def predict(self, text: str) -> tuple[str, str]:
        # High-confidence ML path; otherwise fall back to the deterministic rules.
        ml = self._ml_predict(text)
        r = self.rules.predict(text)
        if ml is None:
            return r
        return ml

    def _ml_predict(self, text: str) -> tuple[str, str] | None:
        if not self.uses_ml():
            return None
        try:
            import numpy as np
            X = self.vectorizer.transform([text])
            sp = self.sentiment_model.predict_proba(X)[0]
            ip = self.stance_model.predict_proba(X)[0]
            s_idx, s_conf = int(sp.argmax()), float(sp.max())
            i_idx, i_conf = int(ip.argmax()), float(ip.max())
            if s_conf < 0.5 or i_conf < 0.4:
                return None
            return (str(self.sentiment_classes[s_idx]), str(self.stance_classes[i_idx]))
        except Exception:
            return None

    def save(self, path: str = MODEL_FILE):
        payload = {
            "vectorizer": self.vectorizer if self.uses_ml() else None,
            "sentiment_model": self.sentiment_model,
            "stance_model": self.stance_model,
            "sentiment_classes": self.sentiment_classes,
            "stance_classes": self.stance_classes,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str = MODEL_FILE) -> "SentimentClassifier":
        clf = cls()
        if os.path.exists(path):
            with open(path, "rb") as f:
                payload = pickle.load(f)
            clf.vectorizer = payload.get("vectorizer")
            clf.sentiment_model = payload.get("sentiment_model")
            clf.stance_model = payload.get("stance_model")
            clf.sentiment_classes = payload.get("sentiment_classes", list(SENTIMENTS))
            clf.stance_classes = payload.get("stance_classes", list(STANCES))
        return clf


def _entries_for_training():
    from sentiment_data import DATA
    return DATA


_singleton: SentimentClassifier | None = None


def get_classifier() -> SentimentClassifier:
    global _singleton
    if _singleton is None:
        _singleton = SentimentClassifier.load()
        if not _singleton.uses_ml():
            _singleton.train()
    return _singleton
