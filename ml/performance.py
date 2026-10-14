import os
import pickle

import config as C
from ml.features import FEATURES

_TOPIC_DIM = 6  # len(config.TOPICS)


def _fallback_probability(feat: list[float]) -> float:
    skill = feat[0] * 0.0 + feat[2]       # feature order: topic,difficulty,skill,...
    accuracy = feat[3]
    difficulty = feat[1]
    hinted = feat[5]
    is_weak = feat[6]
    p = 0.45 + 0.55 * skill + 0.25 * (accuracy - 0.5) - 0.06 * (difficulty - 3)
    if hinted:
        p += 0.05
    if is_weak:
        p -= 0.15
    return min(0.97, max(0.03, p))


class PerformancePredictor:
    def __init__(self):
        self.model = None
        self.classes = None
        self.train_accuracy = 0.0

    def uses_ml(self) -> bool:
        return self.model is not None

    def fit(self, X: list[list[float]], y: list[int]):
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.1, max_depth=3,
            random_state=C.ML_RANDOM_STATE)
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.train_accuracy = float(sum(a == b for a, b in zip(preds, y))) / len(y)
        self.classes = self.model.classes_

    def predict_accuracy(self, feat: list[float]) -> tuple[float, bool]:
        """Return (P(correct), used_ml)."""
        if not self.uses_ml():
            return _fallback_probability(feat), False
        try:
            proba = self.model.predict_proba([feat])[0]
            classes = list(self.classes)
            idx_true = classes.index(1) if 1 in classes else 1
            p = float(proba[idx_true]) if len(proba) > idx_true else _fallback_probability(feat)
            # defer to rules when the model is unsure
            if p < C.ML_CONFIDENCE_FLOOR and p > 1 - C.ML_CONFIDENCE_FLOOR:
                return _fallback_probability(feat), False
            return min(0.97, max(0.03, p)), True
        except Exception:
            return _fallback_probability(feat), False

    def save(self, path: str | None = None):
        path = path or C.MODEL_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self.model,
            "classes": self.classes,
            "train_accuracy": self.train_accuracy,
            "features": FEATURES,
            "model_type": "GradientBoostingClassifier",
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | None = None) -> "PerformancePredictor":
        path = path or C.MODEL_FILE
        p = cls()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                p.model = payload.get("model")
                p.classes = payload.get("classes")
                p.train_accuracy = payload.get("train_accuracy", 0.0)
            except Exception:
                p.model = None
        return p
