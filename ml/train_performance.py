import os
import random

import numpy as np

import config as C
from ml.features import FEATURES
from ml.performance import PerformancePredictor

N_TOPICS = len(C.TOPICS)
SEED = C.SEED


def _true_probability(feat: list[float]) -> float:
    """Ground-truth relationship for the bootstrap dataset (documented)."""
    _, difficulty, skill, accuracy, attempts, _hint, is_weak, prev5 = feat
    p = 0.42 + 0.52 * skill + 0.22 * (accuracy - 0.5) + 0.06 * min(1.0, attempts / 10)
    p -= 0.07 * (difficulty - 3)
    p -= 0.18 * is_weak
    p += 0.08 * (prev5 - 0.5)
    return min(0.96, max(0.04, p))


def build_dataset(n_rows: int = 800) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(SEED)
    X: list[list[float]] = []
    y: list[int] = []
    for _ in range(n_rows):
        topic_idx = rng.randrange(N_TOPICS)
        difficulty = rng.randint(1, 5)
        skill = rng.choice([
            rng.uniform(0.25, 0.45),   # weak archetype
            rng.uniform(0.45, 0.68),   # mid archetype
            rng.uniform(0.68, 0.95),   # strong archetype
        ])
        accuracy = min(1.0, max(0.0, skill + rng.uniform(-0.15, 0.15)))
        attempts = rng.randint(0, 15)
        hint = 1 if rng.random() < 0.3 else 0
        is_weak = 1 if skill <= 0.45 else 0
        prev5 = min(1.0, max(0.0, accuracy + rng.uniform(-0.1, 0.1)))
        feat = [float(topic_idx), float(difficulty), round(skill, 4), round(accuracy, 4),
                float(attempts), float(hint), float(is_weak), round(prev5, 4)]
        p_ok = _true_probability(feat)
        y.append(1 if rng.random() < p_ok else 0)
        X.append(feat)
    return X, y


def main() -> int:
    rng = random.Random(SEED)
    X, y = build_dataset(1600)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    Xtr = [X[i] for i in tr]
    ytr = [y[i] for i in tr]
    Xte = [X[i] for i in te]
    yte = [y[i] for i in te]

    pred = PerformancePredictor()
    pred.fit(Xtr, ytr)
    te_acc = sum(1 for i in range(len(Xte))
                 if int(pred.model.predict([Xte[i]])[0]) == yte[i]) / len(Xte)
    print(f"Trained GradientBoosting performance predictor  train_acc={pred.train_accuracy:.3f}")
    print(f"  held-out test accuracy: {te_acc:.3f}   (n_test={len(Xte)}, features={FEATURES})")
    pred.save()
    print("Wrote", C.MODEL_FILE)

    # reproducibility check: reload and score a couple of rows
    back = PerformancePredictor.load()
    print("Reloaded uses_ml =", back.uses_ml(),
          "| sample p(correct) =", round(back.predict_accuracy(Xte[0])[0], 3),
          "| sampled outcome =", yte[0])
    return 0


if __name__ == "__main__":
    main()
