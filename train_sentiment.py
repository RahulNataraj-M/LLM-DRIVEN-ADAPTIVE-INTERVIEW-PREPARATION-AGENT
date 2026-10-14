import sys

from sentiment import SentimentClassifier, INTENT_TO_STANCE
from sentiment_data import DATA


def cross_validation_report() -> None:
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import KFold
    except ImportError:
        print("[warn] scikit-learn unavailable; skipping cross-validation report.")
        return

    texts = [e["text"] for e in DATA]
    sent = [e["sentiment"] for e in DATA]
    stance = [e.get("stance") or INTENT_TO_STANCE.get(e.get("intent"), "smalltalk")
              for e in DATA]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), max_features=20000)
    X = vec.fit_transform(texts)

    for name, target in (("sentiment", sent), ("stance", stance)):
        model = RandomForestClassifier(n_estimators=250, class_weight="balanced",
                                       random_state=7, n_jobs=-1)
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        pred = np.empty(len(DATA), dtype=object)
        for tr, te in kf.split(X):
            model.fit(X[tr], [target[i] for i in tr])
            pred[te] = model.predict(X[te])
        acc = float((pred == np.array(target)).mean())
        print(f"[{name}] holdout accuracy: {acc:.2f}  (n={len(DATA)})")


def main() -> int:
    print(f"Training {len(DATA)} labeled examples...")
    clf = SentimentClassifier()
    clf.fit(DATA)
    clf.save()
    cross_validation_report()
    print("\nWrote sentiment_model.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
