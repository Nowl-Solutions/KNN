"""Choose k for KNN on Iris with cross-validation, then compare against scikit-learn.

    python iris_knn.py
Writes results/k_vs_accuracy.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from knn_model import KNN, StandardScaler

SEED = 42
K_VALUES = range(1, 31)
RESULTS = Path(__file__).resolve().parent / "results"


def cv_accuracy(X, y, k, scale, folds=5):
    """Mean accuracy over stratified folds. The scaler is fit on each training fold only."""
    scores = []
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        X_tr, X_va = X[tr], X[va]
        if scale:
            scaler = StandardScaler().fit(X_tr)
            X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
        scores.append(KNN(k).fit(X_tr, y[tr]).score(X_va, y[va]))
    return round(float(np.mean(scores)), 6)   # rounding stops float noise from breaking ties 


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, stratify=iris.target, random_state=SEED)

    # 1. Pick k on the training set only; the test set stays untouched until the end.
    raw = [cv_accuracy(X_train, y_train, k, scale=False) for k in K_VALUES]
    scaled = [cv_accuracy(X_train, y_train, k, scale=True) for k in K_VALUES]
    # Best (k, scaling) by CV. On ties prefer raw features (simpler), then the smallest k.
    best_acc, best_scale, best_k = max(
        ((acc, -s, -k) for s, curve in enumerate((raw, scaled)) for k, acc in zip(K_VALUES, curve)))
    best_scale, best_k = bool(-best_scale), -best_k
    print(f"5-fold CV on {len(X_train)} training flowers")
    for k in sorted({1, 3, 5, best_k, 15, 30}):
        print(f"  k={k:<2}  raw {raw[k - 1]:.3f}   scaled {scaled[k - 1]:.3f}")
    print(f"Best: k={best_k}, {'scaled' if best_scale else 'raw'} features (CV {best_acc:.3f})")

    # 2. Final check on the held-out test set.
    Xs_train, Xs_test = X_train, X_test
    if best_scale:
        scaler = StandardScaler().fit(X_train)
        Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
    ours = KNN(best_k).fit(Xs_train, y_train)
    sk = KNeighborsClassifier(n_neighbors=best_k).fit(Xs_train, y_train)
    ours_pred, sk_pred = ours.predict(Xs_test), sk.predict(Xs_test)
    print(f"\nTest set ({len(X_test)} flowers, k={best_k})")
    print(f"  our KNN      accuracy {np.mean(ours_pred == y_test):.3f}")
    print(f"  scikit-learn accuracy {np.mean(sk_pred == y_test):.3f}")
    print(f"  same prediction on {np.sum(ours_pred == sk_pred)}/{len(X_test)} flowers")

    # 3. Plot.
    RESULTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(K_VALUES, raw, marker="o", ms=4, label="raw features")
    ax.plot(K_VALUES, scaled, marker="o", ms=4, label="standardized features")
    ax.axvline(best_k, color="gray", ls="--", lw=1)
    ax.annotate(f"best k = {best_k}", (best_k, min(raw + scaled)), xytext=(5, 0),
                textcoords="offset points", color="gray")
    ax.set(xlabel="k (number of neighbours)", ylabel="5-fold CV accuracy",
           title="KNN on Iris: accuracy vs k")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "k_vs_accuracy.png", dpi=120)
    print(f"\nSaved {RESULTS / 'k_vs_accuracy.png'}")


if __name__ == "__main__":
    main()
