import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def uncertainty_sampling(model, X_pool):
    probs = model.predict_proba(X_pool)
    uncertainty = np.abs(probs[:, 1] - 0.5)
    return np.argmin(uncertainty)


def active_learning(X, y, initial_size=50, query_size=50, iterations=5):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    labeled_idx = indices[:initial_size]
    unlabeled_idx = indices[initial_size:]
    model = LogisticRegression(max_iter=1000)
    accuracies = []

    for i in range(iterations):
        model.fit(X[labeled_idx], y[labeled_idx])
        preds = model.predict(X)
        accuracies.append(accuracy_score(y, preds))

        queries = []
        for _ in range(query_size):
            idx = uncertainty_sampling(model, X[unlabeled_idx])
            queries.append(unlabeled_idx[idx])
            unlabeled_idx = np.delete(unlabeled_idx, idx)

        labeled_idx = np.concatenate([labeled_idx, queries])

    return accuracies


def get_accuracies():    # ← top-level, importable by Plots.py
    df = pd.read_csv("insurance.csv")
    df["target"] = (df["charges"] > 16000).astype(int)
    X = pd.get_dummies(df.drop(columns=["charges", "target"]), drop_first=True).values
    y = df["target"].values
    return active_learning(X, y)


if __name__ == "__main__":
    for i, acc in enumerate(get_accuracies(), 1):
        print(f"Iteration {i}: {acc:.4f}")