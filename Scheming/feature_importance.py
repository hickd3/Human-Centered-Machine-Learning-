import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def compute_fnr_by_feature(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    baseline_pred = model.predict(X_test)

    def fnr(y_true, y_pred):
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        return fn / (fn + tp) if (fn + tp) > 0 else None

    baseline_fnr = fnr(y_test, baseline_pred)
    results = {}

    for col in X_test.columns:
        X_test_permuted = X_test.copy()
        X_test_permuted[col] = X_test[col].sample(frac=1, random_state=42).values
        perm_pred = model.predict(X_test_permuted)
        perm_fnr = fnr(y_test, perm_pred)
        if perm_fnr is not None and baseline_fnr is not None:
            results[col] = perm_fnr - baseline_fnr

    return sorted(results.items(), key=lambda x: x[1], reverse=True)


def get_fnr_results():   # ← top-level, importable by Plots.py
    df = pd.read_csv("insurance.csv")
    df["target"] = (df["charges"] > 16000).astype(int)
    X = pd.get_dummies(df.drop(columns=["charges", "target"]), drop_first=True)
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return compute_fnr_by_feature(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    for feature, value in get_fnr_results():
        print(f"  {feature:<25s} : {value:+.4f}")