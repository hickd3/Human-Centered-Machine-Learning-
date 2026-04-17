import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


# Load data
df = pd.read_csv("insurance.csv")

# Target: charges > 16000
df['target'] = (df['charges'] > 16000).astype(int)

X = df.drop(columns=['charges', 'target'])
X = pd.get_dummies(X, drop_first=True)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale (for LR + MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, early_stopping=True)
}

for name, model in models.items():
    if name == "random_forest":
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    print(f"{name.upper()}")
    print(classification_report(y_test, preds))

