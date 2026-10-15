from sklearn.inspection import permutation_importance
import yfinance as yf
import pennylane as qml
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Download stock data
stock = yf.Ticker("AAPL")
data = stock.history(period="2y")

# Step 2: Create stock features
data["Daily_Return"] = data["Close"].pct_change()
data["MA_5"] = data["Close"].rolling(window=5).mean()
data["Volatility_5"] = data["Daily_Return"].rolling(window=5).std()

# Step 3: Create next-day prediction target
data["Next_Close"] = data["Close"].shift(-1)
data["Target"] = (data["Next_Close"] > data["Close"]).astype(int)

# Step 4: Remove rows with missing values
data = data.dropna(
    subset=[
        "Daily_Return",
        "MA_5",
        "Volatility_5",
        "Next_Close"
    ]
)

# Step 5: Select features and target
feature_names = ["Daily_Return", "MA_5", "Volatility_5"]

X = data[feature_names]
y = data["Target"]

# Step 6: Split chronologically: 80% train, 20% test
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Step 7: Train the classical model
classical_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

classical_model.fit(X_train, y_train)
classical_predictions = classical_model.predict(X_test)

classical_accuracy = accuracy_score(
    y_test, classical_predictions
)

# Step 8: Scale features for the quantum circuit
quantum_scaler = MinMaxScaler()

X_train_scaled = quantum_scaler.fit_transform(X_train)
X_test_scaled = quantum_scaler.transform(X_test)

# Keep quantum input angles within 0 and pi
X_train_scaled = np.clip(X_train_scaled, 0, 1)
X_test_scaled = np.clip(X_test_scaled, 0, 1)

# Add a fourth input for the four-qubit circuit
X_train_quantum = np.column_stack([
    X_train_scaled,
    np.zeros(len(X_train_scaled))
])

X_test_quantum = np.column_stack([
    X_test_scaled,
    np.zeros(len(X_test_scaled))
])

# Step 9: Define the quantum circuit
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_circuit(inputs):
    qml.AngleEmbedding(
        inputs * np.pi,
        wires=range(4),
        rotation="Y"
    )

    for wire in range(3):
        qml.CNOT(wires=[wire, wire + 1])

    return tuple(
        qml.expval(qml.PauliZ(wires=i))
        for i in range(4)
    )

# Step 10: Transform stock features with the circuit
X_train_q = np.array([
    quantum_circuit(row) for row in X_train_quantum
], dtype=float)

X_test_q = np.array([
    quantum_circuit(row) for row in X_test_quantum
], dtype=float)

# Step 11: Train a model on quantum features
quantum_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

quantum_model.fit(X_train_q, y_train)
quantum_predictions = quantum_model.predict(X_test_q)

quantum_accuracy = accuracy_score(
    y_test, quantum_predictions
)

# Step 12: Compare both models
print("\nMODEL COMPARISON")
print("------------------------------")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"Classical accuracy: {classical_accuracy:.2%}")
print(f"Quantum-feature accuracy: {quantum_accuracy:.2%}")

print("\nNote: Higher accuracy on this test set")
print("does not guarantee better future predictions.")


# Step 13: Explain the classical model
importance = permutation_importance(
    classical_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

print("\nCLASSICAL MODEL: FEATURE IMPORTANCE")
print("-----------------------------------")

for name, value in zip(feature_names, importance.importances_mean):
    print(f"{name}: {value:.4f}")

