
import streamlit as st
import yfinance as yf
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Hybrid Quantum Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("Hybrid Quantum Neural Network")
st.subheader("Stock Market Prediction Using Explainable Learning")

st.info(
    "This educational prototype compares Classical ML, a fixed "
    "Quantum-feature model, and a Trainable Hybrid QNN."
)

ticker = st.selectbox(
    "Select stock", ["AAPL", "MSFT", "GOOGL", "AMZN"]
)

period = st.selectbox(
    "Historical data period", ["1y", "2y", "5y"], index=1
)

# Fetch historical stock data
if st.button("Fetch Stock Data"):
    try:
        raw = yf.Ticker(ticker).history(period=period)

        if raw.empty:
            st.error("No stock data found. Please try again.")
        else:
            st.success("Stock data loaded successfully!")
            st.subheader(f"{ticker} Closing Price")
            st.line_chart(raw["Close"])

            raw["Daily_Return"] = raw["Close"].pct_change()
            raw["MA_5"] = raw["Close"].rolling(5).mean()
            raw["Volatility_5"] = raw["Daily_Return"].rolling(5).std()

            st.subheader("Latest Historical Data")
            st.dataframe(raw.tail(10))

            st.subheader("Stock Features")
            st.dataframe(
                raw[[
                    "Close", "Daily_Return", "MA_5", "Volatility_5"
                ]].tail(10)
            )
    except Exception as e:
        st.error(f"Could not fetch stock data: {e}")


st.divider()
st.header("Three-Model Comparison")

if st.button("Run Model Comparison"):
    try:
        with st.spinner(
            "Downloading data and training all three models..."
        ):
            data = yf.Ticker(ticker).history(period=period)

            if data.empty:
                st.error("No stock data found.")
                st.stop()

            # 1. Prepare features and target
            data["Daily_Return"] = data["Close"].pct_change()
            data["MA_5"] = data["Close"].rolling(5).mean()
            data["Volatility_5"] = data["Daily_Return"].rolling(5).std()
            data["Next_Close"] = data["Close"].shift(-1)
            data["Target"] = (
                data["Next_Close"] > data["Close"]
            ).astype(int)

            feature_names = [
                "Daily_Return", "MA_5", "Volatility_5"
            ]

            data = data.dropna(
                subset=feature_names + ["Next_Close"]
            )

            if len(data) < 50:
                st.error("Not enough data. Select a longer period.")
                st.stop()

            X = data[feature_names].to_numpy(dtype=float)
            y = data["Target"].to_numpy(dtype=int)

            # Same chronological test rows for all models
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if (
                len(X_test) == 0
                or len(np.unique(y_train)) < 2
                or len(np.unique(y_test)) < 2
            ):
                st.error(
                    "Training/testing data needs both target classes. "
                    "Try a different stock or longer period."
                )
                st.stop()

            # 2. Classical ML model
            classical_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000)
            )
            classical_model.fit(X_train, y_train)
            classical_predictions = classical_model.predict(X_test)
            classical_accuracy = accuracy_score(
                y_test, classical_predictions
            )

            # 3. Fixed quantum-feature model
            q_scaler = MinMaxScaler()
            X_train_qin = np.clip(
                q_scaler.fit_transform(X_train), 0, 1
            )
            X_test_qin = np.clip(
                q_scaler.transform(X_test), 0, 1
            )

            X_train_qin = np.column_stack([
                X_train_qin, np.zeros(len(X_train_qin))
            ])
            X_test_qin = np.column_stack([
                X_test_qin, np.zeros(len(X_test_qin))
            ])

            fixed_dev = qml.device("default.qubit", wires=4)

            @qml.qnode(fixed_dev)
            def fixed_circuit(inputs):
                qml.AngleEmbedding(
                    inputs * np.pi, wires=range(4), rotation="Y"
                )
                for wire in range(3):
                    qml.CNOT(wires=[wire, wire + 1])
                return tuple(
                    qml.expval(qml.PauliZ(i)) for i in range(4)
                )

            X_train_q = np.array(
                [fixed_circuit(row) for row in X_train_qin],
                dtype=float
            )
            X_test_q = np.array(
                [fixed_circuit(row) for row in X_test_qin],
                dtype=float
            )

            quantum_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000)
            )
            quantum_model.fit(X_train_q, y_train)
            quantum_predictions = quantum_model.predict(X_test_q)
            quantum_accuracy = accuracy_score(
                y_test, quantum_predictions
            )

            # 4. Trainable Hybrid Quantum Neural Network
            hybrid_scaler = StandardScaler()
            X_train_h = np.clip(
                hybrid_scaler.fit_transform(X_train), -1, 1
            )
            X_test_h = np.clip(
                hybrid_scaler.transform(X_test), -1, 1
            )

            # Pad the three features to four qubits
            X_train_h = np.column_stack([
                X_train_h, np.zeros(len(X_train_h))
            ])
            X_test_h = np.column_stack([
                X_test_h, np.zeros(len(X_test_h))
            ])

            n_qubits = 4
            n_layers = 2
            hybrid_dev = qml.device(
                "default.qubit", wires=n_qubits
            )

            @qml.qnode(
                hybrid_dev,
                interface="autograd",
                diff_method="backprop"
            )
            def hybrid_circuit(inputs, weights):
                qml.AngleEmbedding(
                    inputs, wires=range(n_qubits), rotation="Y"
                )
                qml.StronglyEntanglingLayers(
                    weights, wires=range(n_qubits)
                )
                return [
                    qml.expval(qml.PauliZ(i))
                    for i in range(n_qubits)
                ]

            # Fixed seed makes initialization reproducible
            rng = np.random.default_rng(42)
            weights = pnp.array(
                0.01 * rng.standard_normal(
                    (n_layers, n_qubits, 3)
                ),
                requires_grad=True
            )
            readout_weights = pnp.array(
                0.01 * rng.standard_normal(n_qubits),
                requires_grad=True
            )
            bias = pnp.array(0.0, requires_grad=True)

            def hybrid_probability(x, w, rw, b):
                outputs = hybrid_circuit(x, w)
                score = pnp.dot(pnp.array(outputs), rw) + b
                return 1.0 / (1.0 + pnp.exp(-score))

            def hybrid_loss(w, rw, b):
                losses = []
                for x, label in zip(X_train_h, y_train):
                    prob = hybrid_probability(x, w, rw, b)
                    prob = pnp.clip(prob, 1e-7, 1 - 1e-7)
                    losses.append(
                        -label * pnp.log(prob)
                        - (1 - label) * pnp.log(1 - prob)
                    )
                return pnp.mean(pnp.stack(losses))

            optimizer = qml.AdamOptimizer(stepsize=0.03)
            epochs = 10
            progress = st.progress(
                0, text="Training Hybrid QNN: Epoch 0/10"
            )
            loss_history = []

            for epoch in range(epochs):
                weights, readout_weights, bias = optimizer.step(
                    hybrid_loss, weights, readout_weights, bias
                )
                loss_value = float(
                    hybrid_loss(weights, readout_weights, bias)
                )
                loss_history.append(loss_value)
                progress.progress(
                    (epoch + 1) / epochs,
                    text=f"Training Hybrid QNN: Epoch {epoch + 1}/{epochs}"
                )

            progress.empty()

            hybrid_probabilities = np.array([
                float(hybrid_probability(
                    x, weights, readout_weights, bias
                ))
                for x in X_test_h
            ])
            hybrid_predictions = (
                hybrid_probabilities >= 0.5
            ).astype(int)
            hybrid_accuracy = accuracy_score(
                y_test, hybrid_predictions
            )

        # 5. Display results
        st.success("All three models trained and evaluated!")

        st.caption(
            f"Stock: {ticker} | Period: {period} | "
            f"Training rows: {len(X_train)} | "
            f"Testing rows: {len(X_test)}"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Classical ML", f"{classical_accuracy:.2%}")
        c2.metric("Quantum-feature", f"{quantum_accuracy:.2%}")
        c3.metric("Trainable Hybrid QNN", f"{hybrid_accuracy:.2%}")

        accuracy_data = pd.DataFrame({
            "Model": [
                "Classical ML",
                "Quantum-feature",
                "Trainable Hybrid QNN"
            ],
            "Accuracy (%)": [
                classical_accuracy * 100,
                quantum_accuracy * 100,
                hybrid_accuracy * 100
            ]
        })

        st.subheader("Model Accuracy Comparison")
        st.bar_chart(accuracy_data.set_index("Model"))

        # 6. Training loss
        st.subheader("Hybrid QNN Training Loss")
        st.line_chart(pd.DataFrame({
            "Training Loss": loss_history
        }))

        # 7. Explainability: classical feature importance
        st.subheader("Explainable Learning: Feature Importance")
        importance = permutation_importance(
            classical_model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="accuracy"
        )

        importance_data = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance.importances_mean
        }).sort_values("Importance", ascending=False)

        st.bar_chart(importance_data.set_index("Feature"))
        st.dataframe(importance_data, hide_index=True)

        st.caption(
            "Permutation importance measures how test accuracy changes "
            "when a feature is shuffled. It describes the classical model."
        )

        # 8. Actual vs predictions
        st.subheader("Actual vs Model Predictions")
        results = pd.DataFrame({
            "Date": data.index[split:].strftime("%Y-%m-%d"),
            "Actual Direction": y_test,
            "Classical Prediction": classical_predictions,
            "Quantum-feature Prediction": quantum_predictions,
            "Hybrid QNN Prediction": hybrid_predictions
        })
        st.dataframe(results.tail(20), hide_index=True)

        st.warning(
            "These are historical test results, not guaranteed future "
            "predictions. This prototype does not establish quantum "
            "advantage. Feature importance currently explains the "
            "classical model only."
        )

    except Exception as e:
        st.error(f"Model comparison failed: {e}")

st.divider()
st.caption(
    "Educational project only. Accuracy does not guarantee future "
    "stock-market performance."
)
