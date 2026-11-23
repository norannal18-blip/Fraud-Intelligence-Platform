import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# 2. CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #60a5fa;
}

.metric-card {
    background: #172033;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #26344d;
    text-align: center;
}

div[data-testid="stMetric"] {
    background: #172033;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #26344d;
}

.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. HEADER
# =====================================================

st.title("🛡️ AI-Powered Fraud & Anomaly Detection Platform")

st.markdown("""
### Intelligent Financial Transaction Monitoring

An AI-powered platform designed to detect suspicious
financial transactions using Machine Learning.
""")

st.divider()

# =====================================================
# 4. MODEL LOADING
# =====================================================

MODEL_PATH = Path("final_fraud_model (1).pkl")
DATA_PATH = Path("raw/creditcard.csv")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle.get("model")
        threshold = float(bundle.get("threshold", 0.4))
    else:
        model = bundle
        threshold = 0.4

    return model, threshold


model, threshold = load_model()

if model is None:
    st.error("Model file not found. Please check final_fraud_model.pkl.")
    st.stop()

st.success("Machine Learning Model Loaded Successfully")

# =====================================================
# 5. DATASET OVERVIEW
# =====================================================

st.header("📊 Dataset Overview")

if DATA_PATH.exists():

    df = pd.read_csv(DATA_PATH)

    total_transactions = len(df)
    fraud_transactions = int(df["Class"].sum())
    normal_transactions = total_transactions - fraud_transactions

    fraud_rate = (
        fraud_transactions / total_transactions * 100
        if total_transactions else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Transactions", f"{fraud_transactions:,}")
    col3.metric("Normal Transactions", f"{normal_transactions:,}")
    col4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.subheader("Transaction Distribution")

    distribution = pd.DataFrame({
        "Transaction Type": ["Normal", "Fraud"],
        "Count": [normal_transactions, fraud_transactions]
    }).set_index("Transaction Type")

    st.bar_chart(distribution)

    if "Amount" in df.columns:
        st.subheader("Transaction Amount Analysis")

        amount_summary = df.groupby("Class")["Amount"].mean()

        amount_summary.index = [
            "Normal" if x == 0 else "Fraud"
            for x in amount_summary.index
        ]

        st.bar_chart(amount_summary)

    with st.expander("View Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)

else:
    st.info(
        "The original dataset is not bundled with this deployment. "
        "You can upload a CSV file below to analyze transactions."
    )

# =====================================================
# 6. TRANSACTION ANALYSIS
# =====================================================

st.divider()

st.header("🔍 Fraud Detection Analysis")

st.write(
    "Upload a CSV file containing transaction features "
    "to detect potentially fraudulent transactions."
)

uploaded_file = st.file_uploader(
    "Upload Transactions CSV",
    type=["csv"]
)

# =====================================================
# 7. REQUIRED FEATURES
# =====================================================

required_columns = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)

# =====================================================
# 8. PREDICTION
# =====================================================

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(uploaded_df.head(10), use_container_width=True)

        missing_columns = [
            col for col in required_columns
            if col not in uploaded_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            X = uploaded_df[required_columns].copy()

            if X.isnull().any().any():
                st.error(
                    "The uploaded data contains missing values. "
                    "Please clean the data and upload it again."
                )
                st.stop()

            if not all(
                pd.api.types.is_numeric_dtype(X[col])
                for col in required_columns
            ):
                st.error("All transaction features must be numeric.")
                st.stop()

            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)

                if expected_features != required_columns:
                    st.error(
                        "The model feature order does not match "
                        "the uploaded CSV columns."
                    )
                    st.stop()

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(X)[:, 1]

            else:
                st.error(
                    "The loaded model does not support probability predictions."
                )
                st.stop()

            predictions = (
                probabilities >= threshold
            ).astype(int)

            results = uploaded_df.copy()

            results["Fraud Probability"] = probabilities
            results["Prediction"] = predictions

            results["Prediction Label"] = results["Prediction"].map({
                0: "Normal",
                1: "Fraud"
            })

            st.success("Transaction Analysis Completed Successfully!")

            # -------------------------------------------------
            # 9. PREDICTION SUMMARY
            # -------------------------------------------------

            st.subheader("Prediction Summary")

            total = len(results)
            fraud_count = int(predictions.sum())
            normal_count = total - fraud_count

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Analyzed", f"{total:,}")
            col2.metric("Predicted Fraud", f"{fraud_count:,}")
            col3.metric("Predicted Normal", f"{normal_count:,}")

            # -------------------------------------------------
            # 10. PREDICTION CHART
            # -------------------------------------------------

            st.subheader("Fraud vs Normal Predictions")

            prediction_chart = pd.DataFrame({
                "Transaction Type": ["Normal", "Fraud"],
                "Count": [normal_count, fraud_count]
            }).set_index("Transaction Type")

            st.bar_chart(prediction_chart)

            # -------------------------------------------------
            # 11. FRAUD RESULTS
            # -------------------------------------------------

            st.subheader("Detected Fraudulent Transactions")

            fraud_results = results[
                results["Prediction"] == 1
            ]

            if len(fraud_results) > 0:
                st.dataframe(
                    fraud_results,
                    use_container_width=True
                )
            else:
                st.info("No fraudulent transactions were detected.")

            # -------------------------------------------------
            # 12. ALL RESULTS
            # -------------------------------------------------

            st.subheader("All Prediction Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            # -------------------------------------------------
            # 13. DOWNLOAD RESULTS
            # -------------------------------------------------

            csv_data = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_data,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

            # -------------------------------------------------
            # 14. EVALUATION
            # -------------------------------------------------

            if "Class" in uploaded_df.columns:

                st.divider()
                st.header("📈 Model Evaluation")

                y_true = uploaded_df["Class"].astype(int)

                accuracy = accuracy_score(y_true, predictions)
                precision = precision_score(
                    y_true, predictions, zero_division=0
                )
                recall = recall_score(
                    y_true, predictions, zero_division=0
                )
                f1 = f1_score(
                    y_true, predictions, zero_division=0
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                col2.metric("Precision", f"{precision * 100:.2f}%")
                col3.metric("Recall", f"{recall * 100:.2f}%")
                col4.metric("F1 Score", f"{f1 * 100:.2f}%")

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_true,
                    predictions,
                    labels=[0, 1]
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Normal", "Actual Fraud"],
                    columns=["Predicted Normal", "Predicted Fraud"]
                )

                st.dataframe(cm_df, use_container_width=True)

                st.caption(
                    "Evaluation metrics above describe this uploaded file. "
                    "For an unbiased performance estimate, use a separate "
                    "held-out test dataset that was not used for training."
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")

# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#94a3b8; padding:15px;">

**AI-Powered Fraud & Anomaly Detection Platform**

Machine Learning | Python | Scikit-learn | Streamlit

</div>
""", unsafe_allow_html=True)
