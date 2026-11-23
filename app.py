import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==============================
# 1. Page Configuration
# ==============================

st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #F1F5F9;
}

h1, h2, h3 {
    color: #38BDF8 !important;
}

[data-testid="stMetric"] {
    background-color: #172033;
    border: 1px solid #263B58;
    padding: 20px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #38BDF8;
}

[data-testid="stDataFrame"] {
    border: 1px solid #263B58;
    border-radius: 12px;
}

.stButton button {
    background-color: #0284C7;
    color: white;
    border-radius: 10px;
    border: none;
}

</style>
""", unsafe_allow_html=True)
st.title("🛡️ AI-Powered Fraud Detection Platform")
st.write("Credit Card Fraud Detection Dashboard")

# ==============================
# 2. Load Dataset
# ==============================

DATA_PATH = Path("raw/creditcard.csv")
MODEL_PATH = Path("final_fraud_model.pkl")

if not DATA_PATH.exists():
    st.error("Dataset file not found: raw/creditcard.csv")
    st.stop()

df = pd.read_csv(DATA_PATH)

# ==============================
# 3. Dashboard Metrics
# ==============================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", f"{len(df):,}")

with col2:
    st.metric("Fraud Transactions", f"{int(df['Class'].sum()):,}")

with col3:
    fraud_percentage = df["Class"].mean() * 100
    st.metric("Fraud Rate", f"{fraud_percentage:.3f}%")

# ==============================
# 4. Dataset Preview
# ==============================

st.subheader("Dataset Preview")
st.dataframe(df.head(10), width="stretch")

# ==============================
# 5. Fraud vs Normal Chart
# ==============================

st.subheader("Fraud vs Normal Transactions")

class_counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
class_counts.index = ["Normal", "Fraud"]

st.bar_chart(class_counts)

# ==============================
# 6. Transaction Amount Analysis
# ==============================

st.subheader("Transaction Amount Analysis")

amount_by_class = df.groupby("Class")["Amount"].mean()
amount_by_class.index = ["Normal", "Fraud"]

st.bar_chart(amount_by_class)

# ==============================
# 7. Fraud Detection
# ==============================

st.divider()
st.header("🔎 Fraud Detection")

st.write("Upload a CSV file to detect potentially fraudulent transactions.")

# ==============================
# 8. Load Final Model
# ==============================

if not MODEL_PATH.exists():
    st.warning(
        "Model file not found. Place final_fraud_model.pkl "
        "in the project folder."
    )
    st.stop()

try:
    model_bundle = joblib.load(MODEL_PATH)

    if isinstance(model_bundle, dict):
        model = model_bundle["model"]
        threshold = model_bundle["threshold"]
    else:
        model = model_bundle
        threshold = 0.5

except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ==============================
# 9. Upload Transactions CSV
# ==============================

uploaded_file = st.file_uploader(
    "Upload Transactions CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        new_data = pd.read_csv(uploaded_file)

        required_columns = (
            ["Time"]
            + [f"V{i}" for i in range(1, 29)]
            + ["Amount"]
        )

        missing_columns = [
            col for col in required_columns
            if col not in new_data.columns
        ]

        if missing_columns:
            st.error(
                "CSV file is missing required columns: "
                + ", ".join(missing_columns)
            )

        else:
            input_data = new_data[required_columns]

            predictions = model.predict_proba(input_data)[:, 1]

            results = new_data.copy()

            results["Fraud Probability"] = predictions
            results["Prediction"] = [
                "Fraud" if p >= threshold else "Normal"
                for p in predictions
            ]

            # ==============================
            # 10. Prediction Results
            # ==============================

            st.subheader("Prediction Results")

            st.dataframe(
                results,
                width="stretch"
            )

            detected_fraud = int(
                (results["Prediction"] == "Fraud").sum()
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Detected Fraud Transactions",
                    detected_fraud
                )

            with col2:
                st.metric(
                    "Transactions Analyzed",
                    len(results)
                )

            # ==============================
            # 11. Download Results
            # ==============================

            csv_output = results.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.success("Prediction completed successfully!")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_output,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                key="download_fraud_results"
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
# =================================
# 12. Model Performance Evaluation
# =================================

st.divider()

st.header("📊 Model Performance Evaluation")

st.write("Evaluation results on the independent test dataset.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Precision", "84.04%")

with col2:
    st.metric("Recall", "80.61%")

with col3:
    st.metric("F1-Score", "82.28%")

with col4:
    st.metric("Accuracy", "99.94%")

st.subheader("Confusion Matrix")

confusion_data = pd.DataFrame(
    [[56849, 15], [19, 79]],
    index=["Actual Normal", "Actual Fraud"],
    columns=["Predicted Normal", "Predicted Fraud"]
)

st.dataframe(confusion_data, width="stretch")
# ==============================
# 12. Footer
# ==============================

st.divider()

st.caption(
    "AI-Powered Fraud & Anomaly Detection Platform"
)