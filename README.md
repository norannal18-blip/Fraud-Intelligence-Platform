# 🛡️ AI-Powered Fraud & Anomaly Detection Platform

An intelligent financial fraud detection platform powered by Machine Learning. The system analyzes financial transactions and identifies potentially fraudulent activities using a tuned Random Forest classifier.

## 🎯 Project Overview

Credit card fraud detection is a challenging problem because fraudulent transactions represent a very small percentage of all transactions.

This project aims to detect suspicious transactions while balancing fraud detection performance and false alarms.

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset.

- Total transactions: 284,807
- Normal transactions: 284,315
- Fraudulent transactions: 492
- Fraud rate: 0.173%
- Missing values: None

## 🤖 Machine Learning Model

The final model is a tuned Random Forest Classifier.

Key configuration:

- Number of estimators: 200
- Maximum depth: 12
- Minimum samples per leaf: 2
- Class weight: Balanced
- Decision threshold: 0.4

## 📈 Final Model Evaluation

The model was evaluated on an independent held-out test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 99.94% |
| Precision | 83.67% |
| Recall | 83.67% |
| F1 Score | 83.67% |

### Confusion Matrix

| | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| Actual Normal | 56,848 | 16 |
| Actual Fraud | 16 | 82 |

The model successfully detected 82 out of 98 fraudulent transactions in the test set.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- PostgreSQL
- Google Colab
- Git and GitHub

## ✨ Platform Features

- Interactive fraud detection dashboard
- Dataset overview and transaction statistics
- Fraud and normal transaction visualizations
- CSV file upload
- Machine Learning predictions
- Fraud probability estimation
- Detected fraudulent transactions display
- Downloadable prediction results
- Model evaluation metrics when labeled data is provided

## 📂 Project Structure

```text
Fraud-Intelligence-Platform/
│
├── app.py
├── final_fraud_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
├── raw/
│   └── creditcard.csv
│
└── .venv/
```

## 🚀 Run Locally

1. Clone the repository:

```bash
git clone https://github.com/norannal18-blip/Fraud-Intelligence-Platform.git
```

2. Navigate to the project directory:

```bash
cd Fraud-Intelligence-Platform
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

## 👩‍💻 Author

Noran Alghamdi

Data Science Student

Umm Al-Qura University

## 📌 Project Purpose

This project was developed as a practical application of Machine Learning techniques for financial fraud detection and data analysis.
