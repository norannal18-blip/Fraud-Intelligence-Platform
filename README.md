AI-Powered Fraud & Anomaly Detection Platform

Project Overview

An intelligent platform designed to detect fraudulent credit card transactions using machine learning.

Dataset

* Total transactions: 284,807
* Normal transactions: 284,315
* Fraudulent transactions: 492
* Fraud rate: 0.173%

Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* PostgreSQL
* Joblib

Machine Learning

The project uses a tuned Random Forest classifier to identify potentially fraudulent transactions.

Model Performance

* Precision: 84.04%
* Recall: 80.61%
* F1-Score: 82.28%
* Accuracy: 99.94%

Features

* Dataset overview and visualization
* Fraud vs. normal transaction analysis
* CSV file upload
* Fraud prediction
* Prediction results download
* Model performance evaluation

How to Run

pip install -r requirements.txt
streamlit run app.py

Project Structure

Fraud-Intelligence-Platform/
├── raw/
│   └── creditcard.csv
├── app.py
├── final_fraud_model.pkl
├── requirements.txt
├── README.md
└── .gitignore

Author

Data Science Student