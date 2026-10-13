
# Telecom Customer Churn Prediction & Retention Analytics

## Overview
A machine learning project that predicts telecom customer churn
and supports customer retention decisions through an interactive
Streamlit application.

## Business Problem
Customer churn can affect recurring revenue. Identifying customers
with elevated churn risk can help a business prioritize retention
outreach.

## Dataset
IBM Telco Customer Churn dataset from Kaggle.
The dataset contains 7,043 customer records and 21 original columns.

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Machine Learning
- Logistic Regression
- Random Forest
- One-hot encoding and numerical feature scaling
- Stratified cross-validation using ROC-AUC
- Classification report and confusion matrix

## Results
Selected model: Logistic Regression

- Cross-validation mean ROC-AUC: 0.8459
- Test ROC-AUC: 0.8413
- Test accuracy: 73.81%
- Churn precision: 50.43%
- Churn recall: 78.34%
- Churn F1-score: 61.36%

These results are from the current model and test split.
They may vary if the data or training process changes.

## Run Locally
1. Install the dependencies:
   pip install -r requirements.txt

2. Start the application:
   python -m streamlit run app.py

## Disclaimer
Predictions are estimates, not guarantees. Risk categories and
retention recommendations are illustrative and should be evaluated
against business costs and outcomes.
