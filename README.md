# Real-Time Algorithmic Fraud & Anomaly Detection in Streaming Data

## Project Overview

This project is developed for PS-02: Real-Time Algorithmic Fraud & Anomaly Detection in Streaming Data.

The system detects potentially fraudulent financial transactions using machine learning and simulates real-time transaction processing.

The project uses the PaySim financial transaction dataset and applies feature engineering and machine learning models for fraud detection.

## Problem Statement

Financial transaction systems generate a large number of transactions continuously. Detecting fraudulent transactions quickly is challenging because fraud cases are rare and transaction patterns can change over time.

This project aims to build a fraud detection pipeline that can process transaction data sequentially and generate a fraud prediction.

## Objectives

- Process financial transaction data.
- Perform feature engineering.
- Handle highly imbalanced fraud data.
- Train multiple machine learning models.
- Compare Logistic Regression, Random Forest and XGBoost.
- Perform transaction-level fraud prediction.
- Simulate streaming transaction processing.
- Display fraud detection results through a dashboard.

## Dataset

The project uses the PaySim synthetic financial transaction dataset.

The dataset contains information such as:

- Transaction type
- Transaction amount
- Sender balance
- Receiver balance
- Transaction time step
- Fraud label

The raw and processed datasets are not included in this repository.

## Machine Learning Models

The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

The trained models are stored in the `models/` directory.

## Feature Engineering

Additional features were created from the original transaction data:

- Balance change of sender
- Balance change of receiver
- Amount to original balance ratio
- One-hot encoded transaction type

## Streaming Detection

The `streaming_engine.py` module simulates a transaction stream by processing transactions sequentially from the processed dataset.

Each transaction is passed to the trained fraud detection model and produces:

- Fraud probability
- Fraud / Legitimate decision

## Project Structure

```text
Streaming_fraud_detection/
│
├── dashboard/
│   └── app.py
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── xgboost.pkl
│
├── notebook/
│   ├── 01_dataset_inspection.ipynb
│   └── 02_model_training.ipynb
│
├── result/
│   ├── logistic_regression_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   └── xgboost_confusion_matrix.png
│
├── src/
│   ├── fraud_detector.py
│   └── streaming_engine.py
│
├── tests/
│
├── .gitignore
└── README.md