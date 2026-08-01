# Credit Risk Scoring Model using ANN

## Problem Statement
Develop a deep learning model to assess credit risk (good/bad) using demographic and financial data.

## Dataset
German Credit Data (UCI Machine Learning Repository) — 1000 records, 20 features
(checking account status, credit history, purpose, credit amount, duration, employment, etc.)

## Approach
1. Data preprocessing: categorical encoding (one-hot), feature scaling (StandardScaler)
2. Handled class imbalance (70% good / 30% bad) using class weights
3. Built an Artificial Neural Network (ANN) using TensorFlow/Keras:
   - Input layer (48 features)
   - Hidden layer 1: 32 neurons, ReLU, Dropout(0.3)
   - Hidden layer 2: 16 neurons, ReLU, Dropout(0.2)
   - Output layer: 1 neuron, Sigmoid (binary classification)
4. Trained with EarlyStopping to prevent overfitting

## Results
- Accuracy: 69%
- ROC-AUC: 0.74
- Bad-risk recall: 68% (improved from 50% baseline using class weighting)

## Files
- `credit_risk_model.ipynb` — full notebook
- `training_history.png` — accuracy/loss curves
- `confusion_matrix.png` — confusion matrix heatmap

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

## How to Run
1. Open notebook in Google Colab
2. Run all cells sequentially
3. Dataset auto-loads from UCI repository
