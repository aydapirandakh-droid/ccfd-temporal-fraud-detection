# Credit Card Fraud Detection Using Temporal Deep Learning

## Project Overview

This project explores credit card fraud detection using temporal deep learning models. Instead of treating each transaction as an independent record, transactions are organized into chronological sequences to investigate whether recent transaction patterns can help identify fraudulent activity.

The project compares LSTM, Transformer, Hybrid LSTM with Temporal Attention, and an Ensemble model.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle.

- Total transactions: 284,807
- Fraudulent transactions: 492
- Normal transactions: 284,315
- Fraud rate: 0.173%
- Features: V1–V28 and Amount
- Target: Class

The `Time` column is used to preserve chronological order, while the model uses 29 transaction features.

## Temporal Analysis

An initial temporal analysis was performed to examine the time distribution of fraudulent transactions.

Key observations:

- Average time between consecutive transactions: 0.61 seconds
- Median time difference: 0 seconds
- 37.47% of fraud events occurred within 60 seconds of a previous fraud event.
- 67.82% occurred within 5 minutes of a previous fraud event.

These observations motivated the use of sequential models.

## Methodology

The main workflow includes:

1. Dataset loading and exploratory analysis
2. Chronological sorting of transactions
3. Temporal proximity analysis
4. Feature scaling using training data
5. Chronological train, validation, and test split (70% / 15% / 15%)
6. Sliding-window sequence construction
7. Model training and evaluation

The main experiments use a window size of 10 transactions, with each sequence containing 29 features.

Class imbalance is addressed using weighted binary cross-entropy loss (`BCEWithLogitsLoss`).

## Models Implemented

- **LSTM Classifier:** A recurrent neural network for sequential transaction classification.
- **Transformer Classifier:** A Transformer encoder with positional information for modeling transaction sequences.
- **Hybrid LSTM + Temporal Attention:** Combines LSTM sequence representation with temporal attention.
- **Ensemble Model:** Combines predictions from the LSTM and Hybrid models.

## Results

The following results are reported for the main test-set experiments with a window size of 10.

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| LSTM | 0.7585 | 0.9688 |
| Hybrid LSTM + Temporal Attention | 0.7600 | 0.9535 |
| Ensemble (LSTM + Hybrid) | **0.7676** | **0.9693** |

The Ensemble model achieved the highest PR-AUC and ROC-AUC among the models with reported test results.

*Note: Final test metrics for the standalone Transformer model were not available in the saved notebook outputs.*

## Window Size Experiment

Different sequence lengths were also compared:

| Window Size | PR-AUC | ROC-AUC |
|---:|---:|---:|
| 3 | 0.7457 | 0.9412 |
| 5 | **0.7754** | 0.9563 |
| 10 | 0.7440 | **0.9720** |
| 20 | 0.7408 | 0.9223 |

In this experiment, a window size of 5 achieved the highest PR-AUC, while a window size of 10 achieved the highest ROC-AUC.

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- KaggleHub

## Repository Structure

```text
ccfd-temporal-fraud-detection/
│
├── CCFD.Temporal.ipynb
└── README.md
```

## How to Run

1. Open the notebook in Google Colab or a compatible Jupyter environment.
2. Install the required libraries if they are not already available.
3. Make sure Kaggle dataset access is configured for KaggleHub.
4. Run the notebook cells in order.

## Notes

This project is an experimental implementation focused on temporal modeling for highly imbalanced credit card fraud detection.

Attention weights are examined as an auxiliary way to analyze the model's focus across sequence positions; they should not be interpreted as a complete explanation of individual predictions.
