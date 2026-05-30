# 📉 Subscription Customer Churn Stacking Classifier Suite

A production-ready data science workspace built with **Streamlit** and **Scikit-Learn** that implements an offline-trained **Stacking Classifier Ensemble** (combining RandomForest and XGBoost via a Logistic Regression meta-learner) to accurately predict subscription account retention risks.

---

## 🏗️ Project Architecture Layout

The workspace follows a clean, decoupled layout to keep production scripts organized:

```text
rf_classifier_insurance/
│
├── app.py                  # Streamlit Web User Interface Orchestrator
├── train.py                # Standalone One-Time Offline Model Training Script
├── requirements.txt        # Framework Dependencies List
├── README.md               # Project Walkthrough Documentation
│
├── data/
│   ├── train.csv           # Historical training records with labels
│   └── test.csv            # Independent evaluation records
│
├── models/
│   └── model.pkl           # Saved production Stacking Pipeline binary
│
└── src/
    ├── data_preprocessing.py   # Robust cleaning pipelines & transformers
    ├── model_training.py       # Stacking configuration structure (Layer 0 & Layer 1)
    ├── evaluation.py           # Out-of-sample confusion matrix & class reporting
    ├── visualization.py        # Isolated memory-safe Matplotlib rendering assets
    └── prediction.py           # Live transaction inference broker
