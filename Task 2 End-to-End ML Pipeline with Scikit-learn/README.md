# 📊 Task 2: Telco Customer Churn Prediction

This project is part of my **AI/ML Engineering Internship** at **DevelopersHub Corporation**. It focuses on building a production-ready machine learning pipeline to predict **customer churn** using telecom usage data. The task emphasizes data preprocessing, pipeline development, hyperparameter tuning, and model export.

---

## 🎯 Objective

To predict whether a customer is likely to **churn (leave the telecom service)** using customer demographics, service subscriptions, and billing information.

---

## 📂 Dataset

- **Source:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 entries
- **Features:**
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Services: `InternetService`, `StreamingTV`, `TechSupport`, etc.
  - Contract & Billing: `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target:** `Churn` — 1 (yes), 0 (no)

---

## 🔧 Data Preprocessing

- Dropped `customerID` column
- Converted `TotalCharges` to numeric and filled missing values
- Separated numerical and categorical features
- Used `ColumnTransformer` to apply:
  - `StandardScaler` for numeric features
  - `OneHotEncoder` for categorical features

---

## ⚙️ Machine Learning Pipeline

- Created a full **Scikit-learn `Pipeline`**:
  - Preprocessing (`ColumnTransformer`)
  - Model (Random Forest Classifier)
- Used `GridSearchCV` for hyperparameter tuning
  - Tuned `n_estimators`, `max_depth`

---

## 🤖 Model Building

- **Algorithms Used:**
  - `Logistic Regression` (baseline)
  - `Random Forest Classifier` ✅ (final selected model)
- Model trained using `train_test_split` and `cross-validation`
- Final model pipeline exported using `joblib` as:
  - `telco_churn_pipeline.pkl`

---

## 📊 Evaluation Metrics

| Metric            | Result         |
|-------------------|----------------|
| Accuracy          | ~82%           |
| Confusion Matrix  | ✔️ Visualized  |
| Classification Report | ✔️ Precision, Recall, F1 |
| Feature Importance| ✔️ Visualized (Top 20 Features) |

---

## 🔮 Prediction

 The model was tested on:
- Random samples from test set
- Custom new customer input (simulated realistic scenario)

```python
Churn Prediction: YES — The customer is likely to leave.
Churn Probability: 0.74
```

## Folder Structure

Task-2-End-to-End-ML-Pipeline/
├── Telco-Customer-Churn.csv
├── Task_2_Telco_Churn_Pipeline.ipynb
├── telco_churn_pipeline.pkl
└── README.md

## How to Run
1.Clone the repository:

```python
git clone https://github.com/Username/Task2-Telco-Churn-Prediction.git
cd Task2-Telco-Churn-Prediction
```
2.Install required packages:

```python
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```
3.Open the notebook:

```python
jupyter notebook Task_2_Telco_Churn_Pipeline.ipynb
```

## Author
Abdul Wahab
GitHub: @Abdul-Wahab1010
Internship Project - DevelopersHub Corporation | July 2025