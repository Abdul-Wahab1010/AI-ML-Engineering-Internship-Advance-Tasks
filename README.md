# 🧠 AI/ML Engineering Internship Advance Tasks

This repository contains all tasks completed as part of my **AI/ML Engineering Internship** at **DevelopersHub Corporation**. Each task demonstrates a different machine learning use case, ranging from natural language processing and customer churn classification to multimodal house price regression.

---

## 📁 Tasks Overview

| Task No. | Task Name                             | Model Used                                  | Type           | Output                     |
|----------|----------------------------------------|----------------------------------------------|----------------|-----------------------------|
| Task 1   | News Topic Classifier Using BERT       | BERT (Transformers)                          | Classification | News category (0–3)         |
| Task 2   | Telco Customer Churn Prediction        | Random Forest Classifier                     | Classification | Churn: Yes / No             |
| Task 3   | Housing Price Prediction (Images + Tabular) | Multimodal CNN + Dense Neural Network  | Regression     | House Price (USD)           |

---

## 📰 Task 1: News Topic Classifier Using BERT

### 🎯 Objective
To develop a robust text classifier that automatically predicts the **topic category** of a given news headline using BERT.

### 📂 Dataset
- **Source:** [AG News Dataset – Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Classes:** World, Sports, Business, Sci/Tech
- **Sample Size:** 2000 train, 500 test

### 🛠️ Model
- `bert-base-uncased` tokenizer and model
- Fine-tuned with PyTorch & HuggingFace
- 1 epoch, batch size 8, max length 128, `fp16` training

### 📊 Evaluation
| Metric   | Value   |
|----------|---------|
| Accuracy | ~89%    |
| F1 Score | ~88%    |

### 🌐 Gradio Demo
- Input: Custom news headline
- Output: Predicted category (e.g., Sci/Tech)

---

## 📊 Task 2: Telco Customer Churn Prediction

### 🎯 Objective
Predict whether a customer is likely to **churn (leave)** using their demographics, billing, and service history.

### 📂 Dataset
- **Source:** [Telco Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7043 records
- **Features:** Demographics, services, contracts
- **Target:** Churn (Yes/No)

### 🛠️ Pipeline
- Used `ColumnTransformer` + `Pipeline`
- Handled numeric and categorical columns
- Model: `RandomForestClassifier` (best)
- Tuned using `GridSearchCV`

### 📊 Evaluation
| Metric            | Value    |
|-------------------|----------|
| Accuracy          | ~82%     |
| Precision, Recall | ✔️       |
| Confusion Matrix  | ✔️       |
| Feature Importance| ✔️ Top 20|

### 🔮 Prediction Sample
```python
Churn Prediction: YES
Churn Probability: 0.74
```

## 🏠 Task 3: Housing Price Prediction Using Images + Tabular Data

### 🎯 Objective
To predict house prices using both tabular features (e.g., bedrooms, sqft) and image data via a hybrid neural network.

### 📂 Dataset
Images: socal_pics/ (15,000 images)
Tabular: listings.csv
Features: bed, bath, sqft, n_citi
Target: House price

### 🛠️ Model
Tabular branch: Dense layers
Image branch: CNN (224x224 input)
Merged into a regression head
Trained on 2000 samples for 5 epochs

### 📊 Evaluation
Metric	Value
MAE	~60,000–80,000
Validation MAE	~64K–85K
Loss Trend	Stable

### 💡 Insights
Tabular features (especially sqft) strongly impacted price
Images helped capture architectural and condition cues
Multimodal model outperformed tabular-only baseline


### 🛠️ How to Run Any Task
```python
# Clone the repository
git clone https://github.com/Abdul-Wahab1010/AI-ML-Engineering-Internship-Tasks.git
cd AI-ML-Engineering-Internship-Tasks

# Enter any task folder
cd Task_01-News-Topic-Classifier
# or
cd Task_02-Telco-Churn-Prediction
# or
cd Task_03-Housing-Price-Prediction
```

Install Common Dependencies
```python
pip install pandas numpy matplotlib seaborn scikit-learn transformers torch tensorflow opencv-python gradio joblib
```
Open Notebook
```python
jupyter notebook <TASK_NOTEBOOK>.ipynb

```
### 👤 Author
Abdul Wahab
AI/ML Engineering Intern – DevelopersHub Corporation
GitHub: @Abdul-Wahab1010


