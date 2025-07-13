# 📰 Task 1: News Topic Classifier Using BERT

This project is part of my **AI/ML Engineering Internship** at **DevelopersHub Corporation**. It focuses on building a BERT-based text classification model to predict the category of a news headline — **World, Sports, Business, or Sci/Tech**. The task highlights NLP data handling, transformer fine-tuning, and deployment using Gradio.

---

## 🎯 Objective

To develop a robust text classifier that automatically predicts the **topic category** of a given news headline using BERT.

---

## 📂 Dataset

- **Source:** [AG News Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Classes:**
  - 0: World
  - 1: Sports
  - 2: Business
  - 3: Sci/Tech
- **Records:**
  - Train: 120,000 samples (reduced to 2,000 for quicker training)
  - Test: 7,600 samples (reduced to 500 for testing)

---

## 🧹 Preprocessing Steps

- Downloaded CSV files for `train` and `test`
- Renamed columns to: `label`, `title`, `description`
- Shifted label values from 1–4 to 0–3 for 0-based indexing
- Selected only the **title** field for classification
- Sampled a subset for fast training (2,000 train, 500 test)

---

## 🤖 Model Development

- **Tokenizer:** `bert-base-uncased`
- **Model:** `BertForSequenceClassification` with `num_labels=4`
- **Training Parameters:**
  - Epochs: 1
  - Batch size: 8
  - Max length: 128 tokens
  - `fp16` training enabled
- **Frameworks Used:** PyTorch, Transformers (HuggingFace)

---

## 🧪 Evaluation Metrics

| Metric   | Value   |
|----------|---------|
| Accuracy | ~89%    |
| F1 Score | ~88%    |

---

## 🌐 Web App with Gradio

A simple Gradio interface was created to let users input any custom news headline and get the predicted topic:

### Example:

```text
Input: "NASA plans mission to explore Europa"
Output: Sci/Tech

Input: "Apple releases new iPhone with AI camera"
Output: Sci/Tech
```

## 📁 Folder Structure

Task-1-News-Topic-Classifier/
├── ag_news/
│   ├── train.csv
│   └── test.csv
├── Task_1_News_Topic_Classifier.ipynb
└── README.md

## 🚀 How to Run


1. Clone the repository:

```python

git clone https://github.com/Username/Task1-News-Topic-Classifier.git
cd Task1-News-Topic-Classifier
```

2. Install dependencies:

```python

pip install transformers torch pandas scikit-learn gradio

```

## ✍️ Author

Abdul Wahab
GitHub: @Abdul-Wahab1010
Internship Project – DevelopersHub Corporation | July 2025