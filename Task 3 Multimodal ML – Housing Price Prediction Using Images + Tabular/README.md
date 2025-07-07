# 🏠 Task 3: Housing Price Prediction Using Images + Tabular Data

This project is part of my **AI/ML Engineering Internship** at **DevelopersHub Corporation**. The goal is to build a regression model that predicts house prices by combining structured tabular features and image data. This task focuses on multimodal machine learning using both CNNs and tabular data processing.

---

## 🎯 Objective

To predict the **price of a house** using both its **image** and **structured attributes** (like number of beds, baths, city ID, etc.) through a hybrid deep learning model.

---

## 📂 Dataset

- **Source:** Custom dataset with:
  - **Images folder:** `socal_pics` (15,000 JPEG files)
  - **Tabular data:** `listings.csv`
- **Records Used for Training:** 2000 samples
- **Tabular Features:**
  - `n_citi`: Numerical city ID
  - `bed`: Number of bedrooms
  - `bath`: Number of bathrooms
  - `sqft`: Area in square feet
- **Target Variable:** `price` (house price)

---

## 🔧 Data Preprocessing

- Removed string-type columns (`street`, `citi`) from tabular data
- Selected only numeric features
- Scaled features using `StandardScaler`
- Matched image files with tabular entries via `image_id`
- Resized all images to 224x224 and normalized pixel values
- Merged image and tabular arrays for model input
- Split data into **training (80%)** and **validation (20%)**

---

## 📊 Exploratory Data Analysis (EDA)

- Summary statistics for numeric features
- Distribution plots of `price`, `bed`, `bath`, and `sqft`
- Image count verification and shape consistency
- Handled missing images with warnings
- Ensured feature alignment for each sample

---

## 🤖 Model Building

- **Multimodal Neural Network** combining:
  - A CNN branch for image feature extraction
  - A dense branch for tabular data
  - Merged layers followed by regression output
- Built using `TensorFlow` and `Keras`
- Training setup:
  - Samples used: **2000**
  - Epochs: **5**
  - Batch size: **16**
- Final trained model saved as:
  - `house_price_model.h5`

---

## ✅ Evaluation Metrics

| Metric     | Result (approx.)   |
|------------|--------------------|
| MAE        | ~60,000–80,000     |
| Loss       | Reduced over epochs |
| Validation MAE | Stable (~64K–85K) |

---

## 💡 Key Insights

- **Tabular features** (especially `sqft` and `bed`) provided strong numeric signals.
- **Image data** helped model visual differences like construction type or house condition.
- Model trained well on 2000 samples without runtime crashes.
- Multimodal learning significantly improved performance compared to tabular-only models.

---

## 📁 Files Included

| File                               | Description                                       |
|------------------------------------|---------------------------------------------------|
| `listings.csv`                     | Structured tabular data of houses                 |
| `socal_pics/`                      | Folder with house images (15,000+ JPEGs)          |
| `multimodal-house-price-model.ipynb` | Full training + prediction notebook              |
| `house_price_model.h5`             | Trained Keras model (hybrid CNN + Dense)          |
| `README.md`                        | Project documentation                             |

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/Abdul-Wahab1010/Task3-Housing-Price-Prediction.git
cd Task3-Housing-Price-Prediction
```

2. Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow opencv-python
```

3. Run the notebook:

```bash
jupyter notebook multimodal-house-price-model.ipynb
```

## 📬 Author
Abdul Wahab
GitHub: @Abdul-Wahab1010
Internship Project — DevelopersHub Corporation | July 2025