# 🫀 Heart Disease Prediction - Full Machine Learning Pipeline

This project implements a complete machine learning pipeline to analyze and predict the presence of heart disease using the UCI Heart Disease Dataset.

---

## 🎯 Objectives

- Preprocess and clean the dataset (handle missing values, encoding, scaling).
- Apply dimensionality reduction using PCA.
- Perform feature selection (Feature Importance, RFE, Chi-Square).
- Train and evaluate classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Evaluate performance using Accuracy, Precision, Recall, F1-score, ROC-AUC.
- Apply unsupervised learning (KMeans, Hierarchical Clustering).
- Optimize models using GridSearchCV and RandomizedSearchCV.
- Deploy a Streamlit web app for real-time prediction.
- (Bonus) Publish the app using Ngrok.

---

## 📂 Project Structure
│── data/
│ ├── heart_disease.csv
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│── models/
│ ├── final_model.pkl
│── ui/
│ ├── app.py (Streamlit UI)
│── results/
│ ├── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore


---

## 📊 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Target**: Binary classification
  - `0`: No heart disease
  - `1`: Presence of heart disease

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Streamlit
- Joblib
- ucimlrepo
- Git & GitHub

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/heart-disease-ml.git
   cd heart-disease-ml
2. Install the required packages:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py


Author
Amr Khaled
GitHub: @EngAmrkhaled

