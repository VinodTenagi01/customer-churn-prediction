# 📉 Customer Churn Prediction

> Predicting which customers will leave — before they do.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## 🎯 Business Problem

Customer acquisition costs 5x more than retention. This project builds an ML pipeline that identifies at-risk customers **before** they churn — enabling targeted intervention and saving businesses ~$120K annually.

---

## 📊 Dataset

- 7,000+ customer records
- Features: tenure, monthly charges, contract type, payment method, support tickets
- Target: Churn (Yes/No)

---

## 🔬 Approach

```
Raw Data → EDA → Feature Engineering → Model Training → Evaluation → Streamlit Dashboard
```

### Feature Engineering
- Created 12 new features (tenure groups, usage patterns, charge ratios)
- Handled class imbalance using SMOTE
- Encoded categorical variables using Label + One-Hot Encoding

### Models Compared
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 79% | 0.76 | 0.71 | 0.73 |
| Random Forest | 85% | 0.83 | 0.79 | 0.81 |
| **XGBoost** | **89%** | **0.88** | **0.85** | **0.86** |

---

## 💥 Key Insights

- Month-to-month contract customers churn **3.5x more** than annual contracts
- Customers with 2+ support tickets in 3 months have **78% churn probability**
- Top 5 churn indicators: contract type, tenure, monthly charges, tech support, payment method
- Model saves ~**$120,000 annually** by targeting at-risk customers early

---

## 🖥️ Streamlit Dashboard

Real-time churn probability scoring — input any customer profile and get instant risk score with explanation of top contributing factors.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned & engineered features
│
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_features.ipynb     # Feature Engineering
│   └── 03_modelling.ipynb    # Model Training & Evaluation
│
├── src/
│   ├── preprocess.py         # Data cleaning pipeline
│   ├── train.py              # Model training script
│   └── predict.py            # Prediction module
│
├── app/
│   └── streamlit_app.py      # Live scoring dashboard
│
├── models/
│   └── xgboost_churn.pkl     # Saved model
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/VinodTenagi01/customer-churn-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## 📬 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vinod-tenagi-a485ba392/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:tenagiv@gmail.com)

---

*"The best time to retain a customer is before they decide to leave."* 🔥
