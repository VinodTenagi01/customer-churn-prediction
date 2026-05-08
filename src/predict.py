import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append("src")
from preprocess import engineer_features, encode_data

def load_model(path="models/xgboost_churn.pkl"):
    try:
        model = joblib.load(path)
        print("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("❌ Model not found! Run src/train.py first.")
        sys.exit(1)

def predict_single(model, customer_data: dict):
    df = pd.DataFrame([customer_data])
    df = engineer_features(df)
    df = encode_data(df)
    if "Churn" in df.columns:
        df.drop(columns=["Churn"], inplace=True)
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]
    return {
        "churn_prediction": "YES — Will Churn" if pred == 1 else "NO — Will Stay",
        "churn_probability": f"{prob*100:.1f}%",
        "risk_level": "🔴 HIGH" if prob > 0.6 else "🟡 MEDIUM" if prob > 0.3 else "🟢 LOW"
    }

def predict_batch(model, csv_path: str):
    df = pd.read_csv(csv_path)
    original = df.copy()
    df = engineer_features(df)
    df = encode_data(df)
    if "Churn" in df.columns:
        df.drop(columns=["Churn"], inplace=True)
    original["churn_probability"] = model.predict_proba(df)[:, 1]
    original["churn_prediction"] = model.predict(df)
    original["risk_level"] = original["churn_probability"].apply(
        lambda x: "HIGH" if x > 0.6 else "MEDIUM" if x > 0.3 else "LOW"
    )
    output_path = "data/predictions.csv"
    original.to_csv(output_path, index=False)
    print(f"✅ Batch predictions saved to {output_path}")
    print(f"\n📊 Risk Summary:")
    print(original["risk_level"].value_counts())
    return original

if __name__ == "__main__":
    model = load_model()

    print("\n🔍 Single Customer Prediction:")
    sample_customer = {
        "tenure": 3,
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "NumSupportTickets": 4
    }
    result = predict_single(model, sample_customer)
    print(f"Prediction  : {result['churn_prediction']}")
    print(f"Probability : {result['churn_probability']}")
    print(f"Risk Level  : {result['risk_level']}")

    print("\n📦 Batch Prediction:")
    predict_batch(model, "data/sample_data.csv")
