import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path="data/sample_data.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID
    df.drop(columns=["customerID"], inplace=True)

    return df

def engineer_features(df):
    # Tenure groups
    df["tenure_group"] = pd.cut(df["tenure"],
                                bins=[0, 12, 24, 48, 72],
                                labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])

    # Charge ratio
    df["charge_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # High risk flag
    df["high_risk"] = ((df["NumSupportTickets"] >= 3) &
                       (df["Contract"] == "Month-to-month")).astype(int)

    return df

def encode_data(df):
    le = LabelEncoder()
    cat_cols = ["Contract", "PaymentMethod", "TechSupport", "tenure_group"]
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print("✅ Data preprocessed successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
