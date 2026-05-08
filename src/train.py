import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
from preprocess import load_data, clean_data, engineer_features, encode_data, split_data

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42,
                                  use_label_encoder=False, eval_metric="logloss")
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"✅ {name} trained!")
    return trained

def evaluate_models(trained_models, X_test, y_test):
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred), 2),
            "Recall": round(recall_score(y_test, y_pred), 2),
            "F1 Score": round(f1_score(y_test, y_pred), 2),
            "ROC AUC": round(roc_auc_score(y_test, y_pred), 2),
        })
    results_df = pd.DataFrame(results)
    print("\n📊 Model Comparison:")
    print(results_df.to_string(index=False))
    return results_df

def save_best_model(trained_models, X_test, y_test):
    best_model = trained_models["XGBoost"]
    joblib.dump(best_model, "models/xgboost_churn.pkl")
    print("\n✅ Best model (XGBoost) saved to models/xgboost_churn.pkl")

def plot_confusion_matrix(trained_models, X_test, y_test):
    model = trained_models["XGBoost"]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title("XGBoost Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    print("✅ Confusion matrix saved!")

def plot_feature_importance(trained_models, X_train):
    model = trained_models["XGBoost"]
    importance = pd.Series(model.feature_importances_,
                           index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance.values[:10], y=importance.index[:10], palette="Blues_r")
    plt.title("Top 10 Feature Importances — XGBoost")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    print("✅ Feature importance plot saved!")

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    trained_models = train_models(X_train, y_train)
    evaluate_models(trained_models, X_test, y_test)
    save_best_model(trained_models, X_test, y_test)
    plot_confusion_matrix(trained_models, X_test, y_test)
    plot_feature_importance(trained_models, X_train)
