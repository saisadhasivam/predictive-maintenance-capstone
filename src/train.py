import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from huggingface_hub import hf_hub_download, login

login(token=os.environ["HF_TOKEN"])

HF_USERNAME  = "SaiSadhasivam"
DATASET_REPO = "predictive-maintenance-capstone"
REPO_ID      = f"{HF_USERNAME}/{DATASET_REPO}"

print("Step 1: Loading train and test data from Hugging Face...")
X_train = pd.read_csv(hf_hub_download(
    repo_id=REPO_ID, filename="data/X_train.csv", repo_type="dataset"))
X_test  = pd.read_csv(hf_hub_download(
    repo_id=REPO_ID, filename="data/X_test.csv",  repo_type="dataset"))
y_train = pd.read_csv(hf_hub_download(
    repo_id=REPO_ID, filename="data/y_train.csv", repo_type="dataset")).squeeze()
y_test  = pd.read_csv(hf_hub_download(
    repo_id=REPO_ID, filename="data/y_test.csv",  repo_type="dataset")).squeeze()

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

print("Step 2: Training XGBoost model with best parameters...")
mlflow.set_experiment("Predictive_Maintenance_Pipeline")

with mlflow.start_run(run_name="XGBoost_Pipeline_Run"):
    params = {
        "n_estimators"     : 100,
        "max_depth"        : 4,
        "learning_rate"    : 0.05,
        "random_state"     : 42,
        "eval_metric"      : "logloss"
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    mlflow.log_params(params)
    mlflow.log_metric("accuracy",  round(accuracy,  4))
    mlflow.log_metric("precision", round(precision, 4))
    mlflow.log_metric("recall",    round(recall,    4))
    mlflow.log_metric("f1_score",  round(f1,        4))
    mlflow.sklearn.log_model(model, "xgboost_pipeline_model")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

joblib.dump(model, "best_model.pkl")
print("Model saved as best_model.pkl")
print("Model training complete.")
