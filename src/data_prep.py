import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi, login

login(token=os.environ["HF_TOKEN"])
api = HfApi()

HF_USERNAME  = "SaiSadhasivam"
DATASET_REPO = "predictive-maintenance-capstone"
REPO_ID      = f"{HF_USERNAME}/{DATASET_REPO}"

print("Step 1: Loading dataset from Hugging Face...")
raw_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="data/engine_data.csv",
    repo_type="dataset"
)
df = pd.read_csv(raw_path)
print(f"Dataset loaded: {df.shape}")

print("Step 2: Cleaning data...")
df.columns = [
    "Engine_RPM", "Lub_Oil_Pressure", "Fuel_Pressure",
    "Coolant_Pressure", "Lub_Oil_Temp", "Coolant_Temp", "Engine_Condition"
]
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("Step 3: Handling outliers using IQR capping...")
features = [
    "Engine_RPM", "Lub_Oil_Pressure", "Fuel_Pressure",
    "Coolant_Pressure", "Lub_Oil_Temp", "Coolant_Temp"
]
for col in features:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

print("Step 4: Splitting into train and test sets...")
X = df.drop(columns=["Engine_Condition"])
y = df["Engine_Condition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv",   index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv",   index=False)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

print("Step 5: Uploading split files to Hugging Face...")
for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=f"data/{fname}",
        repo_id=REPO_ID,
        repo_type="dataset"
    )
    print(f"  Uploaded: {fname}")

print("Data preparation complete.")
