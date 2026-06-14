import os
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])
api = HfApi()

HF_USERNAME   = "SaiSadhasivam"
MODEL_REPO    = "predictive-maintenance-best-model"
MODEL_REPO_ID = f"{HF_USERNAME}/{MODEL_REPO}"

print("Step 1: Creating model repository on Hugging Face...")
api.create_repo(
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    exist_ok=True,
    private=False
)
print(f"Model repo ready: {MODEL_REPO_ID}")

print("Step 2: Uploading best_model.pkl to Hugging Face Model Hub...")
api.upload_file(
    path_or_fileobj="best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id=MODEL_REPO_ID,
    repo_type="model"
)
print(f"Model registered at: https://huggingface.co/{MODEL_REPO_ID}")
print("Model registration complete.")
