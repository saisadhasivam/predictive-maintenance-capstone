import os
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])
api = HfApi()

HF_USERNAME   = "SaiSadhasivam"
SPACE_NAME    = "predictive-maintenance-app"
SPACE_REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

print("Step 1: Creating Hugging Face Space...")
api.create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False
)
print(f"Space ready: {SPACE_REPO_ID}")

print("Step 2: Uploading deployment files...")
for fname in ["app.py", "requirements.txt", "Dockerfile"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_id=SPACE_REPO_ID,
        repo_type="space"
    )
    print(f"  Uploaded: {fname}")

print("Deployment complete.")
print(f"Live app: https://huggingface.co/spaces/{SPACE_REPO_ID}")
