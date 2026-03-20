"""
upload_to_hf.py
================
Upload the collected raw dataset to HuggingFace Hub.

Usage:
    uv run python data/upload_to_hf.py

You will be prompted to enter your HF token on first run.
The token is cached in ~/.cache/huggingface/token for future runs.

Uploads to: https://huggingface.co/datasets/sdxshuai/deadlock-match-forecast
"""

from huggingface_hub import login, upload_folder

login()  # reads HF_TOKEN env var or prompts interactively; cached after first run

upload_folder(
    folder_path="data/raw",
    repo_id="sdxshuai/deadlock-match-forecast",
    repo_type="dataset",
    path_in_repo="raw",         # uploaded under raw/ in the HF repo
    ignore_patterns=[
        "*.pyc",
        "__pycache__/**",
        "logs/**",
        "*.log",
    ],
    commit_message="Upload raw dataset",
)

print("Upload complete.")
