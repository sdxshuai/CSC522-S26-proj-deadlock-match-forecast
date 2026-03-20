"""
upload_to_hf.py
================
Zip and upload the collected raw dataset to HuggingFace Hub.

Directories with many small files (matches/, player_stats/, match_list/) are
zipped first so the upload is a handful of requests instead of tens of thousands.

Usage:
    uv run python data/upload_to_hf.py

You will be prompted to enter your HF token on first run.
The token is cached in ~/.cache/huggingface/token for future runs.

Uploads to: https://huggingface.co/datasets/sdxshuai/deadlock-match-forecast
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import login, upload_folder

REPO_ID = "sdxshuai/deadlock-match-forecast"
RAW_DIR = Path("data/raw")
TOKEN_FILE = Path(".hf_token")  # project-root token file (git-ignored)

# Subdirectories to zip (many small files → single archive each)
ZIP_DIRS = ["matches", "match_list", "player_stats"]
# Subdirectories to upload as-is (few large files)
FLAT_DIRS = ["hero_stats"]


def main() -> None:
    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
    login(token=token)  # falls back to HF_TOKEN env var or interactive prompt if token is None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Zip the large dirs
        for name in ZIP_DIRS:
            src = RAW_DIR / name
            if not src.exists():
                print(f"  Skipping {name}/ (not found)")
                continue
            archive = tmp_path / name
            print(f"  Zipping {src} → {archive}.zip …")
            shutil.make_archive(str(archive), "zip", root_dir=str(src.parent), base_dir=name)
            size_mb = (tmp_path / f"{name}.zip").stat().st_size / 1024 / 1024
            print(f"    {name}.zip: {size_mb:.1f} MB")

        # Copy flat dirs as-is
        for name in FLAT_DIRS:
            src = RAW_DIR / name
            if not src.exists():
                continue
            dst = tmp_path / name
            print(f"  Copying {src} → {dst} …")
            shutil.copytree(src, dst)

        print(f"\nUploading to {REPO_ID} …")
        upload_folder(
            folder_path=tmp,
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="raw",
            commit_message="Upload raw dataset",
        )

    print("Upload complete.")


if __name__ == "__main__":
    main()
