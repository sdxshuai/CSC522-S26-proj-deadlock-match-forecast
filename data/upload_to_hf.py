"""
upload_to_hf.py
================
Upload dataset artifacts to HuggingFace Hub.

Usage:
    # Upload everything
    uv run python data/upload_to_hf.py

    # Upload only specific parts
    uv run python data/upload_to_hf.py --raw
    uv run python data/upload_to_hf.py --processed
    uv run python data/upload_to_hf.py --readme
    uv run python data/upload_to_hf.py --raw --readme

Repo layout on HF:
    raw/          ← zipped raw data + hero_stats/ + README.md
    processed/    ← matches.parquet

Uploads to: https://huggingface.co/datasets/sdxshuai/deadlock-match-forecast
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import login, upload_file, upload_folder

REPO_ID = "sdxshuai/deadlock-match-forecast"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TOKEN_FILE = Path(".hf_token")

# Subdirectories to zip (many small files → single archive each)
ZIP_DIRS = ["matches", "match_list", "player_stats"]
# Subdirectories to upload as-is (few large files)
FLAT_DIRS = ["hero_stats"]


def upload_raw() -> None:
    print("=== Uploading raw data ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for name in ZIP_DIRS:
            src = RAW_DIR / name
            if not src.exists():
                print(f"  Skipping {name}/ (not found)")
                continue
            archive = tmp_path / name
            print(f"  Zipping {src} …")
            shutil.make_archive(str(archive), "zip", root_dir=str(src.parent), base_dir=name)
            size_mb = (tmp_path / f"{name}.zip").stat().st_size / 1024 / 1024
            print(f"    {name}.zip: {size_mb:.1f} MB")

        for name in FLAT_DIRS:
            src = RAW_DIR / name
            if not src.exists():
                continue
            dst = tmp_path / name
            print(f"  Copying {src} …")
            shutil.copytree(src, dst)

        print(f"  Uploading to {REPO_ID}/raw …")
        upload_folder(
            folder_path=tmp,
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="raw",
            commit_message="Upload raw dataset",
        )
    print("  Done.")


def upload_processed() -> None:
    print("=== Uploading processed data ===")
    parquet = PROCESSED_DIR / "matches.parquet"
    if not parquet.exists():
        print("  data/processed/matches.parquet not found — run src/preprocess.py first")
        return
    size_mb = parquet.stat().st_size / 1024 / 1024
    print(f"  matches.parquet: {size_mb:.1f} MB")
    upload_file(
        path_or_fileobj=str(parquet),
        path_in_repo="processed/matches.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload processed feature matrix",
    )
    print("  Done.")


def upload_readme() -> None:
    print("=== Uploading README ===")
    readme = Path("data/README.md")
    if not readme.exists():
        print("  data/README.md not found")
        return
    upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Update README",
    )
    print("  Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",       action="store_true", help="Upload raw data (zipped)")
    parser.add_argument("--processed", action="store_true", help="Upload processed/matches.parquet")
    parser.add_argument("--readme",    action="store_true", help="Upload data/README.md")
    args = parser.parse_args()

    # Default: upload everything if no flag given
    upload_all = not (args.raw or args.processed or args.readme)

    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
    login(token=token)

    if upload_all or args.raw:
        upload_raw()
    if upload_all or args.processed:
        upload_processed()
    if upload_all or args.readme:
        upload_readme()

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
