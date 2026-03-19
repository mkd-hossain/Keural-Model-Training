#!/usr/bin/env python3

import os
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi

# ===== CONFIG =====
REPO_ID = "mkd-hossain/keural-stage1-dataset"
DATA_FOLDER = "data/binary"

# Files to ignore
IGNORE = ["locks", "logs", "state"]

api = HfApi()

def get_files(folder):
    files = []
    for root, dirs, filenames in os.walk(folder):

        # remove ignored folders
        dirs[:] = [d for d in dirs if d not in IGNORE]

        for f in filenames:
            path = os.path.join(root, f)
            files.append(path)

    return files


def upload():

    files = get_files(DATA_FOLDER)

    print(f"\nTotal files to upload: {len(files)}\n")

    for file in tqdm(files, desc="Uploading files"):

        path = Path(file)

        repo_path = str(path.relative_to(DATA_FOLDER))

        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )


if __name__ == "__main__":
    upload()