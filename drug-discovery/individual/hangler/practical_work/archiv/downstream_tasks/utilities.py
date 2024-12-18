import os
import pickle
import re
from typing import List, Dict, Tuple

def save_checkpoint(records: List[Dict], base_path: str, checkpoint_index: int):
    filename = f"{base_path}_checkpoint_{checkpoint_index}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(records, f)

    print(f"Checkpoint {checkpoint_index} saved.")

def find_latest_checkpoint(base_path: str) -> Tuple[str, int]:
    pattern = re.compile(r"_checkpoint_(\d+)\.pkl$")
    max_index = -1
    latest_file = None

    for file in os.listdir('.'):
        match = pattern.search(file)
        if match and file.startswith(base_path):
            index = int(match.group(1))
            if index > max_index:
                max_index = index
                latest_file = file

    return latest_file, max_index

def load_checkpoint(filename: str) -> Tuple[List[Dict], int]:
    if filename and os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None, 0
