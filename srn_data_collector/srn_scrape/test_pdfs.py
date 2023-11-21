import os
from fnmatch import fnmatch

import fitz
from tqdm import tqdm


def get_all_files_in_directory(directory, wildcard=None):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if wildcard and not fnmatch(file, wildcard):
                continue  # Skip files that don't match the wildcard
            all_files.append(file_path)
    return all_files


if __name__ == "__main__":
    all_files = get_all_files_in_directory("/cluster/home/srn_storage", "*.pdf")
    results_dict = {"good": 0, "bad": 0}
    for file in tqdm(all_files, desc="Checking good files cnt"):
        try:
            fitz.open(file)
            results_dict["good"] += 1
        except fitz.FileDataError as e:
            results_dict["bad"] += 1
    print(results_dict)
