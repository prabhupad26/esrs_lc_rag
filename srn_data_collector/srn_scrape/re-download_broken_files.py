import json
import os

import fitz
import requests
from tqdm import tqdm


def check_if_invalid_pdf(fpath: str):
    if os.path.isfile(fpath):
        try:
            fitz.open(fpath)
        except fitz.FileDataError as e:
            print(f"Removing {fpath} as its an invalid pdf file")
            os.remove(fpath)
            return True
        return False
    else:
        print(f"File doesn't exist, will download it anyway")
        return True


def download_document(doc_id: str, main_dataset_path: str, timeout=120):
    """
    Retreives a certain document from the SRN Document Database and
    stores it at the provided file path.

    Args:
        id (str): The SRN document id.
        fpath (str): A sting containt the file path where you want to
            store the file.
        timeout (int, optional): Sometimes, a download API call might
            nlock because of a dying connection or because the data
            is not available. If a timeout is reached, the according
            API request will raise an exception and exit.
            Defaults to 60 seconds.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        # Doc metadata
        response_meta = requests.get(
            f"https://api.sustainabilityreportingnavigator.com/api/documents/{doc_id}",
            timeout=timeout,
            stream=True,
            headers=headers,
            verify=False,
        )
        if response_meta.status_code == 200:
            response_meta_dict = response_meta.json()

            href = response_meta_dict["href"]
            company_id = response_meta_dict["company_id"]
            year = response_meta_dict["year"]
            type = response_meta_dict["type"]
            fpath = os.path.join(main_dataset_path, company_id, year, type, f"{doc_id}.pdf")

            # Remove invalid file
            if check_if_invalid_pdf(fpath):
                # Download file
                response = requests.get(
                    f"https://api.sustainabilityreportingnavigator.com/api/documents/{doc_id}/download",
                    # href,
                    timeout=timeout,
                    stream=True,
                    headers=headers,
                    verify=False,
                )
                total_size = int(response.headers.get("content-length", 0))
                with open(fpath, "wb") as file, tqdm(
                    desc=fpath, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        bar.update(len(data))
                        file.write(data)
                print(f"File {fpath} redownloaded successfully!")

    except requests.exceptions.ReadTimeout:
        print(f"File id : {id} failed with read timeout err , href : {href}")
        return -1

    except Exception as e:
        print(e)
        print(f"Doc download failed for : {id}, href : {href}")
        return -1


if __name__ == "__main__":
    main_dataset_path = "/cluster/home/srn_storage"
    # err_base_pth = '/cluster/home/srn_storage_error_stat_dir'

    # all_files = os.listdir(err_base_pth)

    # all_stat = {}
    # for file in tqdm(all_files, desc="processing err jsons"):
    #     if file.endswith(".json"):
    #         with open(os.path.join(err_base_pth, file), 'r') as f:
    #             try:
    #                 data_dict = json.load(f)
    #             except json.decoder.JSONDecodeError:
    #                 print(f"{file} is not ready yet")
    #         for data_lbl, data in data_dict.items():
    #             if data_lbl not in all_stat:
    #                 all_stat[data_lbl] = data
    #             all_stat[data_lbl] += data
    #             if data_lbl == "FileDataError":
    #                 if "err_file_names" not in all_stat:
    #                     all_stat["err_file_names"] = []
    #                 all_stat["err_file_names"].append(file)

    # err_files = list(set(all_stat["err_file_names"]))

    err_json = "/cluster/home/repo/my_llm_experiments/esrs_data_collection/srn_data_collector/logs/err_stat.json"
    with open(err_json, "r") as f:
        data_dict = json.load(f)

    err_files = list(set(data_dict["FileNotFoundError"]))

    for err_file in tqdm(err_files, desc="retrying to download invalid files"):
        err_file = os.path.splitext(err_file)[0].rstrip("_error_stats")
        download_document(err_file, main_dataset_path)
