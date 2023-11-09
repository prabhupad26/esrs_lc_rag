import os

import requests
from tqdm import tqdm

srn_api_url = "https://api.sustainabilityreportingnavigator.com/api/"


def get_srn_companies():
    """
    Returns a list of companies that are included in SRN Document Database.

    Returns:
        [list{dict}]: A list containg company level metadata
    """
    response = requests.get(srn_api_url + "companies")
    return response.json()


def get_srn_documents():
    """
    Returns a list of documents that are included in SRN Document Database.

    Returns:
        [list{dict}]: A list containg document level metadata
    """
    response = requests.get(srn_api_url + "documents")
    return response.json()


def download_document(id, fpath, href, timeout=60):
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
        response = requests.get(href, timeout=timeout, stream=True, headers=headers, verify=False)
        total_size = int(response.headers.get("content-length", 0))
        with open(fpath, "wb") as file, tqdm(
            desc=fpath, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                file.write(data)
    except requests.exceptions.ReadTimeout:
        print(f"File id : {id} failed with read timeout err , href : {href}")
        return -1

    except Exception as e:
        print(e)
        print(f"Doc download failed for : {id}, href : {href}")
        return -1


if __name__ == "__main__":
    storage_main = "/cluster/home/srn_storage/"
    companies = get_srn_companies()
    documents = get_srn_documents()

    for company in tqdm(companies, desc=f"Processing companies "):
        company_name = company["name"].replace("'", "")
        company_id = company["id"]
        if company_name:
            for document in documents:
                if company_id == document["company_id"]:
                    doc_id = document["id"]
                    doc_year = document["year"]
                    doc_type = document["type"]
                    doc_path = os.path.join(storage_main, company_id, doc_year, doc_type)
                    if not os.path.exists(doc_path):
                        os.makedirs(doc_path)
                    final_doc_path = os.path.join(doc_path, f"{doc_id}{os.path.splitext(document['href'])[-1]}")
                    if not os.path.isfile(final_doc_path):
                        download_document(doc_id, final_doc_path, document["href"])

    # print("Searching company with a name containing 'Allianz'")
    # matches = [c for c in companies if 'Allianz' in c['name']]
    # docs = [d for d in documents if d['company_id'] == matches[0]['id']]
    # print(
    #     f"Found {len(matches)} match(es). " +
    #     "Retrieving the documents for the first match."
    #     f"With {len(docs)} documents"
    # )
    # FPATH = 'test_srn_docs.pdf'
    # print(
    #     f"Found {len(docs)} documents. " +
    #     "Retrieving the first document from the list " +
    #     f"and storing it as '{FPATH}'."
    # )
    # download_document(docs[0]['id'], FPATH)
    # print("done!")
