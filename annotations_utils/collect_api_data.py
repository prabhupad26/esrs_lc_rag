import requests
from tqdm import tqdm

srn_api_url = "https://api.sustainabilityreportingnavigator.com/api/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def download_document(fpath, href, timeout=60):
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
    try:
        response = requests.get(
            href,
            # timeout=timeout,
            stream=True,
            headers=headers,
        )
        total_size = int(response.headers.get("content-length", 0))
        with open(fpath, "wb") as file, tqdm(
            desc=fpath, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                file.write(data)
    except requests.exceptions.ReadTimeout:
        print(f"File : {fpath} failed with read timeout err , href : {href}")
        return -1
    except Exception as e:
        print(e)
        print(f"Failed to save {fpath} with link : {href}")


def get_srn_companies():
    """
    Returns a list of companies that are included in SRN Document Database.
    response json structure
    [{
        "id": "8dee5d4e-2b5d-44c4-a78d-2d5d8dd92df1",
        "name": "1&1",
        "isin": "DE0005545503",
        "country": "Germany",
        "sector": "Media & Entertainment",
        "href": "",
        "href_logo": "",
        "company_type": "public",
        "indices": [
            "c233b29e-f073-426f-88cf-9e9d5e645e6e"
        ]
    }]
    """
    response = requests.get(srn_api_url + "companies")
    return response.json()


def get_srn_documents(document_id: str):
    """
    response json structure
    {
    "id": "5160aa1d-ee97-4b85-812b-132aadb87b40",
    "name": "Electrolux Sustainability Report 2022",
    "href": "https://www.electroluxgroup.com/wp-content/uploads/sites/2/2023/03/sustainability-report-2022.pdf",
    "type": "SR",
    "year": "2022",
    "company_id": "fa1f8fee-2fec-446d-a753-530473c28a24",
    "created_at": "2023-10-14T10:30:05.879596",
    "created_by_info": null
    }
    """
    response = requests.get(srn_api_url + f"documents/{document_id}")
    return response.json()


def get_financial_index(fin_idx):
    """
    Response structure :
    {
    "id": "740401de-1916-45a6-a928-137dd17c602f",
    "name": "Prime Standard Deutsche Börse",
    "name_short": "dbPrime",
    "country": "Germany",
    "coverage": "general",
    "updated": null,
    "href": "https://deutsche-boerse.com/dbg-de/unternehmen/wissen/boersenlexikon/boersenlexikon-article/Prime-Standard-243290"
    }
    """
    response = requests.get(srn_api_url + f"indices/{fin_idx}")
    if response.status_code != 200:
        raise Exception(f"API failed with {response.status_code}")
    return response.json()


def get_compliance_item_instance():
    """
    {
    "id": "d0730bfa-dab7-43c0-b2ac-af0a4946e865",
    "reporting_requirement_id": "938363e2-d6ec-480a-b4d2-9e2d30403724",
    "name": "gross scope 2 GHG emissions for associates, joint ventures",
    "status": "deprecated"
    }
    """
    response = requests.get(srn_api_url + f"compliance_items")
    if response.status_code != 200:
        raise Exception(f"API failed with {response.status_code}")
    return response.json()


def get_reporting_requirement_instance(reporting_requirement_id: str):
    """
    {
    "id": "d0730bfa-dab7-43c0-b2ac-af0a4946e865",
    "reporting_requirement_id": "938363e2-d6ec-480a-b4d2-9e2d30403724",
    "name": "gross scope 2 GHG emissions for associates, joint ventures",
    "status": "deprecated"
    }
    """
    response = requests.get(srn_api_url + f"reporting_requirements/{reporting_requirement_id}")
    if response.status_code != 200:
        raise Exception(f"API failed with {response.status_code}")
    return response.json()


def get_values_with_revisions(company_instance_id: str):
    """
    annotations api
    """
    response = requests.get(srn_api_url + f"companies/{company_instance_id}/values_with_revisions")
    if response.status_code != 200:
        raise Exception(f"API failed with {response.status_code}")
    return response.json()
