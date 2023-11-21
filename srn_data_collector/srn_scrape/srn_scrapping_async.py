import asyncio
import os

import aiohttp
from tqdm import tqdm

srn_api_url = "https://api.sustainabilityreportingnavigator.com/api/"


async def get_srn_companies(session):
    async with session.get(srn_api_url + "companies") as response:
        return await response.json()


async def get_srn_documents(session):
    async with session.get(srn_api_url + "documents") as response:
        return await response.json()


async def download_document(session, id, fpath, href, timeout=60, retry=1):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with session.get(href, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            content = await response.read()
            with open(fpath, "wb") as f, tqdm(
                desc=fpath, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as bar:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    bar.update(len(chunk))
                    f.write(chunk)

    # except asyncio.TimeoutError:
    #     print(fpath, href)
    #     if retry < 3:
    #         print("Going to sleep for 30s, when there is a timeout error")
    #         time.sleep(1)
    #         print("Waking up from sleep, now retrying, when there is a timeout error")
    #         await download_document(session, id, fpath, href, retry=retry+1)

    except Exception as e:
        print(fpath, href)
        print(f"Exception occured : {e}")
        # print(f"Exception occured : {e.message}")

        # if e.message == 'Too Many Requests' and retry < 3:
        #     print("Going to sleep for 30s")
        #     time.sleep(30)
        #     print("Waking up from sleep, now retrying")
        #     await download_document(session, id, fpath, href, retry=retry+1)

        # print("Abort retry and create err file")
        # file_pth = os.path.dirname(fpath)
        # with open(os.path.join(file_pth, "failed.err"), 'w') as f:
        #     f.write(str(e.message))
        #     f.write(href)


async def process_company(session, company, documents, progress_bar):
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
                    await download_document(session, doc_id, final_doc_path, document["href"])

            progress_bar.update(1)


async def main():
    async with aiohttp.ClientSession() as session:
        companies = await get_srn_companies(session)
        documents = await get_srn_documents(session)

        total_tasks = sum(1 for _ in companies for _ in documents)
        progress_bar = tqdm(total=total_tasks, desc="Downloading Documents")

        tasks = [process_company(session, company, documents, progress_bar) for company in companies]
        await asyncio.gather(*tasks)

        progress_bar.close()


if __name__ == "__main__":
    storage_main = "/cluster/home/srn_storage/"
    asyncio.run(main())
