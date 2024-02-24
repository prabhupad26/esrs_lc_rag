import json
import os
from typing import Dict, List

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from srn_data_collector.annotations_utils.data_model import (
    EsrsReqMapping,
    RptRequirementsMapping,
    StandardsList,
)
from srn_data_collector.lc_rag.utils import convert_file_name, metadata_func, get_source_from_citem, clean_text


class WaliMLRetriever(BaseRetriever):
    retriever_config: Dict
    annotation_storage_config: Dict

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Collect the blobs of all the compliance items in a list for every requirement and return top_n blobs
        """
        result = []
        result_buf = []
        recommended_doc_dict = self.custom_retriever()
        # for compliance_item in recommended_doc_dict:
        #     blob_cnt = 0
        #     if recommended_doc_dict[compliance_item]["section_name"] == query:
        #         recommendations = recommended_doc_dict[compliance_item].get("recommendations", [])
        #         recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        #         for recommendation in recommendations:
        #             blob_data, _ = recommendation
        #             if blob_data["class_name"] not in ["header/footer"]:
        #                 result.append(Document(page_content=blob_data.pop("text"), metadata=blob_data))
        #                 blob_cnt += 1
        #                 if blob_cnt > self.retriever_config["debug_doc"].get("top_n"):
        #                     break
        for compliance_item, recom_items in recommended_doc_dict.items():
            source_names: List[str] = get_source_from_citem(compliance_item, self.annotation_storage_config)
            for source_name in source_names:
                if source_name == query:
                    for recommendation, _ in recom_items['recommendations']:
                        if recommendation["class_name"] not in ["header/footer"] and \
                        (recommendation["blob_id"],recommendation["document_ref"]) not in result_buf:
                            result_buf.append((recommendation["blob_id"],recommendation["document_ref"]))
                            result.append(Document(page_content=clean_text(recommendation.pop("text")), metadata=recommendation))

        if self.retriever_config["debug_doc"].get("total_top_n"):
            return result[: self.retriever_config["debug_doc"].get("total_top_n")]
        return result

    def custom_retriever(self):
        if "debug_doc" in self.retriever_config:
            with open(self.retriever_config["debug_doc"]["path"], "r") as f:
                recommended_doc_dict = json.load(f)
        else:
            pass
            # TODO: Implementation for documents in batches
            # with open(self.retriever_config["wali_ml_training_pkldocs"], 'rb') as f:
            #     training_docs = pkl.load(f)
            # validation_doc_list = [doc["id_"] for doc in training_docs
            #                     if doc['split'].name == "VALIDATION"]
        return recommended_doc_dict

    def get_requirements(self, annotation_storage_config: Dict) -> Dict:
        """
        Get the requirements for which the annotation exists
        """
        Base = declarative_base()
        engine = create_engine(f"sqlite:///{annotation_storage_config['sqlite_db']}")
        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        result = (
            session.query(RptRequirementsMapping.source.distinct())
            .join(StandardsList, RptRequirementsMapping.standard == StandardsList.id)
            .filter(StandardsList.family.like("%esrs%"))
            .all()
        )

        requirement_texts = session.query(EsrsReqMapping.section_name, EsrsReqMapping.text).distinct().all()
        requirement_texts = {k: v for k, v in requirement_texts}

        # all_req = {}

        # for row in result:
        #     row = row[0]
        #     retrieved_docs = self._get_relevant_documents(row, run_manager=None)
        #     if row in requirement_texts and len(retrieved_docs) > 0:
        #         all_req[row] = requirement_texts[row]

        session.close()

        return requirement_texts


class ChromaDBRetriever:
    def __init__(self, document_id, retriever_config) -> None:
        self.document_id = document_id
        self.retriever_config = retriever_config

    def __call__(self):
        return self.load_and_process_raw()

    def load_and_process_raw(self):
        """
        Method to load, process and build vectorstore
        """
        raw_json_path = os.path.join(self.retriever_config.pop("file_path"), f"{self.document_id}.json")
        raw_json_path_processed = convert_file_name(raw_json_path)
        search__topk = self.retriever_config.pop("search__topk")

        # load sample data
        with open(raw_json_path, "r") as f:
            sample_doc_dict = json.load(f)

        # process sample data
        doc_formatted_data = {"doc_data": []}
        for doc_ref, blob_list in sample_doc_dict.items():
            for blob_id, blob_data in enumerate(blob_list):
                if blob_data["class_name"] not in ["header/footer"] and len(blob_data["text"]) > 50:
                    doc_formatted_data["doc_data"].append(
                        {"text": blob_data["text"], "blob_id": blob_id, "document_ref": doc_ref}
                    )

        # dump data
        with open(raw_json_path_processed, "w") as f:
            json.dump(doc_formatted_data, f)

        # build vector db
        doc_loader = JSONLoader(file_path=raw_json_path_processed, metadata_func=metadata_func, **self.retriever_config)
        doc_data = doc_loader.load()

        # Build vectorstore
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # splits_doc = text_splitter.split_documents(doc_data)
        vectorstore_doc = Chroma.from_documents(documents=doc_data, embedding=OpenAIEmbeddings())
        retriever_doc = vectorstore_doc.as_retriever(search_type="similarity", search_kwargs={"k": search__topk})
        return retriever_doc

    def get_requirements(self, annotation_storage_config: Dict) -> Dict:
        """
        Get the requirements for which the annotation exists
        """
        Base = declarative_base()
        engine = create_engine(f"sqlite:///{annotation_storage_config['sqlite_db']}")
        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        result = (
            session.query(RptRequirementsMapping.source.distinct())
            .join(StandardsList, RptRequirementsMapping.standard == StandardsList.id)
            .filter(StandardsList.family.like("%esrs%"))
            .all()
        )

        requirement_texts = session.query(EsrsReqMapping.section_name, EsrsReqMapping.text).distinct().all()
        requirement_texts = {k: v for k, v in requirement_texts}

        # all_req = {}

        # for row in result:
        #     row = row[0]
        #     if row in requirement_texts:
        #         all_req[row] = requirement_texts[row]

        session.close()

        return requirement_texts
