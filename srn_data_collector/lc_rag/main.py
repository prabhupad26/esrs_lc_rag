import argparse
import json
import os
from typing import Dict, List

import yaml
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from srn_data_collector.annotations_utils.data_model import (
    EsrsReqMapping,
    RptRequirementsMapping,
    StandardsList,
)
from srn_data_collector.lc_rag.models import Model
from srn_data_collector.lc_rag.utils import format_docs, initialize_env


class WaliMLRetriever(BaseRetriever):
    retriever_config: Dict

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        result = []
        recommended_doc_dict = self.custom_retriever()
        for compliance_item in recommended_doc_dict:
            if recommended_doc_dict[compliance_item]["section_name"] == query:
                recommendations = recommended_doc_dict[compliance_item].get("recommendations", [])
                for recommendation in tqdm(recommendations, desc=f"processing recommendations for {compliance_item}"):
                    blob_data, _ = recommendation
                    result.append(Document(page_content=blob_data.pop("text"), metadata=blob_data))

        if self.retriever_config["debug_doc"].get("top_n"):
            return result[: self.retriever_config["debug_doc"].get("top_n")]
        return result

    def custom_retriever(self):
        if "debug_doc" in self.retriever_config:
            with open(self.retriever_config["debug_doc"]["path"], "r") as f:
                recommended_doc_dict = json.load(f)
        else:
            pass
            # with open(self.retriever_config["wali_ml_training_pkldocs"], 'rb') as f:
            #     training_docs = pkl.load(f)
            # validation_doc_list = [doc["id_"] for doc in training_docs
            #                     if doc['split'].name == "VALIDATION"]
        return recommended_doc_dict


def load_prompt(model_type, prompt_config):
    with open(os.path.join(prompt_config["prompt_dir"], f"{model_type}_{prompt_config['prompt_iu']}")) as f:
        template = f.read()

    return PromptTemplate.from_template(template)


def get_requirements(annotation_storage_config):
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

    all_req = [row[0] for row in result]
    return all_req


def build_chain(**chain_kwargs):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/cluster/home/repo/my_llm_experiments/esrs_data_collection/srn_data_collector/lc_rag/main.yaml",
        type=str,
        help="Path to config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))

    # Initialize environment
    initialize_env(config.pop("gpu_configs"))

    # Get requirements list
    requirements_list = get_requirements(config.pop("annotation_storage_config"))

    # Init and retrieve relevant passages using wali-ml
    retriever_config = config.pop("retriever_config")
    waliml_retriever = WaliMLRetriever(retriever_config=retriever_config)
    # waliml_retriever._get_relevant_documents("E1.AR43",retriever_config=retriever_config, run_manager=None)

    # Init chain
    lc_config = config.pop("lang_chain_config")
    if lc_config.get("enable_langsmith_tracing"):
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true", "LANGCHAIN is not enabled"
        assert "LANGCHAIN_API_KEY" in os.environ, "LANGCHAIN API KEY is not set"
    print(f"Initializing models...")
    model_config = lc_config.pop("model_config")
    model_config = model_config[0]
    model_type = model_config.pop("type")
    model_obj = Model.from_config(type=model_type, name=model_config.pop("name"), **model_config)

    # Load prompt template
    custom_rag_prompt = load_prompt(model_type, config.pop("prompt_config"))

    # Build chain
    rag_chain = (
        {"passage_data": waliml_retriever | format_docs, "sample_requirement": RunnablePassthrough()}
        | custom_rag_prompt
        | model_obj.model_instance
        | StrOutputParser()
    )
    # Execute and Store results
    results = {}

    for req in tqdm(requirements_list, "Processing the list of requirements for given document"):
        results["req"] = rag_chain.invoke(req)

    # Evaluate results
    print(results)


if __name__ == "__main__":
    main()
