import argparse
import os
import pickle as pkl
import re
from typing import Dict, List

import numpy as np
import wandb
import yaml
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from srn_data_collector.annotations_utils.data_model import (
    BlobLvlAnnotations,
    ComplianceItems,
    EsrsReqMapping,
    ReportingRequirements,
    RptRequirementsMapping,
    StandardsList,
    ValuesWithRevisions,
)
from srn_data_collector.lc_rag.models import Model
from srn_data_collector.lc_rag.retriever import ChromaDBRetriever, WaliMLRetriever
from srn_data_collector.lc_rag.utils import format_docs, get_doc_name, initialize_env


def load_prompt(model_type, prompt_config):
    with open(os.path.join(prompt_config["prompt_dir"], f"{model_type}_{prompt_config['prompt_iu']}")) as f:
        template = f.read()

    return PromptTemplate.from_template(template)


def get_requirements(waliml_retriever, annotation_storage_config):
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

    all_req = {}

    for row in result:
        row = row[0]
        retrieved_docs = waliml_retriever._get_relevant_documents(row, run_manager=None)
        if row in requirement_texts and len(retrieved_docs) > 0:
            all_req[row] = requirement_texts[row]

    return all_req


def get_annotation_dict(document_id, annotation_storage_config) -> Dict[str, List[List[str]]]:
    file_name, _ = os.path.splitext(os.path.basename(document_id))

    Base = declarative_base()
    engine = create_engine(f"sqlite:///{annotation_storage_config['sqlite_db']}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    result = (
        session.query(RptRequirementsMapping.source, BlobLvlAnnotations.blob_id, BlobLvlAnnotations.document_ref)
        .join(ValuesWithRevisions, ValuesWithRevisions.document_id == BlobLvlAnnotations.document_id)
        .join(ComplianceItems, ComplianceItems.id == ValuesWithRevisions.compliance_item_id)
        .join(ReportingRequirements, ReportingRequirements.id == ComplianceItems.reporting_requirement_id)
        .join(RptRequirementsMapping, ReportingRequirements.id == RptRequirementsMapping.reporting_requirement_id)
        .join(StandardsList, StandardsList.id == RptRequirementsMapping.standard)
        .filter(StandardsList.family.like("%esrs%"))
        .filter(ValuesWithRevisions.document_id == f"{file_name}")
        .all()
    )

    annotation_map = {}

    for row in result:
        req_label, blob_id, document_ref = row
        if req_label not in annotation_map:
            annotation_map[req_label] = []
        if (blob_id, document_ref) not in annotation_map[req_label]:
            annotation_map[req_label].append((blob_id, document_ref))

    session.close()

    return annotation_map


def parse_chain_output(response_string: str) -> str | None:
    pattern = r"\[\d\](?: > \[\d\])*"
    matches = re.search(pattern, response_string)
    if matches:
        return matches.group(0)
    else:
        return ""


def process_predicted_dict(predicted_dict):
    for _, recommendations in predicted_dict.items():
        for idx, recommendation in enumerate(recommendations):
            if isinstance(recommendation, dict) and parse_chain_output(recommendation["rag_response"]):
                retriever_results = [
                    (result.metadata["blob_id"], result.metadata["doc_ref"])
                    for result in recommendation["retriever_results"]
                ]
                if retriever_results:
                    rag_resp = [
                        int(resp.strip().lstrip("[").rstrip("]")) for resp in recommendation["rag_response"].split(">")
                    ]
                    recommendation = [retriever_results[idx] for idx in rag_resp]
                    recommendations[idx] = recommendation


def calculate_metrics(predicted_dict, actual_dict):
    # Fetch the blob_id , docu_ref from the db for the given requirement and compare
    process_predicted_dict(predicted_dict)

    y_true = []
    y_pred = []
    for label in actual_dict:
        for actual_value in actual_dict[label]:
            y_true.append(label)
            if label in predicted_dict and actual_value in predicted_dict[label]:
                y_pred.append(label)
            else:
                y_pred.append("None")

    for label in predicted_dict:
        if label not in actual_dict:
            for predicted_value in predicted_dict[label]:
                y_true.append("None")
                y_pred.append(label)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    sensitivity = recall
    mAP_score = average_precision_score(np.array(y_true) == np.array(y_pred), np.array(y_pred) == np.array(y_true))

    wandb.log({"sensitivity": sensitivity, "mAP_score": mAP_score, "f1_score": f1_score})

    return sensitivity, mAP_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="main.yaml",
        type=str,
        help="Path to config",
    )
    return parser.parse_args()


def main(
    annotation_file: str,
    annotation_storage_config: Dict,
    document_id: str,
    retriever_type: str,
    req2blob_gt: dict,
    **config,
):
    # Initialize environment
    initialize_env(config.pop("gpu_configs"))

    if retriever_type == "cosine-retriever":
        retriever = ChromaDBRetriever(retriever_config=retriever_config)
        # Get requirements list
        requirements_dict = retriever.get_requirements(annotation_storage_config)
        retriever = retriever()
        # retriever_results: List[Document] = retriever.invoke("GHG emissions")

    elif retriever_type == "wali-ml":
        retriever = WaliMLRetriever(retriever_config=retriever_config)
        # Get requirements list
        requirements_dict = retriever.get_requirements(annotation_storage_config)
        # retriever_results: List[Document] = waliml_retriever._get_relevant_documents("E1.AR43", run_manager=None)

    # req2blob_gt: Dict[str, List[List[str]]] = get_annotation_dict(annotation_file, annotation_storage_config)

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
        {"passage_data": retriever | format_docs, "sample_requirement": lambda x: requirements_dict[x]}
        | custom_rag_prompt
        | model_obj.model_instance
        | StrOutputParser()
    )
    # Execute and Store results
    results = {}

    for req_label, _ in tqdm(req2blob_gt.items(), "Processing the list of requirements for given document"):
        rag_response = rag_chain.invoke(req_label)

        if req_label not in results:
            results[req_label] = []

        if retriever_type == "cosine-retriever":
            retriever_results: List[Document] = retriever.invoke(requirements_dict[req_label])
        elif retriever_type == "wali-ml":
            retriever_results: List[Document] = retriever._get_relevant_documents(req_label, run_manager=None)

        results[req_label].append(
            {"rag_response": parse_chain_output(rag_response), "retriever_results": retriever_results}
        )

    # Saving the response
    with open(f"results/{document_id}_{retriever_type}.pkl", "wb") as f:
        pkl.dump(results, f)

    # Evaluate results
    calculate_metrics(results, req2blob_gt)


if __name__ == "__main__":
    # run = wandb.init(project="thesis-llm")

    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))

    annotation_file = config.pop("annotation_file")
    annotation_storage_config = config.pop("annotation_storage_config")
    result_path = config.pop("result_path")
    document_id = get_doc_name(annotation_file)
    retriever_config = config.pop("retriever_config")[0]
    retriever_type = retriever_config.pop("type")
    response_pickle_file = os.path.join(result_path, f"{document_id}_{retriever_type}.pkl")

    req2blob_gt: Dict[str, List[List[str]]] = get_annotation_dict(annotation_file, annotation_storage_config)

    if os.path.exists(response_pickle_file):
        # code for debug existing response.pkl file
        with open(response_pickle_file, "rb") as file:
            results = pkl.load(file)
    else:
        main(
            annotation_file=annotation_file,
            annotation_storage_config=annotation_storage_config,
            document_id=document_id,
            retriever_type=retriever_type,
            req2blob_gt=req2blob_gt,
            **config,
        )

    sensitivity, mAP_score, f1_score = calculate_metrics(results, req2blob_gt)

    # run.finish()
