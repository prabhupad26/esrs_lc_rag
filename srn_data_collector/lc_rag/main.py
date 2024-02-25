import argparse
import os
import pickle as pkl
import json
import re
from typing import Dict, List

import numpy as np
import wandb
import yaml
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import pandas as pd

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
from srn_data_collector.lc_rag.utils import *


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


def get_source_section_annotations(source_section_dict: str, document_id,  annotation_storage_config: Dict) -> Dict[str, List[List[str]]]:
    annotations = {}
    
    for section in source_section_dict:
        annotations[section] = get_annotations_db(section, document_id, annotation_storage_config)
    
    return annotations
    

def get_annotation_dict(annotation_file, annotation_storage_config) -> Dict[str, List[List[str]]]:

    with open(annotation_file, "r") as f:
        recommended_doc_dict = json.load(f)

    annotation_map = {}

    for compliance_item, recom_items in recommended_doc_dict.items():
        gt = recom_items['ground_truth']
        source_names: List[str] = get_source_from_citem(compliance_item, annotation_storage_config)
        for source_name in source_names:
            if source_name not in annotation_map:
                annotation_map[source_name] = []
            for recommendation, _ in recom_items['recommendations']:
                if gt == recommendation['text'] and \
                recommendation["class_name"] not in ["header/footer"] and \
                (recommendation["blob_id"],recommendation["document_ref"]) not in annotation_map[source_name]:
                    annotation_map[source_name].append((recommendation["blob_id"], 
                                                        recommendation["document_ref"]))
    
    return annotation_map


def parse_chain_output(response_string: str) -> str | None:
    pattern = r"\[\d\](?: > \[\d\])*"
    matches = re.search(pattern, response_string)
    if matches:
        return matches.group(0)
    else:
        return ""


def process_predicted_dict(predicted_dict):
    for rec_label, recommendations in predicted_dict.items():
        for idx, recommendation in enumerate(recommendations):
            if isinstance(recommendation, dict) and parse_chain_output(recommendation["rag_response"]):
                retriever_results = [
                    (result.metadata["blob_id"], result.metadata["document_ref"])
                    for result in recommendation["retriever_results"]
                ]
                if retriever_results:
                    rag_resp = [
                        int(resp.strip().lstrip("[").rstrip("]")) for resp in recommendation["rag_response"].split(">")
                    ]
                    try:
                        recommendation = [retriever_results[idx] for idx in rag_resp]
                    except IndexError:
                        print("Invalid RAG response")
                        recommendation = []
                    recommendations[idx] = list(set(recommendation))
        predicted_dict[rec_label] = [blob for rec in recommendations for blob in rec]



def calculate_metrics(predicted_dict, actual_dict, model_name, 
                      precision_accum, recall_accum, f1_dict, n_support):
    # Fetch the blob_id , document_ref from the db for the given requirement and compare
    process_predicted_dict(predicted_dict)

    y_true = []
    y_pred = []
    labels = list(predicted_dict.keys())
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
    
    # Overall score
    precision, sensitivity, f1_score_value, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")


    n_support = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix
    # cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate metrics for each class and accumulate
    for i in range(len(labels)):
        true_class_idx = y_true == labels[i]
        pred_class_idx = y_pred == labels[i]

        if labels[i] not in precision_accum:
            precision_accum[labels[i]] = 0
        if labels[i] not in recall_accum:
            recall_accum[labels[i]] = 0
        if labels[i] not in f1_dict:
            f1_dict[labels[i]] = 0

        precision_accum[labels[i]] += precision_score(true_class_idx, pred_class_idx)
        recall_accum[labels[i]] += recall_score(true_class_idx, pred_class_idx)
        f1_dict[labels[i]] += f1_score(true_class_idx, pred_class_idx)
    
    
    
    
    # mAP_score = average_precision_score(np.array(y_true) == np.array(y_pred), np.array(y_pred) == np.array(y_true))

    # wandb.log({"sensitivity": sensitivity, "precision": precision, 
    #            "f1_score": f1_score, "model_name": model_name})

    return sensitivity, precision, f1_score_value, n_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="main.yaml",
        type=str,
        help="Path to config",
    )
    return parser.parse_args()


def run(
    document_id: str,
    annotation_storage_config: Dict,
    retriever_type: str,
    req2blob_gt: dict,
    model_config:Dict,
    model_name:str,
    req2blob_gt_db:Dict,
    response_pickle_file:str,
    retriever_config: Dict,
    model_type:str,
    precision_accum: Dict,
    recall_accum:Dict,
    f1_dict:Dict,
    n_support: int,
    **config,
):
    # Initialize environment
    initialize_env(config.pop("gpu_configs"))

    if retriever_type == "cosine-retriever":
        retriever = ChromaDBRetriever(document_id=document_id, retriever_config=retriever_config)
        # Get requirements list
        requirements_dict = retriever.get_requirements(annotation_storage_config)
        retriever = retriever()
        # retriever_results: List[Document] = retriever.invoke("GHG emissions")

    elif retriever_type == "wali-ml":
        retriever = WaliMLRetriever(retriever_config=retriever_config, 
                                    annotation_storage_config=annotation_storage_config)
        # Get requirements list
        requirements_dict = retriever.get_requirements(annotation_storage_config)
        # retriever_results: List[Document] = waliml_retriever._get_relevant_documents("E1.AR43", run_manager=None)

    # req2blob_gt: Dict[str, List[List[str]]] = get_annotation_dict(annotation_file, annotation_storage_config)

    print(f"Initializing models...")
    model_obj = Model.from_config(type=model_type, name=model_name, **model_config)

    # Load prompt template
    custom_rag_prompt = load_prompt(model_type, config.pop("prompt_config"))

    # Build chain
    rag_chain = (
        {"passage_data": retriever | format_docs, "sample_requirement": lambda x: clean_text(requirements_dict[x])}
        | custom_rag_prompt
        | model_obj.model_instance
        | StrOutputParser()
    )
    # Execute and Store results
    results = {}

    for req_label, _ in tqdm(req2blob_gt.items(), "Processing the list of requirements for given document"):
        if req_label not in requirements_dict:
            print(f"Requirement{req_label} text is not available in database")
            # requirements_dict[req_label] = ''
            continue
        
        rag_response = ''
        
        try:
            rag_response = rag_chain.invoke(req_label)
        except SystemError:
            print("Recursion depth mismatch , invalid input")

        if req_label not in results:
            results[req_label] = []

        if retriever_type == "cosine-retriever":
            retriever_results: List[Document] = retriever.invoke(requirements_dict[req_label])
        elif retriever_type == "wali-ml":
            retriever_results: List[Document] = retriever._get_relevant_documents(req_label, run_manager=None)

        # if retriever_results:
        #     print(retriever_results)
        # else:
        #     print(f"{req_label} has not recs")

        results[req_label].append(
            {"rag_response": parse_chain_output(rag_response), "retriever_results": retriever_results}
        )

    # Saving the response
    with open(response_pickle_file, "wb") as f:
        pkl.dump(results, f)

    # Evaluate results
    return calculate_metrics(results, req2blob_gt_db, model_name, 
                             precision_accum, recall_accum, f1_dict, n_support)


def initiate_chain(config, annotation_file, model_name, model_type, model_config,
                   precision_accum, recall_accum, f1_dict, retriever_type,retriever_config,
                   n_support=0):
    annotation_storage_config = config.pop("annotation_storage_config")
    result_path = config.pop("result_path")
    document_id = get_doc_name(annotation_file)
    
    

    # Init chain
    lc_config = config.pop("lang_chain_config")
    if lc_config.get("enable_langsmith_tracing"):
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true", "LANGCHAIN is not enabled"
        assert "LANGCHAIN_API_KEY" in os.environ, "LANGCHAIN API KEY is not set"
    
    
    response_pickle_file = os.path.join(result_path, f"{document_id}_{retriever_type}_{model_name}.pkl")
    
    # req2blob_gt: Dict[str, List[List[str]]] = get_annotation_dict(annotation_file, annotation_storage_config)
    req2blob_gt: Dict[str, List[List[str]]] = get_annotation_dict(annotation_file, annotation_storage_config)
    req2blob_gt_db = get_source_section_annotations(source_section_dict=req2blob_gt,
                                                    document_id=document_id,
                                                    annotation_storage_config=annotation_storage_config)

    if os.path.exists(response_pickle_file):
        # code for debug existing response.pkl file
        with open(response_pickle_file, "rb") as file:
            results = pkl.load(file)
        
        # sensitivity, precision, f1_score = calculate_metrics(results, req2blob_gt)
        sensitivity, precision, f1_score, n_support = calculate_metrics(results, req2blob_gt_db, model_name,
                                                             precision_accum, recall_accum, f1_dict,
                                                             n_support=n_support)
    else:
        sensitivity, precision, f1_score, n_support = run(
                                                document_id=document_id,
                                                annotation_storage_config=annotation_storage_config,
                                                retriever_type=retriever_type,
                                                req2blob_gt=req2blob_gt,
                                                model_config=model_config,
                                                model_name=model_name,
                                                req2blob_gt_db=req2blob_gt_db,
                                                response_pickle_file=response_pickle_file,
                                                retriever_config=retriever_config,
                                                model_type=model_type,
                                                precision_accum=precision_accum,
                                                recall_accum=recall_accum,
                                                f1_dict=f1_dict,
                                                n_support=n_support,
                                                **config
                                            )
    return sensitivity, precision, f1_score, n_support

def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    precision_accum = {}
    recall_accum = {}
    f1_dict = {}
    sensitivity_list = []
    precision_list = []
    f1_score_list = []
    n_support = 0
    metrics_dict = {}

    


    run_wandb = wandb.init(project="thesis-llm")

    annotation_files = config.pop("annotation_files", None)
    retriever_config = config.pop("retriever_config")[0]
    retriever_type = retriever_config.pop("type")
    
    annotation_files = [os.path.join(annotation_files, file)for file in os.listdir(annotation_files)]

    if annotation_files:
        for annotation_file in tqdm(annotation_files, desc="Running main chain for documents"):
            config = yaml.safe_load(open(args.config, "r"))
            # model config
            model_config = config.pop("model_config")
            model_config = model_config[0]
            model_type = model_config.pop("type")
            model_name = model_config.pop("name")
            retriever_config = config.pop("retriever_config")[0]
            retriever_type = retriever_config.pop("type")

            sensitivity, precision, f1_score, n_support = initiate_chain(config, 
                                                          annotation_file=annotation_file,
                                                          model_name=model_name,
                                                          model_type=model_type, 
                                                          model_config=model_config,
                                                          precision_accum=precision_accum,
                                                          recall_accum=recall_accum, 
                                                          f1_dict=f1_dict,
                                                          retriever_type=retriever_type,
                                                          retriever_config=retriever_config,
                                                          n_support=n_support)
            sensitivity_list.append(sensitivity)
            precision_list.append(precision)
            f1_score_list.append(f1_score)

        # wandb.log({"sensitivity": sum(sensitivity_list)/len(annotation_files), 
        #            "precision": sum(precision_list)/len(annotation_files), 
        #            "f1_score": sum(f1_score_list)/len(annotation_files), 
        #            "model_name": model_name})

        wandb.log({"sensitivity_best": max(sensitivity_list), 
                   "precision_best": max(precision_list), 
                   "f1_score_best": max(f1_score_list), 
                   "sensitivity_avg": sum(sensitivity_list) / len(annotation_files), 
                   "precision_avg": sum(precision_list) / len(annotation_files), 
                   "f1_score_avg": sum(f1_score_list) / len(annotation_files), 
                   "model_name": model_name,
                   "retriever_type": retriever_type})

        # Aggregate metrics and log as media to wandb
        for label in precision_accum:
            precision_avg = precision_accum[label] / len(annotation_files)
            recall_avg = recall_accum[label] / len(annotation_files)
            f1_avg = f1_dict[label] / len(annotation_files)

            # Create a dictionary with metrics
            metrics_dict[f'Precision_Class_{label}'] = [precision_avg]
            metrics_dict[f'Recall_Class_{label}'] = [recall_avg]
            metrics_dict[f'F1_Score_Class_{label}'] = [f1_avg]

        # Log metrics as media
        wandb.log({'metrics': wandb.Table(data=pd.DataFrame(metrics_dict), 
                                          columns=list(metrics_dict.keys()))}, commit=False)


    
    else:
        # For single doc debug
        sensitivity, precision, f1_score = initiate_chain(config, 
                                                          annotation_file=config.pop("annotation_file"),
                                                          model_name=model_name,
                                                          model_type=model_type, 
                                                          model_config=model_config,
                                                          precision_accum=precision_accum,
                                                          recall_accum=recall_accum, 
                                                          retriever_type=retriever_type,
                                                          retriever_config=retriever_config,
                                                          f1_dict=f1_dict)


    run_wandb.finish()


if __name__ == "__main__":
    main()