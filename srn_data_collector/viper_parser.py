import os

import torch

os.environ["VIPER_BATCH_SIZE"] = "2"
os.environ["VIPER_DEBUG"] = "true"
os.environ["DATALOADER_NUM_WORKERS"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] =  'max_split_size_mb:512' #134217728
torch.cuda.empty_cache()

import argparse
import datetime
import json
import multiprocessing as mp
import re
import tempfile
from typing import Dict, List

import fitz
import yaml
from fluidml import Flow, Task, TaskSpec, configure_logging
from sqlalchemy import MetaData, Table, and_, create_engine
from tqdm import tqdm
from vipercore.data.data_classes import ParsedDocument
from vipercore.logging import core_logger
from vipercore.model.base import Model
from vipercore.model.prediction import Predictor
from vipercore.parser import convert_pdf
from vipercore.refinement.ocr_refinement import ocr_refinement
from vipercore.refinement.order_refinement import sort_predictions
from vipercore.refinement.rule_refinement import rule_refinement
from vipercore.refinement.text_refinement import text_refinement

from srn_data_collector.annotations_utils.fluidml_helper import (
    TaskResource,
    get_balanced_devices,
    get_raw_pdf_fps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/cluster/home/repo/my_llm_experiments/esrs_data_scraping",
        type=str,
        help="Path to config",
    )

    parser.add_argument(
        "--cuda-ids",
        default=None,
        type=int,
        nargs="+",
        help="GPU ids, e.g. `--cuda-ids 0 1`",
    )

    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Number of workers for multiprocessing",
    )

    parser.add_argument(
        "--cfg-expansion",
        type=str,
        default="product",
        choices=["product", "zip"],
        help="Method to expand config for grid search",
    )

    return parser.parse_args()


def process_annotation_idx(doc_ref_raw: str, error_stat_dict=None) -> tuple[List[int], dict] | None:
    doc_ref_idx = []
    delimiters = [",", "and", "/", ";", "&", "+", "iVm", "#"]

    try:
        for delimiter in delimiters:
            if delimiter in doc_ref_raw:
                doc_ref_idx.extend([int(i.strip()) for i in doc_ref_raw.split(delimiter) if i])
                break
        else:
            if "-" in doc_ref_raw and len(doc_ref_raw.split("-")) == 2:
                start, end = map(int, doc_ref_raw.split("-"))
                doc_ref_idx.extend(range(start, end + 1))
            elif re.match(r"^[+-]?\d+$", doc_ref_raw):
                doc_ref_idx.append(int(doc_ref_raw.strip()))
    except ValueError:
        if error_stat_dict:
            error_stat_dict.setdefault("annotation_pattern_match_ValueError", 0)
            error_stat_dict["annotation_pattern_match_ValueError"] += 1
        core_logger.error(f"Pattern {doc_ref_raw} failed to match, handle this!!!!")

    return doc_ref_idx, error_stat_dict


def get_annotated_page_indicator(
    db_url, doc_ref_table_name: str, pdf_path: str, pdf_img_list: List[str], error_stat_dict: Dict
):
    # Initialize db
    engine = create_engine(db_url)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    doc_ref_table_obj = Table(doc_ref_table_name, metadata, autoload=True, autoload_with=engine)

    # params
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]
    indicators_list = [False for _ in range(len(pdf_img_list))]
    annotated_page_idx = []

    # execute query and fetch results
    with engine.connect() as connection:
        query = doc_ref_table_obj.select().where(
            and_(doc_ref_table_obj.columns.document_id == document_id, doc_ref_table_obj.columns.document_ref.isnot(""))
        )
        result = connection.execute(query)

        for row in result.fetchall():
            refs, error_stat_dict = process_annotation_idx(row.document_ref, error_stat_dict)
            for ref in refs:
                if ref not in annotated_page_idx:
                    annotated_page_idx.append(ref)

    for annotated_page in annotated_page_idx:
        try:
            indicators_list[annotated_page] = True
        except IndexError:
            if "annotation_IndexError" not in error_stat_dict:
                error_stat_dict["annotation_IndexError"] = 0
            error_stat_dict["annotation_IndexError"] += 1
            core_logger.warning(f"Invalid annotated page number for doc id : {document_id}")

    core_logger.debug(f"Found {len(indicators_list)} annotated pages in doc_id : {document_id}")
    return indicators_list, error_stat_dict


def get_page_dict(
    pdf_path: str,
    img_path: str,
    model_path: str,
    annotated_pages_ind_list: List[int] = [],
    device: str = "cuda",
    use_pdf2image=True,
):
    error_statistics = {}
    model = Model(model_path, device=device)

    # Set up predictor
    predictor = Predictor(model, device=device)
    parsed_document = ParsedDocument()

    try:
        # Convert raw pdf to images
        imgs: List[str] = convert_pdf(pdf_path, img_path, use_pdf2image=use_pdf2image)

        annotated_pages_ind_list = [True for _ in range(len(imgs))]

        parsed_document = predictor(parsed_document, imgs)
        # ocr_refinement
        parsed_document = ocr_refinement(parsed_document, imgs, annotated_pages_ind_list=annotated_pages_ind_list)

        # rule refinement
        parsed_document = rule_refinement(parsed_document)

        # text refinement
        parsed_document = text_refinement(pdf_path, parsed_document)

        # sort refinement
        parsed_document = sort_predictions(parsed_document)

        # save json
        task_save_json_file(pdf_path, parsed_document)

        # cleanup
        remove_processed_pdf_images(imgs)

    except torch.cuda.CudaError as e:
        # Handle CUDA-related errors, including OutOfMemoryError
        if "CUDA_Error" not in error_statistics:
            error_statistics["CUDA_Error"] = 0
        error_statistics["CUDA_Error"] += 1
        core_logger.error(f"CUDA Error: {e}")

    except fitz.FileDataError as e:
        if "FileDataError" not in error_statistics:
            error_statistics["FileDataError"] = 0
        error_statistics["FileDataError"] += 1
        core_logger.error(f"FileDataError for file: {pdf_path}")

    except IndexError as ie:
        if "IndexError" not in error_statistics:
            error_statistics["IndexError"] = 0
        error_statistics["IndexError"] += 1
        core_logger.error(f"IndexError for file: {pdf_path}")


def task_process_pdfs(
    pdf_path_list: List[str],
    img_path,
    model_path: str,
    task: Task,
    db_url: str,
    doc_ref_table_name: str,
    error_statistics: Dict,
    error_statistics_path: str,
    use_pdf2image=True,
):
    for pdf_path in tqdm(pdf_path_list.split(";"), desc="Processing pdfs"):
        core_logger.debug("Processing PDF path: %s", pdf_path)

        if not img_path or not os.path.isdir(img_path):
            # Use temporary output directoy
            tmp_path = tempfile.TemporaryDirectory()
            img_path = tmp_path.name

        # Extracting images from pdf
        imgs: List[str] = convert_pdf(pdf_path, img_path, use_pdf2image=use_pdf2image)

        # Get annotated pages indicator list
        annotated_pages_ind_list, error_statistics = get_annotated_page_indicator(
            db_url, doc_ref_table_name, pdf_path, imgs, error_stat_dict=error_statistics
        )

        # Loading predictor
        device = task.resource.device
        # Load model
        model = Model(model_path, device=device)
        # Set up predictor
        predictor = Predictor(model, device=device)
        parsed_document = ParsedDocument()

        try:
            parsed_document = predictor(parsed_document, imgs)
            # ocr_refinement
            parsed_document = ocr_refinement(parsed_document, imgs, annotated_pages_ind_list=annotated_pages_ind_list)

            # rule refinement
            parsed_document = rule_refinement(parsed_document)

            # text refinement
            parsed_document = text_refinement(pdf_path, parsed_document)

            # sort refinement
            parsed_document = sort_predictions(parsed_document)

            # save json
            task_save_json_file(pdf_path, parsed_document)

            # cleanup
            remove_processed_pdf_images(imgs)

        except torch.cuda.CudaError as e:
            # Handle CUDA-related errors, including OutOfMemoryError
            if "CUDA_Error" not in error_statistics:
                error_statistics["CUDA_Error"] = 0
            error_statistics["CUDA_Error"] += 1
            core_logger.error(f"CUDA Error: {e}")

        except fitz.FileDataError as e:
            if "FileDataError" not in error_statistics:
                error_statistics["FileDataError"] = 0
            error_statistics["FileDataError"] += 1
            core_logger.error(f"FileDataError for file: {pdf_path}")

        except IndexError as ie:
            if "IndexError" not in error_statistics:
                error_statistics["IndexError"] = 0
            error_statistics["IndexError"] += 1
            core_logger.error(f"IndexError for file: {pdf_path}")

        finally:
            doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
            if not os.path.isdir(error_statistics_path):
                os.makedirs(error_statistics_path)
            with open(f"{error_statistics_path}/{doc_id}_error_stats.json", "w", encoding="utf8") as file:
                json.dump(error_statistics, file, indent=4)


def remove_processed_pdf_images(pdf_img_lists: List[str]) -> None:
    [os.remove(pdf_img_list) for pdf_img_list in tqdm(pdf_img_lists, desc="removing pdf images")]


def get_page_data_dict(parsed_document_sp: ParsedDocument):
    pages_data_dict = {}
    class_map = parsed_document_sp.class_map
    class_map[-1] = "undefined"
    pages_data = parsed_document_sp.pages
    for page_num, page_data in pages_data.items():
        blobs = page_data.elements
        pages_data_dict[page_num] = []
        for blob in blobs:
            pages_data_dict[page_num].append(
                {
                    "class_id": blob.class_id,
                    "class_name": class_map[blob.class_id],
                    "confidence": float(blob.confidence),
                    "box": [blob.x1, blob.x2, blob.y1, blob.y2],
                    "text": blob.text,
                }
            )
    return pages_data_dict


def task_save_json_file(pdf_path: str, parsed_document_sp: ParsedDocument):
    filename_without_extension = os.path.splitext(os.path.basename(pdf_path))[0]
    file_directory = os.path.dirname(pdf_path)
    output_file = os.path.join(file_directory, f"{filename_without_extension}.json")

    if os.path.isfile(output_file):
        core_logger.error(f"File already exists : {output_file}")
        raise Exception(f"File already exists : {output_file}")

    core_logger.info(f"Creating json {output_file}")

    with open(output_file, "w", encoding="utf8") as file:
        json.dump(get_page_data_dict(parsed_document_sp), file, indent=4)

    core_logger.info(f"Json file save to {output_file}")


def main():
    start = datetime.datetime.now()

    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    base_dir = config.pop("base_dir")
    data_base_path = config.pop("data_base_path")
    model_path = config.pop("model_path")
    pdf2img_out_path = config.pop("pdf2img_out_path")

    # configure logging
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    configure_logging(level="INFO")

    # get balanced devices
    devices = get_balanced_devices(count=args.num_workers, cuda_ids=args.cuda_ids)
    resources = [TaskResource(device=devices[i]) for i in range(args.num_workers)]

    # prerequisites for parsing
    db_url = config.pop("db_url")
    doc_ref_table_name = config.pop("doc_ref_table_name")
    error_statistics_path = config.pop("error_statistics_path")
    error_statistics = {}

    raw_pdfs_splits: List[List[str]] = get_raw_pdf_fps(data_base_path, n_splits=8)
    [
        os.remove(os.path.join(pdf2img_out_path, i))
        for i in tqdm(os.listdir(pdf2img_out_path), desc="removing pdf images")
    ]

    # define tasks
    process_pdfs_task = TaskSpec(
        task=task_process_pdfs,
        config={
            "pdf_path_list": raw_pdfs_splits,
            "model_path": model_path,
            "img_path": pdf2img_out_path,
            "db_url": db_url,
            "doc_ref_table_name": doc_ref_table_name,
            "error_statistics": error_statistics,
            "error_statistics_path": error_statistics_path,
        },
        expand=args.cfg_expansion,
    )

    tasks = [
        process_pdfs_task,
    ]

    flow = Flow(tasks=tasks)
    flow.run(
        num_workers=args.num_workers,
        resources=resources,
        # results_store=results_store
    )

    end = datetime.datetime.now()
    core_logger.info(f"Task finished in {end - start}s")

    [
        os.remove(os.path.join(pdf2img_out_path, i))
        for i in tqdm(os.listdir(pdf2img_out_path), desc="removing pdf images")
    ]


if __name__ == "__main__":
    main()
