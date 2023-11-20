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
import tempfile
from typing import List

import fitz
import yaml
from fluidml import Flow, Task, TaskSpec, configure_logging
from fluidml_helper import TaskResource, get_balanced_devices, get_raw_pdf_fps
from pdfminer.pdfparser import PDFSyntaxError
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


def task_process_pdfs(pdf_path_list: List[str], img_path, model_path: str, task: Task, use_pdf2image=True):
    for pdf_path in tqdm(pdf_path_list.split(";"), desc="Processing pdfs"):
        core_logger.debug("Processing PDF path: %s", pdf_path)

        if not img_path or not os.path.isdir(img_path):
            # Use temporary output directoy
            tmp_path = tempfile.TemporaryDirectory()
            img_path = tmp_path.name

        # Extracting images from pdf
        imgs: List[str] = convert_pdf(pdf_path, img_path, use_pdf2image=use_pdf2image)

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
            parsed_document = ocr_refinement(parsed_document, imgs)

            # rule refinement
            parsed_document = rule_refinement(parsed_document)

            # text refinement
            parsed_document = text_refinement(pdf_path, parsed_document)

            # sort refinement
            parsed_document = sort_predictions(parsed_document)

            # save json
            task_save_json_file(pdf_path, parsed_document)
        except torch.cuda.CudaError as e:
            # Handle CUDA-related errors, including OutOfMemoryError
            core_logger.error(f"CUDA Error: {e}")

        except PDFSyntaxError as pdfse:
            core_logger.error(f"PDF syntax errors: {pdfse}")

        except fitz.FileDataError as e:
            core_logger.error(f"FileDataError for file: {pdf_path}")


# def task_convert_pdf_fn(pdf_path: str, img_path: str, task:Task, use_pdf2image=True):

#     if not img_path or not os.path.isdir(img_path):
#         # Use temporary output directoy
#         tmp_path = tempfile.TemporaryDirectory()
#         img_path = tmp_path.name

#         core_logger.debug("Setting up temporary directory for converted PDF images: %s",
#                           tmp_path.name)

#     imgs: List[str] = convert_pdf(pdf_path, img_path, use_pdf2image=use_pdf2image)

#     task.save(imgs, name="pdf_images_list")
#     task.save(pdf_path, name="pdf_path")


# def task_predictor(pdf_images_list:List[str], model_path: str, task: Task):

#     device = task.resource.device

#     # Load model
#     model = Model(model_path, device=device)

#     # Set up predictor
#     predictor = Predictor(model, device=device)

#     parsed_document = ParsedDocument()

#     parsed_document = predictor(parsed_document, pdf_images_list)

#     task.save(parsed_document, name="parsed_document")


# def task_ocr_refinement(parsed_document, pdf_images_list, task: Task):

#     parsed_document = ocr_refinement(parsed_document, pdf_images_list)

#     task.save(parsed_document, name="parsed_document_ocr")


# def task_rule_refinement(parsed_document_ocr, task: Task):

#     parsed_document = rule_refinement(parsed_document_ocr)

#     task.save(parsed_document, name="parsed_document_rr")


# def task_text_refinement(pdf_path, parsed_document_rr, task: Task):

#     parsed_document = text_refinement(pdf_path, parsed_document_rr)

#     task.save(parsed_document, name="parsed_document_tr")


# def task_sort_predictions(parsed_document_tr, task:Task):

#     parsed_document = sort_predictions(parsed_document_tr)

#     task.save(parsed_document, name="parsed_document_sp")


def get_page_data_dict(parsed_document_sp: ParsedDocument):
    pages_data_dict = {}
    class_map = parsed_document_sp.class_map
    class_map[-1] = "undefined"  # ask daniel about this -1
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
    # results_store = ParserResultsStore()

    # prerequisites for parsing
    raw_pdfs_splits: List[List[str]] = get_raw_pdf_fps(data_base_path, n_splits=8)
    [
        os.remove(os.path.join(pdf2img_out_path, i))
        for i in tqdm(os.listdir(pdf2img_out_path), desc="removing pdf images")
    ]

    # define tasks
    process_pdfs_task = TaskSpec(
        task=task_process_pdfs,
        config={"pdf_path_list": raw_pdfs_splits[:4], "model_path": model_path, "img_path": pdf2img_out_path},
        expand=args.cfg_expansion,
    )

    # predictor_task = TaskSpec(task=task_predictor,
    #                        config= {"model_path": model_path})

    # ocr_refinement_task = TaskSpec(task=task_ocr_refinement)

    # rule_refinement_task = TaskSpec(task=task_rule_refinement)

    # text_refinement_task = TaskSpec(task=task_text_refinement)

    # sort_predictions_task = TaskSpec(task=task_sort_predictions)

    # save_json_file_task = TaskSpec(task=task_save_json_file)

    # define dependencies
    # predictor_task.requires(convert_pdf_task)

    # ocr_refinement_task.requires(predictor_task, convert_pdf_task)

    # rule_refinement_task.requires(ocr_refinement_task)

    # text_refinement_task.requires(rule_refinement_task)

    # sort_predictions_task.requires(text_refinement_task)

    # save_json_file_task.requires(convert_pdf_task, sort_predictions_task)

    # run tasks
    # tasks = [convert_pdf_task, predictor_task, ocr_refinement_task,
    #          rule_refinement_task, text_refinement_task,
    #          sort_predictions_task, save_json_file_task]

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
