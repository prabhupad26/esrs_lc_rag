import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from fluidml.storage import ResultsStore, StoreContext
from tqdm import tqdm
from vipercore.data.data_classes import ParsedDocument

logger = logging.getLogger(__name__)


def get_available_devices(use_cuda: bool = True, cuda_ids: Optional[List[int]] = None) -> List[str]:
    if (use_cuda or cuda_ids) and torch.cuda.is_available():
        if cuda_ids is not None:
            devices = [f"cuda:{id_}" for id_ in cuda_ids]
        else:
            devices = [f"cuda:{id_}" for id_ in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    return devices


def get_balanced_devices(
    count: Optional[int] = None,
    use_cuda: bool = True,
    cuda_ids: Optional[List[int]] = None,
    devices: Optional[List[str]] = None,
) -> List[str]:
    if devices is None:
        devices = get_available_devices(use_cuda, cuda_ids)

    count = count if count is not None else mp.cpu_count()
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices


def get_raw_pdf_fps(data_base_path, n_splits=8):
    raw_files = []
    for file_path in tqdm(get_all_files_in_directory(data_base_path), desc="Getting list of raw pdf files"):
        if file_path.endswith(".pdf"):
            # check if json is already created
            filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
            file_directory = os.path.dirname(file_path)
            output_file = os.path.join(file_directory, f"{filename_without_extension}.json")
            if not os.path.isfile(output_file):
                raw_files.append(file_path)
    return [
        ";".join(raw_files[i : i + len(raw_files) // n_splits])
        for i in range(0, len(raw_files), len(raw_files) // n_splits)
    ]


def get_all_files_in_directory(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


@dataclass
class TaskResource:
    device: str


class ParserResultsStore(ResultsStore):
    def load(self, name: str, task_name: str, task_unique_config: Dict, **kwargs) -> Optional[Any]:
        """Query method to load an object based on its name, task_name and task_config if it exists"""
        pass

    def save(self, parsed_doc: ParsedDocument, output_file):
        filename_without_extension = os.path.splitext(os.path.basename(output_file))[0]
        file_directory = os.path.dirname(output_file)
        output_file = os.path.join(file_directory, f"{filename_without_extension}.json")

        if os.path.isfile(output_file):
            raise Exception("File already exists")

        parsed_dict = parsed_doc.to_dict()

        logger.info(f"Creating json {output_file}")

        with open(output_file, "w", encoding="utf8") as file:
            json.dump(parsed_dict, file, indent=4)

        logger.info(f"Json file save to {output_file}")

    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        """Method to delete any artifact"""
        pass

    def delete_run(self, task_name: str, task_unique_config: Dict):
        """Method to delete all task results from a given run config"""
        pass

    def get_context(self, task_name: str, task_unique_config: Dict) -> StoreContext:
        """Method to get store specific storage context, e.g. the current run directory for Local File Store"""
        pass
