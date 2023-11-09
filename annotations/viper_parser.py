import os

os.environ["VIPER_BATCH_SIZE"] = "16"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] =  'max_split_size_mb:768' #134217728
# torch.cuda.empty_cache()

import json

from tqdm import tqdm
from vipercore.parser import parse

os.environ["VIPER_DEBUG"] = "true"


base_path = "/cluster/home/srn_storage"
model_path = "/cluster/home/repo/my_llm_experiments/FRCNN_cbo_275.viper"


def get_all_files_in_directory(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


files_in_directory = get_all_files_in_directory(base_path)

for file_path in tqdm(files_in_directory, desc="Processing pdf files"):
    if file_path.endswith(".pdf"):
        filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        file_directory = os.path.dirname(file_path)
        output_file = os.path.join(file_directory, f"{filename_without_extension}.json")
        if not os.path.isfile(output_file):
            parsed_doc = parse(file_path, model_path=model_path)
            parsed_dict = parsed_doc.to_dict()

            print(f"creating json {output_file}")

            with open(output_file, "w", encoding="utf8") as file:
                json.dump(parsed_dict, file, indent=4)
        else:
            print(f"File: {output_file} already exists")
