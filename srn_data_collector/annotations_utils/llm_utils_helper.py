import concurrent.futures
import json
import os
import re
import traceback
from glob import glob
from typing import Dict, List, Optional

from llm_utils import LLMWrapper
from llm_utils.misc import calculate_api_call_len_and_price, set_openai_api_key
from tqdm import std, tqdm

# Setting up openai api key
set_openai_api_key()


ANNOTATION_PROMPT_TEMPLATE = """
As an expert in sustainability reports and the new CSRD directive, your task is to identify relevant paragraph IDs from a list of paragraphs based on a given regulatory requirement and pre-annotated text value. 
A paragraph is considered relevant if it contains the pre-annotated text value and addresses the provided regulatory requirement.

Requirement: "{compliance_name}" 
Annotated text value: "{annotated_text}" 
Paragraphs: "{paragraphs}"

Please provide your results in the following structured list format, using valid JSON: 
{{"relevant_paragraph_ids": [<Id_1>, <Id_2>, ...]}}
If there are not relevant paragraph return the below json:
{{"relevant_paragraph_ids": []}}
"""

TRANSLATE_PROMPT_TEMPLATE = """
    Translate the following text from english to german language:
    Text to translate: "{source_text}"
    Please respond with the below structured format and no additional text:
    Translated text: <translated text here>
    """
blob_type_ignore_list = ["header/footer", "headline"]


def process_sample(sample: Dict, model: LLMWrapper, storage_path: str):
    results = {"relevant_paragraph_ids": []}

    complete_paragraph = ""
    for i, paragraph in enumerate(sample["blobs"]):
        if paragraph["class_name"] not in blob_type_ignore_list:
            complete_paragraph += "\n\n" + f"\nId {i}: {paragraph}"

    prompt_template = ANNOTATION_PROMPT_TEMPLATE.format(
        compliance_name=sample["compliance_item"],
        annotated_text=sample["annotation"].value.lower(),
        paragraphs=complete_paragraph,
    )
    messages = [{"role": "user", "content": prompt_template}]

    existing_jsons_list = glob(
        os.path.join(storage_path, f"{sample['annotation'].document_id}_{sample['annotation'].id}_*.json")
    )

    if not existing_jsons_list:
        # Annotaion json doesn"t exist, call the api the get results
        try:
            # run model
            results_str = model.run(messages=messages, stream=False, max_attempts=3, max_tokens=256)
            results = json.loads(results_str)

            # Save as json file
            concat_para_ids: str = (
                "_".join(str(results["relevant_paragraph_ids"])) if results["relevant_paragraph_ids"] else ""
            )
            save_path = os.path.join(
                storage_path, f"{sample['annotation'].document_id}_{sample['annotation'].id}_{concat_para_ids}.json"
            )
            if not os.path.exists(save_path):
                with open(save_path, "w") as json_file:
                    json.dump(concat_para_ids, json_file, indent=4)

        except json.decoder.JSONDecodeError:
            results["relevant_paragraph_ids"] = []

        except OSError as oserr:
            print(f"Failed with OS error{oserr} for prompt : {prompt_template}")
            results["relevant_paragraph_ids"] = []
    else:
        # Annotation exists and we read it from the file name
        pattern = r"_(\d+)_"
        for file in existing_jsons_list:
            results["relevant_paragraph_ids"] = re.findall(pattern, file)

    return {
        "annotation": sample["annotation"],
        "blobs": sample["blobs"],
        "matched_blob": results["relevant_paragraph_ids"],
    }


def predict_samples_parallel(samples: List, pbar: Optional[std.tqdm], model_name: str = "gpt-3.5-turbo") -> List:
    model = LLMWrapper(model_name=model_name)

    # run model in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for sample in samples:
            try:
                result = executor.submit(process_sample, sample=sample, model=model)
                futures.append(result)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error occurred while processing prompt {sample}: {e}\n{tb}")

        # Wait for all tasks to complete, regardless of whether they were successful or not
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if pbar is not None:
                pbar.update()
    return results


def process_document_with_llm(
    compliance_name: str, annotated_text: str, list_of_strings: str, model_name: str = "gpt-3.5-turbo"
) -> List[Dict]:
    model = LLMWrapper(model_name=model_name)
    pdf_blobs = "\n\n".join([f"\nId {i}: {paragraph}" for i, paragraph in enumerate(list_of_strings)])
    prompt_template = ANNOTATION_PROMPT_TEMPLATE.format(
        compliance_name=compliance_name, annotated_text=annotated_text, paragraphs=pdf_blobs
    )
    messages = [{"role": "user", "content": prompt_template}]

    result_str = "".join([elem for elem in tqdm(model.run(messages, max_tokens=256, max_attempts=3))])

    # todo: Catch json parsing errors
    try:
        result = json.loads(result_str)
    except json.decoder.JSONDecodeError:
        result = []
    return result


def estimate_api_cost(compliance_name: str, annotated_text: str, list_of_strings: str, max_tokens=2048):
    model = LLMWrapper(model_name="gpt-3.5-turbo")
    pdf_blobs = "\n\n".join([f"\nId {i}: {paragraph}" for i, paragraph in enumerate(list_of_strings)])
    prompt_template = ANNOTATION_PROMPT_TEMPLATE.format(
        compliance_name=compliance_name, annotated_text=annotated_text, paragraphs=pdf_blobs
    )
    return calculate_api_call_len_and_price(prompt_template, model, max_tokens=max_tokens)


def translate_json_parallel(
    model_name: str, source_text_list: List[str], pbar: Optional[std.tqdm], max_tokens: int = 2048
):
    model = LLMWrapper(model_name=model_name)

    # run model in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for sample in source_text_list:
            try:
                result = executor.submit(translate_json, model=model, source_text=sample, max_tokens=max_tokens)
                futures.append(result)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error occurred while processing prompt {sample}: {e}\n{tb}")

        # Wait for all tasks to complete, regardless of whether they were successful or not
        for future in concurrent.futures.as_completed(futures):
            if pbar is not None:
                pbar.update()


def translate_json(model, source_text, max_tokens):
    message = [
        {
            "role": "system",
            "content": "You are a translation assistant who replies with german translated text for the given english text without any english text",
        },
        {"role": "user", "content": source_text["text"]},
    ]
    result_str = model.run(message, max_tokens=max_tokens, max_attempts=3, stream=False)
    try:
        source_text["text"] = result_str.strip()
    except json.decoder.JSONDecodeError:
        print(f"Prompt that couldnt be translated : {source_text['text']} \n Response string {result_str}")


def json_translate_estimate_gpt(source_text, max_tokens):
    model = LLMWrapper(model_name="gpt-3.5-turbo")
    prompt_template = TRANSLATE_PROMPT_TEMPLATE.format(source_text=source_text)

    return calculate_api_call_len_and_price(prompt_template, model, max_tokens=max_tokens)
