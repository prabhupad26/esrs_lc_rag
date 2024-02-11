import json
import re


def clean_text(text):
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s$|^\s", "", text)
    return text


def find_text_dict(data, label):
    requirement_data_dict_list = []
    for k, d in data.items():
        new_label = f"{label}:::{k}"
        if isinstance(d, dict):
            if "text" in d:
                requirement_data_dict_list.append({"text": clean_text(d["text"]), "label": new_label})
            if "sub_req" in d:
                requirement_data_dict_list.extend(find_text_dict(d["sub_req"], label))
    return requirement_data_dict_list


def parse_data(src_file_name):
    categories_dict = {
        "climate change": [],
        "pollution": [],
        "water and marine resources": [],
        "biodiversity and ecosystems": [],
        "circular economy": [],
        "own workforce": [],
        "workers in the value chain": [],
        "affected communities": [],
        "consumers and end-users": [],
        "business conduct": [],
    }
    with open(src_file_name, "r") as f:
        data_dict = json.load(f)

        for esrs_type_label, esrs_type_data in data_dict.items():
            esrs_type_label = esrs_type_label.replace(" ", "_")
            for esrs_item_label, esrs_item in esrs_type_data.items():
                esrs_item_label = f"{esrs_type_label}:::{esrs_item_label.replace(' ', '_')}"
                if "topic" in esrs_item.keys():
                    esrs_item["topic"] = re.sub(r"\s+", " ", esrs_item["topic"])
                    esrs_item["topic"] = esrs_item["topic"].replace("end- ", "end-")
                    for esrs_item_req_label, esrs_item_req_data in esrs_item.items():
                        if isinstance(esrs_item_req_data, dict):
                            esrs_item_req_label = f"{esrs_item_label}:::{esrs_item_req_label}"
                            categories_dict[esrs_item["topic"].lower()].extend(
                                find_text_dict(esrs_item_req_data, esrs_item_req_label)
                            )

    return categories_dict


if __name__ == "__main__":
    src_file = "/cluster/home/repo/my_llm_experiments/esrs_data_collection/srn_data_collector/parse_esrs_requirements/esrs_requirement_main.json"
    parse_data(src_file)
