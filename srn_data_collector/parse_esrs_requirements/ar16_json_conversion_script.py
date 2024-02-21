import json


def process_data(data):
    processed_data = {}

    for item in data:
        key = item["0"].lower().replace(" ", "_")
        values = {}

        if "1" in item and "2" in item and "3" in item:
            values["label"] = item["1"]
            values["sub_topics"] = item["2"].split(" ·") if item["2"] else []
            values["sub_sub_topics"] = item["3"].split(" ·") if item["3"] else []

        processed_data[key] = values

    return processed_data


def main():
    with open("processed_jsons/ar_16_table_edited.json", "r") as file:
        raw_data = json.load(file)

    processed_data = process_data(raw_data)

    with open("ar_16_table_edited_kv_fmt.json", "w") as output_file:
        json.dump(processed_data, output_file, indent=2)


if __name__ == "__main__":
    main()
