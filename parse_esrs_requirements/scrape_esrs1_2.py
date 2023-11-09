import json


def get_toc():
    toc = {"toc_esrs1": {}, "toc_esrs2": {}}
    file_names = ["processed_jsons/toc_esrs1.json", "processed_jsons/toc_esrs2.json"]
    for idx, file in enumerate(file_names, start=1):
        with open(file, "r") as f:
            toc[f"toc_esrs{idx}"] = json.load(f)
    return toc


def exit_condition():
    pass


def entry_condition():
    pass


def get_section(elem_list, cur_idx):
    li_para = ""
    return li_para, cur_idx


def get_cross_cutting_std(elem_list, cur_idx, current_esrs_tag):
    exit_markers = ["ESRS 2", "ESRS E1"]
    section_headers = ["chapters", "appendix_sections"]
    sub_req = {}
    text_content = elem_list[cur_idx].text_content().strip()
    # Get the toc list of dict : {"Section name": "section description"}
    toc = get_toc()
    last_idx = cur_idx
    for current_esrs in toc:
        current_esrs = toc.pop(current_esrs)
        for section_header in section_headers:
            for chapter_id, chapter_dict in current_esrs[section_header].items():
                print(chapter_id, chapter_dict)
                data, last_idx = get_section(elem_list, cur_idx)

    return sub_req, cur_idx
