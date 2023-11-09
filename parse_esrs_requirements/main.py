import json
import re
from typing import Dict, List, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from lxml import html as lh
from pydantic import BaseModel
from rapidfuzz import fuzz
from scrape_esrs1_2 import get_cross_cutting_std


def fuzzy_match_strings(line, tgt_list, threshold=97, partial_match=False):
    match_fn = fuzz.partial_ratio if partial_match else fuzz.ratio
    for tgt in tgt_list:
        if match_fn(line, tgt) >= threshold:
            return True
    return False


def process_ar16_table_df(df):
    df.columns = df.iloc[1, :]
    df = df.iloc[2:, :]
    return df


def get_ar_16_topic(req_text, ar16_table_df, topic_lvl="topic", thresh=80):
    if len(ar16_table_df["Topic"]) < 1:
        return ""
    if topic_lvl == "topic":
        return ar16_table_df["Topic"].values[0]
    elif topic_lvl == "Sub-topic" or topic_lvl == "Sub-sub-topics":
        if not pd.isnull(ar16_table_df[topic_lvl]).any():
            sub_topics = ar16_table_df[topic_lvl].values[0]
            sub_topics = sub_topics.split("·")
            sub_topics = [value.replace("\xa0", " ") for value in sub_topics if value]
            sub_topics = [
                value for value in sub_topics if fuzz.partial_ratio(value.lower(), req_text.lower()) >= thresh
            ]
            return sub_topics
        else:
            return ""
    else:
        raise Exception("Incorrect level of topic")


class ESRSParser(BaseModel):
    data_path: str
    general_disclosure_categories: List[str] = [
        "governance",
        "strategy",
        "impact, risk and opportunity management",
        "metrics and targets",
    ]
    ar16_table_df: pd.DataFrame = process_ar16_table_df(pd.read_json("processed_jsons/ar_16_table_edited.json"))

    class Config:
        arbitrary_types_allowed = True

    cross_cutting_std: List[str] = ["ESRS 1", "ESRS 2"]

    full_esrs_tag_list: List[str] = [
        "ESRS E1",
        "ESRS E2",
        "ESRS E3",
        "ESRS E4",
        "ESRS E5",
        "ESRS S1",
        "ESRS S2",
        "ESRS S3",
        "ESRS S4",
        "ESRS G1",
    ]

    related_to_esrs2: List[str] = [
        "GOV-1",
        "GOV-2",
        "GOV-3",
        "GOV-4",
        "GOV-5",
        "SBM-1",
        "SBM-2",
        "SBM-3",
        "IRO-1",
        "IRO-2",
    ]

    def match_esrs2_rel_categories(self, descr, thresh=97):
        for cat in self.related_to_esrs2:
            if fuzz.partial_ratio(cat, descr) >= thresh:
                return cat
        raise Exception("Didn't match any category")

    @staticmethod
    def get_section_start_idx(marker_label_str: str, elem_list: List, cur_idx: int, count_thresh: int = 4) -> int:
        count: int = 0
        # since each item on the list will be inside the p tag,
        # e.g. [<p> <span> Objective</span></p>, <span> Objective</span>]
        # on the count th occurrence of marker_label_str the section will start
        while count < count_thresh:
            cur_idx += 1
            text_content = elem_list[cur_idx].text_content().strip().lower()
            if text_content == marker_label_str:
                count += 1
        return cur_idx

    @staticmethod
    def is_elem_qualify(text_content, elem, category_type="objective_requirements"):
        # Entry conditions
        if category_type == "objective_requirements":
            return (
                text_content
                and elem.get("class")
                in ["li ListParagraph", "Heading1", "Heading2", "li Heading3", "li Heading4", "li Heading5"]
                and (
                    re.match(r"^(?:[1-9]\d{0,2}\.|\(.\)|[IVX]+\.)\s*", text_content)
                    or re.match(r"^[IVXLCDM]+\.\s*(.*)", text_content)
                )
            )
        elif category_type == "application_requirements":
            return (
                text_content
                and (elem.get("class") in ["li ListParagraph", "li BodyText", "li Heading4"])
                and (
                    re.match(r"^(?:AR\s*|[IVX]+\.\s*|\(.\)|[1-9]\d{0,2}\.)\s*", text_content)
                    or re.match(r"^[IVXLCDM]+\.\s*(.*)", text_content)
                )
            )
        elif category_type == "footnotes":
            return text_content and elem.tag == "dd" and re.match(r"\((\d{1,3}|[a-zA-Z])\)\s*(.*)", text_content)
        else:
            raise Exception(f"Category {category_type} not configured")

    def li_para_exit_condition(self, element, category_type="objective_requirements"):
        # Exit condition
        if category_type == "objective_requirements":
            return (element.get("class") not in ["li ListParagraph", "li Heading3", "Heading1"]) and (
                element.text_content().strip().lower()
                not in [
                    "appendix a: application requirements",
                    "interactions with other esrs",
                    "interaction with other esrs",
                    "disclosure requirements",
                ]
                and element.tag not in ["dl", "dd", "div"]
            )
        elif category_type == "application_requirements":
            return (
                element.get("class") not in ["li ListParagraph", "li BodyText", "Heading4"]
                and element.text_content().strip() not in self.full_esrs_tag_list
                and element.tag not in ["dl", "dd", "div"]
            )
        elif category_type == "footnotes":
            return element.tag not in ["dl", "dd", "div"]
        else:
            raise Exception(f"Category {category_type} not configured")

    def get_li_para(self, element_list, cur_idx, category_type="objective_requirements"):
        # assert element_list[cur_idx].get('class') == 'li ListParagraph', "Not a valid list_paragraph"
        cur_idx += 1
        num_ = ""
        text = ""
        invalid_div_class = ["footnoteRef"]
        while cur_idx < len(element_list) and self.li_para_exit_condition(element_list[cur_idx], category_type):
            div_class = element_list[cur_idx].get("class")
            if div_class in invalid_div_class:
                cur_idx += 1
                continue

            if element_list[cur_idx].get("class") == "num":
                num_ = element_list[cur_idx].text_content().strip().replace(".", "")
            else:
                text += " " + element_list[cur_idx].text_content().strip()
            cur_idx += 1

        return num_, text

    @staticmethod
    def remove_spl_char(inp_str):
        return inp_str.replace("\xa0", " ")

    @staticmethod
    def match_disclosure_req(text):
        patterns = [
            r"^disclosure requirement (\w+-\w+)\s*–\s*(.*)",
            r"^disclosure requirements (\w+-\w+)\s*–\s*(.*)",  # hyphen sign scenario
            r"^disclosure requirement (\w+-\w+)\s*-\s*(.*)",  # minus sign scenario
            r"^(disclosure requirement related to esrs 2)(.*)",
            r"^(disclosure requirements related to esrs 2)(.*)",
        ]
        for pattern in patterns:
            match = re.match(pattern, text.lower())
            if match is not None:
                return match.group(1).upper(), match.group(2)
        return None, None

    def get_all_requirements(self, elem_list, cur_idx, current_esrs_tag):
        prev_num = 0
        prev_anum = "(a)"
        category_label = ""
        collect_footnotes = False

        # starting point, valid values - general_requirements, application_requirements
        current_tag = "general_requirements"

        current_category_tag = "objective_requirements"
        gen_req = {current_tag: {current_category_tag: {}}}
        text_content = elem_list[cur_idx].text_content().strip()

        while cur_idx < len(elem_list) and text_content not in self.full_esrs_tag_list:
            text_content = elem_list[cur_idx].text_content().strip()

            if fuzzy_match_strings(text_content.lower(), self.general_disclosure_categories):
                category_label = text_content.lower()

            if self.remove_spl_char(text_content.lower()) in [
                "interactions with other esrs",
                "interaction with other esrs",
            ]:
                current_tag = text_content.lower().replace(" ", "_")
                gen_req[current_tag] = {f"{current_category_tag}_description": text_content, current_category_tag: {}}

            if self.remove_spl_char(text_content.lower()) == "appendix a: application requirements":
                current_category_tag = "application_requirements"
                current_tag = ""  # reset current tag when appendix A section starts

            if elem_list[cur_idx].get("id") == "footnotes":
                current_tag = "footnotes"
                current_category_tag = "footnotes"
                gen_req[current_tag] = {current_category_tag: {}}

            if self.is_elem_qualify(text_content, elem_list[cur_idx], category_type=current_category_tag) and (
                current_tag or "footnotes" in gen_req
            ):
                num, text = self.get_li_para(elem_list, cur_idx, category_type=current_category_tag)

                if re.match(r"^(?:[1-9]\d{0,2}\.|\(.\)|[IVX]+\.)\s*", num) and "footnotes" not in gen_req:
                    # When there are sub points under the numeric bullet points
                    assert prev_num != 0, "this sub requirement has not parent requirement"
                    if "sub_req" not in gen_req[current_tag][current_category_tag][prev_num]:
                        gen_req[current_tag][current_category_tag][prev_num]["sub_req"] = {}
                    prev_anum = num
                    gen_req[current_tag][current_category_tag][prev_num]["sub_req"][num] = {"text": text}

                elif re.match(r"^[IVXLCDM]+\.\s*(.*)", text_content) and "footnotes" not in gen_req:
                    # When there are sub points under the sub points
                    if "sub_sub_req" not in gen_req[current_tag][current_category_tag][prev_num]["sub_req"][prev_anum]:
                        gen_req[current_tag][current_category_tag][prev_num]["sub_req"][prev_anum]["sub_sub_req"] = {}
                    gen_req[current_tag][current_category_tag][prev_num]["sub_req"][prev_anum]["sub_sub_req"][num] = {
                        "text": text
                    }

                else:
                    # collect the numeric bullet points
                    # TODO : Handling footnotes is not perfect yet
                    prev_num = str(num)
                    gen_req[current_tag][current_category_tag][str(num)] = {"text": text}

            # Match pattern: Disclosure Requirement E1-1 - etc.
            label, description = self.match_disclosure_req(text_content)
            if description:
                prev_num = 0  # reset the bullet point index when a new section is found
                # Exit the sub requirement collection only when there is a new sub requirement or new appendix a ar

                if fuzzy_match_strings(label.lower(), ["disclosure requirement related to esrs 2"]):
                    label = f"{label}_{self.match_esrs2_rel_categories(description.upper())}"
                current_tag = label
                if current_tag not in gen_req:
                    topic_label_df = self.ar16_table_df[self.ar16_table_df["Topical ESRS"] == current_esrs_tag]
                    gen_req[current_tag] = {
                        current_category_tag: {},
                        f"{current_category_tag}_description": description,
                        "category": category_label,
                        "topic": get_ar_16_topic(description, topic_label_df, topic_lvl="topic"),
                        "Sub-topic": get_ar_16_topic(description, topic_label_df, topic_lvl="Sub-topic"),
                        "Sub-sub-topics": get_ar_16_topic(description, topic_label_df, topic_lvl="Sub-sub-topics"),
                        "application_requirements": {},
                    }
                elif f"{current_category_tag}_description" not in gen_req[current_tag]:
                    gen_req[current_tag][f"{current_category_tag}_description"] = description

            cur_idx += 1
        return gen_req, cur_idx

    def get_sub_requirements(self, elem_list: List, cur_idx: int, current_esrs_tag: str) -> Tuple[Dict, int]:
        # This will collect all the sub requirements + general requirement for the given topical std
        # TODO:  collect other sections apart from objective and appendix ones as well

        sub_requirements = {}
        text_content = elem_list[cur_idx]

        while cur_idx < len(elem_list) and text_content not in self.full_esrs_tag_list:
            cur_idx += 1
            text_content = elem_list[cur_idx].text_content().strip()
            if text_content:
                sub_requirements, cur_idx = self.get_all_requirements(elem_list, cur_idx, current_esrs_tag)
                if cur_idx <= len(elem_list):
                    break
                text_content = elem_list[cur_idx].text_content().strip()

        return sub_requirements, cur_idx

    def collect_dr(
        self, elem_list: List, curr_idx: int, current_esrs_tag: str, collect_esrs1_2: bool = False
    ) -> Tuple[Dict, int]:
        if current_esrs_tag in self.cross_cutting_std and collect_esrs1_2:
            curr_idx = self.get_section_start_idx("objective", elem_list, curr_idx)
            disclosure_req, curr_idx = get_cross_cutting_std(elem_list, curr_idx, current_esrs_tag)
        else:
            curr_idx = self.get_section_start_idx("objective", elem_list, curr_idx)  # 3726
            # curr_idx = self.get_section_start_idx("disclosure requirements", elem_list,  # 3818
            #                                       curr_idx, count_thresh=1)
            disclosure_req, curr_idx = self.get_sub_requirements(elem_list, curr_idx, current_esrs_tag)

        return disclosure_req, curr_idx

    def parse(self) -> Dict:
        tree = lh.parse(self.data_path)
        cover_letter, annex_1, annex_2 = tree.xpath('//div[@class="contentWrapper"]')

        for elem in annex_1.xpath(
            f"//style | "
            f"//FootnoteReference | "
            f"//footnoteRef | "
            f"//table | "
            f"//tbody | "
            f"//tr | "
            f"//td | "
            f"//img | "
            f"//map | "
            f"//link | "
            "//textbox-border | "
            f"//s | "
            f'//p[@class="mw-empty-elt"] | '  # remove empty p tags
            f'//span[contains(@class, "shortdescription ")] | '  # remove short descriptions
            f'//span[contains(@class, "thumb ")] |'  # remove thumbnails
            f'//span[@class="toc"] |'  # remove tocs
            f'//span[@role="note"] |'  # remove notes
            f'//table[not(@class="wikitable")] |'  # remove non-wikitables
            f'//ul[contains(@class, "gallery ")] |'  # remove galleries
            f'//span[@class="mw-editsection"] |'  # remove [edit]
            f'//sup[@class="reference"] |'  # remove references, e.g. [43]
            f'//sup[@class="noprint Inline-Template"] |'  # remove general no prints
            f'//sup[@class="plainlinks noexcerpt noprint asof-tag update"] |'  # remove [update]
            f'//sup[@class="noprint Inline-Template Template-Fact"]'  # remove [citation needed]
        ):
            elem.drop_tree()

        requirements = {}

        elem_list = list(annex_1.iter())
        cur_idx = 0

        while cur_idx < len(elem_list):
            elem = elem_list[cur_idx]
            try:
                text_content = elem.text_content().strip()
            except ValueError as ve:
                # this is done for the comments in html file issue
                cur_idx += 1
                continue

            # if text_content in self.full_esrs_tag_list + self.cross_cutting_std:
            if text_content in self.full_esrs_tag_list:
                current_esrs_tag = text_content

                requirements[current_esrs_tag] = {}
                # new ESRS starts here

                dr_dict, cur_idx = self.collect_dr(elem_list, cur_idx + 2, current_esrs_tag)
                # cur_idx above should include next section of the next ESRS section

                # offset by 1 so that next section start in next iteration
                cur_idx -= 1

                # TODO : Handle foot notes separately
                if "footnotes" in dr_dict:
                    requirements["global_footnotes"] = dr_dict.pop("footnotes")

                requirements[current_esrs_tag].update(dr_dict)

            cur_idx += 1

        # TODO: Parse Annex_2 table
        with open(self.data_path, "r") as f:
            html_content = f.read()
        annex_2_soup = BeautifulSoup(html_content, "html.parser")
        _, _, annex_2 = annex_2_soup.find_all("div", class_="contentWrapper")

        acronyms_table_soup, glossary_table_soup = annex_2.find_all("table")

        annex_2_tables_dict = {}
        tables_list = ["acronyms_dict", "glossary_dict"]

        for idx, table_soup in enumerate([acronyms_table_soup, glossary_table_soup]):
            for row in table_soup.find_all("tr"):
                row_data = [cell.text.strip() for cell in row.find_all(["td", "th"])]
                if tables_list[idx] not in annex_2_tables_dict:
                    annex_2_tables_dict[tables_list[idx]] = {}
                annex_2_tables_dict[tables_list[idx]][row_data[0]] = row_data[1]

        requirements["annex_2_tables"] = annex_2_tables_dict

        return requirements


def main():
    data_path = "/home/ppradhan/Documents/my_learnings/llm_tuts/raw_files/" "esrs_stds.html"
    output_path = "output_jsons/esrs_draft.json"
    esrsparser = ESRSParser(data_path=data_path)
    reqs = esrsparser.parse()
    with open(output_path, "w") as f:
        json.dump(reqs, f)


if __name__ == "__main__":
    main()
