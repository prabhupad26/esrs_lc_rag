import json
import os
from typing import Dict, List

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, aliased
from tqdm import tqdm

from srn_data_collector.annotations_utils.data_model import (
    BlobLvlAnnotations,
    CompanyDetailsTable,
    ComplianceItems,
    DocumentInstances,
    ReportingRequirements,
    RptRequirementsMapping,
    StandardsList,
    ValuesWithRevisions,
)


def get_all_files_in_directory(session, storage_location: str) -> List[str]:
    docs_objs = session.query(DocumentInstances).all()
    all_files = [f"{doc_obj.company_id}/{doc_obj.year}/{doc_obj.type}/{doc_obj.id}.json" for doc_obj in docs_objs]
    return [file for file in all_files if os.path.exists(os.path.join(storage_location, file))]


def get_total_pages(doc_path):
    with open(doc_path, "r") as f:
        parsed_doc_data = json.load(f)
    return len(parsed_doc_data)


def get_annotations(session, doc_id_list: List[str], storage_location: str):
    # Define the SQLAlchemy models
    fvwr = aliased(ValuesWithRevisions)
    dci = aliased(ComplianceItems)
    drr = aliased(ReportingRequirements)
    dcd = aliased(CompanyDetailsTable)
    ddi = aliased(DocumentInstances)
    dbla = aliased(BlobLvlAnnotations)
    mrrsl = aliased(RptRequirementsMapping)
    dsl = aliased(StandardsList)

    # Build the SQLAlchemy query
    stmt = (
        select(
            fvwr.id.label("REVISION_ID"),  # 0
            dsl.family.label("STD_FAMILY"),  # 1
            dsl.name.label("STD_NAME"),  # 2
            mrrsl.id.label("SOURCE_ID"),  # 3
            mrrsl.source.label("REQ_SECTION"),  # 4
            dci.name.label("COMPLIANCE_ITEM_NAME"),  # 5
            # fvwr.document_ref.label('DOC_PAGE_NO'),
            fvwr.document_id,  # 6
            dcd.name.label("COMPANY_NAME"),  # 7
            ddi.href.label("DOC_HREF"),  # 8
            fvwr.value.label("ORIGINAL_ANNOTATION_TEXT"),  # 9
            dbla.blob_text.label("PARSED_DOC_BLOB_TEXT"),  # 10
            dbla.blob_class_name,  # 11
            dbla.blob_id.label("BLOB_ID"),  # 12
            # 13
            dbla.document_ref.label("PAGE_NO"),
        )
        .join(dci, dci.id == fvwr.compliance_item_id)
        .join(drr, drr.id == dci.reporting_requirement_id)
        .join(dcd, dcd.id == fvwr.company_id)
        .join(ddi, ddi.id == fvwr.document_id)
        .join(dbla, dbla.revision_id == fvwr.id)
        .join(mrrsl, drr.id == mrrsl.reporting_requirement_id)
        .join(dsl, dsl.id == mrrsl.standard)
    )
    # .where(dsl.family.like('%esrs%'))

    annotations_detailed = session.execute(stmt)

    annotation_counts = {
        os.path.basename(os.path.splitext(doc_id)[0]): {"doc_location": doc_id} for doc_id in doc_id_list
    }

    for annotations in tqdm(annotations_detailed, desc="Geting annotation dict"):
        if not annotation_counts[annotations[6]] or "annotated_page_cnt" not in annotation_counts[annotations[6]]:
            annotation_counts[annotations[6]] = {
                "annotated_page_cnt": {},
                "total_pages": get_total_pages(
                    os.path.join(storage_location, annotation_counts[annotations[6]]["doc_location"])
                ),
            }
        if annotations[13] not in annotation_counts[annotations[6]]["annotated_page_cnt"]:
            annotation_counts[annotations[6]]["annotated_page_cnt"][annotations[13]] = 0
        annotation_counts[annotations[6]]["annotated_page_cnt"][annotations[13]] += 1

    return annotation_counts


def create_plot(annotations: Dict):
    pass


def main():
    db_path = "/cluster/home/repo/my_llm_experiments/esrs_data_collection/srn_data_collector/main.db"

    # Assume you have an SQLAlchemy engine already created
    engine = create_engine(f"sqlite:///{db_path}")

    dataset_path = "/cluster/home/srn_storage"
    session = Session(engine)
    json_doc_list = get_all_files_in_directory(session, dataset_path)
    annotations = get_annotations(session, json_doc_list, dataset_path)

    session.close()


if __name__ == "__main__":
    main()
