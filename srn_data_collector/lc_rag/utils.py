import os
import re
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Dict

from srn_data_collector.annotations_utils.data_model import (
    ComplianceItems,
    ReportingRequirements,
    RptRequirementsMapping,
    StandardsList,
    BlobLvlAnnotations,
    ValuesWithRevisions
)


def initialize_env(env_configs):
    local_inference = env_configs.get("local_inference", {})
    env_variables = local_inference.get("env_variables", {})

    for key, value in env_variables.items():
        os.environ[key] = value


def format_docs(docs):
    return "\n\n".join(f"[{doc_idx}]: {doc.page_content}" for doc_idx, doc in enumerate(docs))


def convert_file_name(original_filename):
    directory, file_with_extension = os.path.split(original_filename)
    filename, extension = os.path.splitext(file_with_extension)
    new_filename = filename + "_processed" + extension
    new_path = os.path.join(directory, new_filename)
    return new_path


def get_doc_name(original_filename):
    directory, file_with_extension = os.path.split(original_filename)
    filename, extension = os.path.splitext(file_with_extension)
    return filename


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["blob_id"] = record.get("blob_id")
    metadata["document_ref"] = record.get("document_ref")
    return metadata


def get_source_from_citem(citem_name:str, annotation_storage_config: Dict):
    Base = declarative_base()
    engine = create_engine(f"sqlite:///{annotation_storage_config['sqlite_db']}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    result = (
                session.query(RptRequirementsMapping.source)
                .join(ReportingRequirements, RptRequirementsMapping.reporting_requirement_id == ReportingRequirements.id)\
                .join(ComplianceItems, ReportingRequirements.id == ComplianceItems.reporting_requirement_id)\
                .join(StandardsList, StandardsList.id == RptRequirementsMapping.standard)\
                .filter(ComplianceItems.name == f'{citem_name}')\
                .filter(StandardsList.family.like("%esrs%"))\
                .all()
        )

    session.close()

    return [i[0] for i in result]


def get_annotations_db(source_section, document_id, annotation_storage_config):
    Base = declarative_base()
    engine = create_engine(f"sqlite:///{annotation_storage_config['sqlite_db']}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    result = (
        session.query(RptRequirementsMapping.source, BlobLvlAnnotations.blob_id, BlobLvlAnnotations.document_ref)
        .join(ValuesWithRevisions, ValuesWithRevisions.document_id == BlobLvlAnnotations.document_id)
        .join(ComplianceItems, ComplianceItems.id == ValuesWithRevisions.compliance_item_id)
        .join(ReportingRequirements, ReportingRequirements.id == ComplianceItems.reporting_requirement_id)
        .join(RptRequirementsMapping, ReportingRequirements.id == RptRequirementsMapping.reporting_requirement_id)
        .join(StandardsList, StandardsList.id == RptRequirementsMapping.standard)
        .filter(StandardsList.family.like("%esrs%"))
        .filter(RptRequirementsMapping.source == f"{source_section}")
        .filter(ValuesWithRevisions.document_id == f"{document_id}")
        .all()
    )


    annotations = []

    for row in result:
        _, blob_id, document_ref = row
        if (blob_id, document_ref) not in annotations:
            annotations.append((blob_id, document_ref))

    session.close()    

    return annotations


def clean_text(text):
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s$|^\s", "", text)
    return text