import os
from typing import Dict, List

from collect_api_data import *
from data_model import *
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm


def collect_companies_data(session):
    companies = get_srn_companies()
    for company in tqdm(companies, desc="Processing companies"):
        company_fin_idx_ids = company.pop("indices")
        indices = []
        for company_fin_idx_id in company_fin_idx_ids:
            fin_idx_data = get_financial_index(company_fin_idx_id)
            indices_model = IndicesModel(fin_idx_id=None, **fin_idx_data)
            indices.append(
                Indices(
                    fin_idx_id=indices_model.id,
                    name=indices_model.name,
                    name_short=indices_model.name_short,
                    country=indices_model.country,
                    coverage=indices_model.coverage,
                    updated=indices_model.updated,
                    href=indices_model.href,
                )
            )

        company["indices"] = indices
        company_details_model = CompanyDetailsModel(**company)
        company_details_table = CompanyDetailsTable(
            id=company_details_model.id,
            name=company_details_model.name,
            isin=company_details_model.isin,
            country=company_details_model.country,
            sector=company_details_model.sector,
            href=company_details_model.href,
            href_logo=company_details_model.href_logo,
            company_type=company_details_model.company_type,
            indices=indices,
        )
        session.add(company_details_table)

    # Commit the changes to the database
    session.commit()


def get_custom_revisions(revision_orig):
    revisions_modified = []
    verified_revision = revision_orig.pop("verified_revision")
    all_revisions = revision_orig.pop("revisions")
    verified_at = revision_orig.pop("verified_at")
    for revision in all_revisions:
        revision_obj = {}
        if verified_revision and (revision["id"] == verified_revision["id"]):
            # For all verified revisions
            revision_obj.update({"is_verified": True})
            revision_obj.update({"verified_at": verified_at})
        else:
            # For all unverified revisions
            revision_obj.update({"is_verified": False})
            revision_obj.update({"verified_at": None})
        revision_obj.update(revision_orig)
        revision_obj.update(revision)

        revisions_modified.append(revision_obj)

    return revisions_modified


def collect_annotations(session):
    companies = session.query(CompanyDetailsTable).all()
    for company in tqdm(companies, desc="Retrieving companies annotations"):
        revisions_all: List[Dict] = get_values_with_revisions(company.id)
        for revision in revisions_all:
            modified_revisions = get_custom_revisions(revision)
            for modified_revision in modified_revisions:
                values_with_revision_model = ValuesWithRevisionsModel(**modified_revision)
                values_with_revisions_table = ValuesWithRevisions(
                    id=values_with_revision_model.id,
                    company_id=values_with_revision_model.company_id,
                    compliance_item_id=values_with_revision_model.compliance_item_id,
                    year=values_with_revision_model.year,
                    verified_at=values_with_revision_model.verified_at,
                    company_value_id=values_with_revision_model.company_value_id,
                    created_at=values_with_revision_model.created_at,
                    review_comment=values_with_revision_model.review_comment,
                    flagged_for_review=values_with_revision_model.flagged_for_review,
                    exists=values_with_revision_model.exists,
                    value=values_with_revision_model.value,
                    own_calculation=values_with_revision_model.own_calculation,
                    document_id=values_with_revision_model.document_id,
                    document_ref=values_with_revision_model.document_ref,
                    material=values_with_revision_model.material,
                    material_reason=values_with_revision_model.material_reason,
                    is_verified=values_with_revision_model.is_verified,
                )
                session.add(values_with_revisions_table)

    # Commit the changes to the database
    session.commit()


def get_storage_path(doc_data_model, base_path):
    file_path = os.path.join(base_path, doc_data_model.company_id, doc_data_model.year, doc_data_model.type)
    file_name_abs = os.path.join(file_path, f"{doc_data_model.id}{os.path.splitext(doc_data_model.href)[-1]}")
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)

    if os.path.isfile(file_name_abs):
        # If the file exists that means the pdf is downloaded already
        # download_document(file_name_abs, doc_data_model.href)
        return file_name_abs


def collect_documents_metadata(session, local_storage_path):
    revisions = session.query(ValuesWithRevisions).all()
    for revision in tqdm(revisions, desc="Retrieving document metadata for each revision"):
        document_data = get_srn_documents(revision.document_id)
        if "detail" not in document_data:
            document_instances_model = DocumentInstancesModel(**document_data)

            # Check if the document exists already
            if not session.query(DocumentInstances).filter_by(id=document_instances_model.id).all():
                wz_storage_location = get_storage_path(document_instances_model, base_path=local_storage_path)
                if wz_storage_location:
                    document_instances_table = DocumentInstances(
                        id=document_instances_model.id,
                        name=document_instances_model.name,
                        href=document_instances_model.href,
                        type=document_instances_model.type,
                        year=document_instances_model.year,
                        company_id=document_instances_model.company_id,
                        created_at=document_instances_model.created_at,
                        created_by_info=document_instances_model.created_by_info,
                        wz_storage_location=wz_storage_location,
                    )
                    session.add(document_instances_table)
                    # Commit the changes to the database
                    session.commit()


def collect_compliance_item_data(session):
    revisions = session.query(ValuesWithRevisions).all()
    for revision in tqdm(revisions, desc="Retrieving compliance item data for each annotation"):
        compliance_item_data = get_compliance_item_instance(revision.compliance_item_id)
        compliance_item_model = ComplianceItemsModel(**compliance_item_data)
        if not session.query(ComplianceItems).filter_by(id=compliance_item_model.id).all():
            compliance_items_table = ComplianceItems(
                id=compliance_item_model.id,
                reporting_requirement_id=compliance_item_model.reporting_requirement_id,
                name=compliance_item_model.name,
                status=compliance_item_model.status,
            )
            session.add(compliance_items_table)

            # Commit the changes to the database
            session.commit()


def collect_reporting_item_data(session):
    compliance_items = session.query(ComplianceItems).all()
    existing_rpt_req = session.query(ReportingRequirements).all()
    existing_rpt_req_list = [doc.id for doc in existing_rpt_req]
    for compliance_item in tqdm(compliance_items, desc="Retrieving reporting items for corresponding comliance item"):
        reporting_req_data = get_reporting_requirement_instance(compliance_item.reporting_requirement_id)
        rpt_requirement_model = ReportingRequirementsModel(**reporting_req_data)
        if not session.query(ReportingRequirements).filter_by(id=rpt_requirement_model.id).all():
            rpt_req_table = ReportingRequirements(
                id=rpt_requirement_model.id,
                name=rpt_requirement_model.name,
                topic1=rpt_requirement_model.topic1,
                topic2=rpt_requirement_model.topic2,
                topic3=rpt_requirement_model.topic3,
                topic4=rpt_requirement_model.topic4,
                req_unit=rpt_requirement_model.req_unit,
                req_time=rpt_requirement_model.req_time,
                req_descriptor=rpt_requirement_model.req_descriptor,
                sorting_id=rpt_requirement_model.sorting_id,
            )
            session.add(rpt_req_table)

            # Commit the changes to the database
            session.commit()


def main():
    # Create SQLite database
    engine = create_engine("sqlite:///main.db")

    # Build all the tables
    Base.metadata.create_all(bind=engine)

    # Creates session
    session = Session(engine)

    # Update companies data
    # collect_companies_data(session)

    # Update annotations data
    # collect_annotations(session)

    # collect documents metadata
    collect_documents_metadata(session, local_storage_path="/cluster/home/srn_storage")

    # collect compliance item data
    collect_compliance_item_data(session)

    # collect reporting item data
    collect_reporting_item_data(session)

    session.close()


if __name__ == "__main__":
    main()
