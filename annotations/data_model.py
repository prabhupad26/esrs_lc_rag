from collect_api_data import get_financial_index, get_srn_companies
from pydantic import BaseModel
from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from tqdm import tqdm

Base = declarative_base()


company_fin_index_association = Table(
    "company_fin_index_association",
    Base.metadata,
    Column("company_id", String, ForeignKey("dim_company_details.id")),
    Column("index_id", String, ForeignKey("Indices.fin_idx_id")),
)


# Tables model definitions
class CompanyDetailsTable(Base):
    __tablename__ = "dim_company_details"

    id = Column(String, primary_key=True)
    name = Column(String)
    isin = Column(String)
    country = Column(String)
    sector = Column(String)
    href = Column(String)
    href_logo = Column(String)
    company_type = Column(String)

    indices = relationship("Indices", secondary=company_fin_index_association, back_populates="companies")

    documents = relationship("DocumentInstances", back_populates="company")


class Indices(Base):
    __tablename__ = "Indices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fin_idx_id = Column(String)
    name = Column(String)
    name_short = Column(String)
    country = Column(String)
    coverage = Column(String)
    updated = Column(String)
    href = Column(String)

    companies = relationship("CompanyDetailsTable", secondary=company_fin_index_association, back_populates="indices")


class DocumentInstances(Base):
    __tablename__ = "dim_document_instances"

    id = Column(String, primary_key=True)
    name = Column(String)
    href = Column(String)
    type = Column(String)
    year = Column(String)
    company_id = Column(String, ForeignKey("dim_company_details.id"))
    created_at = Column(String)
    created_by_info = Column(String)
    wz_storage_location = Column(String)

    company = relationship("CompanyDetailsTable", back_populates="documents")


class ComplianceItems(Base):
    __tablename__ = "dim_compliance_items"

    id = Column(String, primary_key=True)
    reporting_requirement_id = Column(String, ForeignKey("dim_reporting_requirements.id"))
    name = Column(String)
    status = Column(String)

    rpt_requirements = relationship("ReportingRequirements", back_populates="compliance_items")


class ReportingRequirements(Base):
    __tablename__ = "dim_reporting_requirements"

    id = Column(String, primary_key=True)
    name = Column(String)
    topic1 = Column(String)
    topic2 = Column(String)
    topic3 = Column(String)
    topic4 = Column(String)
    req_unit = Column(String)
    req_time = Column(String)
    req_descriptor = Column(String)
    sorting_id = Column(String)

    compliance_items = relationship("ComplianceItems", back_populates="rpt_requirements")


class ValuesWithRevisions(Base):
    __tablename__ = "fact_value_with_revision"
    id = Column(String, primary_key=True)
    company_id = Column(String)  # can be a fk ?
    compliance_item_id = Column(String, ForeignKey("dim_compliance_items.id"))
    year = Column(String)
    verified_at = Column(String)

    company_value_id = Column(String)
    created_at = Column(String)
    review_comment = Column(String)
    flagged_for_review = Column(Boolean)
    exists = Column(Boolean)
    value = Column(String)
    own_calculation = Column(Boolean)
    document_id = Column(String, ForeignKey("dim_document_instances.id"))
    document_ref = Column(String)
    material = Column(Boolean)
    material_reason = Column(String)

    is_verified = Column(Boolean)

    compliance_item = relationship("ComplianceItems")
    document_instance = relationship("DocumentInstances")


# Pydantic model for input validation
class CompanyDetailsModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    id: str
    name: str
    isin: str | None
    country: str | None
    sector: str | None
    href: str | None
    href_logo: str | None
    company_type: str | None
    indices: list[Indices] | None


class IndicesModel(BaseModel):
    id: str
    fin_idx_id: str | None  # keeping it optional such that value are gen automatically
    name: str
    name_short: str | None
    country: str | None
    coverage: str | None
    updated: str | None
    href: str | None


class DocumentInstancesModel(BaseModel):
    id: str
    name: str
    href: str | None
    type: str | None
    year: str | None
    company_id: str | None
    created_at: str | None
    created_by_info: str | None
    # wz_storage_location: str | None


class ComplianceItemsModel(BaseModel):
    id: str
    reporting_requirement_id: str
    name: str
    status: str | None


class ReportingRequirementsModel(BaseModel):
    id: str
    name: str
    topic1: str | None
    topic2: str | None
    topic3: str | None
    topic4: str | None
    req_unit: str | None
    req_time: str | None
    req_descriptor: str | None
    sorting_id: str | None


class ValuesWithRevisionsModel(BaseModel):
    id: str
    company_id: str
    compliance_item_id: str
    year: str | None
    verified_at: str | None

    company_value_id: str | None
    created_at: str | None
    review_comment: str | None
    flagged_for_review: bool | None
    exists: bool | None
    value: str | None
    own_calculation: bool | None
    document_id: str | None
    document_ref: str | None
    material: bool | None
    material_reason: str | None

    is_verified: bool | None


if __name__ == "__main__":
    # Create SQLite database
    engine = create_engine("sqlite:///main.db")
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

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
