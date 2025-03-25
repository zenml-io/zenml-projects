from pydantic import BaseModel


class DocumentSections(BaseModel):
    management_discussion: str
    risk_factors: str
    financial_statements: str

class DocumentMetadata(BaseModel):
    company_name: str
    sector: str
    fiscal_year: str