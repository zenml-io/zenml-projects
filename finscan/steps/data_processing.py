from zenml import step, ArtifactConfig
import pandas as pd
from openai import OpenAI
from typing import Dict, Any
from typing_extensions import Annotated
from models.models import DocumentMetadata, DocumentSections
import os
from dotenv import load_dotenv


load_dotenv()
client = OpenAI()

class DocumentMetadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@step
def load_data_step(file_path: str) -> pd.DataFrame:
    """
    Loads financial reports data from a local directory.
    """
    return pd.read_csv(file_path)

@step
def llm_extract_sections_step(df: pd.DataFrame) -> Dict[str, DocumentSections]:
    """
    Uses OpenAI structured parser to extract logical sections from the financial reports.
    """
    sections_data = {}
    
    for idx, row in df.iterrows():
        document_text = row['document']
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Extract logical sections from financial reports."},
                {"role": "user", "content": f"{document_text}"}
            ],
            response_format=DocumentSections,
        )
        
        sections_data[str(idx)] = completion.choices[0].message.parsed
    
    return sections_data

@step
def llm_extract_metadata_step(df: pd.DataFrame) -> Dict[str, DocumentMetadata]:
    """
    Extracts financial metrics from the DataFrame.
    """
    extracted_metadata = {}

    for idx, row in df.iterrows():
        metrics = {
            "trade_names": row.get("trade_names", None),
            "developed_technology": row.get("developed_technology", None),
            "customer_relationships": row.get("customer_relationships", None),
            "total_identifiable_intangible_assets": row.get("total_identifiable_intangible_assets", None),
            "cash_and_cash_equivalents": row.get("cash_and_cash_equivalents", None),
            "inventory": row.get("inventory", None),
            "current_assets": row.get("current_assets", None),
            "intangible_assets": row.get("intangible_assets", None),
            "goodwill": row.get("goodwill", None),
            "total_assets": row.get("total_assets", None),
            "accounts_payable": row.get("accounts_payable", None),
            "current_liabilities": row.get("current_liabilities", None),
            "long_term_debt": row.get("long_term_debt", None),
            "total_stockholders_equity": row.get("total_stockholders_equity", None),
            "total_liabilities": row.get("total_liabilities", None),
            "sales": row.get("sales", None),
            "gross_profit": row.get("gross_profit", None),
            "operating_earnings": row.get("operating_earnings", None),
            "net_earnings": row.get("net_earnings", None),
            "accounts_receivable": row.get("accounts_receivable", None),
        }

        extracted_metadata[idx] = DocumentMetadata(**metrics)

    return extracted_metadata


@step
def store_structured_data_step(sections_data: Dict[str, DocumentSections], metadata_data: Dict[Any, Any]) -> Annotated[Dict[str, Any], ArtifactConfig(
            name="structured_dataset",
            tags=["financial", "FindSum"])]:
    """
    Stores structured document chunks in a ZenML artifact store.
    """
    print(metadata_data)
    structured_data = {
    idx: {
        "sections": sections_data[idx],
        "metadata": vars(metadata_data[int(idx)])
    }
    for idx in sections_data
}
    return structured_data
