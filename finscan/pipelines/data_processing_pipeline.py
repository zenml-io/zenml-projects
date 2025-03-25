from zenml import pipeline
from steps.data_processing import (
    load_data_step,
    store_structured_data_step,
    llm_extract_metadata_step,
    llm_extract_sections_step,
)


@pipeline
def document_processing_pipeline(file_path: str):
    """ZenML pipeline that orchestrates document loading, processing, and storing."""
    df = load_data_step(file_path)
    section_data, metdata_data = (
        llm_extract_sections_step(df),
        llm_extract_metadata_step(df),
    )
    store_structured_data_step(section_data, metdata_data)
