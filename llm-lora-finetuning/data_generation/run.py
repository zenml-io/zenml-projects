from pipelines.data_generation import data_generation_pipeline

from zenml.logger import get_logger

logger = get_logger(__name__)



def main(
):
    """Main entry point for the pipeline execution."""

    data_generation_pipeline()
    
if __name__ == "__main__":
    main()
