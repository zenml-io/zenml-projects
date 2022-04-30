from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline(enable_cache=True, requirements_file="kubeflow_requirements.txt")
def training_pipeline(
    ingest_data,
    encode_cat_cols,
    handle_imbalanced_data,
    drop_cols,
    data_splitter,
    model_trainer,
    evaluator,
):
    """Pipeline for Training."""
    try:
        customer_churn_df = ingest_data()
        customer_churn_df = encode_cat_cols(customer_churn_df)
        customer_churn_df = handle_imbalanced_data(customer_churn_df)
        customer_churn_df = drop_cols(customer_churn_df)
        train, test = data_splitter(customer_churn_df)
        model = model_trainer(train)
        accuracy = evaluator(model, test)
    except Exception as e:
        logger.error(e)
        raise e
