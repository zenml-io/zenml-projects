from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline
def training_pipeline(
    ingest_data,
    encode_cat_cols,
    drop_cols,
    data_splitter,
    model_trainer,
    evaluator,
):
    """Pipeline for training."""
    try:
        customer_churn_df = ingest_data()
        customer_churn_df = encode_cat_cols(customer_churn_df)
        customer_churn_df = drop_cols(customer_churn_df)
        train, test = data_splitter(customer_churn_df)
        model = model_trainer(train)
        evaluator(model, test)
    except Exception as e:
        logger.error(e)
        raise e
