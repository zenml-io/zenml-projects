from zenml.integrations.constants import FACETS
from zenml.pipelines import pipeline


@pipeline(enable_cache=False, required_integrations=[FACETS])
def data_analysis_pipeline(
    ingest_data,
    visualize_statistics,
    data_splitter,
    visualize_train_test_statistics,
):
    """
    Args:
        ingest_data: DataClass
        visualize_statistics: function
    Returns:
        None
    """
    customer_churn_df = ingest_data()
    visualize_statistics()
    X_train, X_test, y_train, y_test = data_splitter(customer_churn_df)
    visualize_train_test_statistics()
