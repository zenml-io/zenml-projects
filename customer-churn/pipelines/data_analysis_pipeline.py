from zenml.integrations.constants import FACETS
from zenml.pipelines import pipeline


@pipeline(enable_cache=False, required_integrations=[FACETS])
def data_analysis_pipeline(
    ingest_data,
    data_splitter,
    visualize_statistics,
    visualize_whylabs_statistics,
):
    """
    Args:
        ingest_data: DataClass
        visualize_statistics: function
    Returns:
        None
    """
    customer_churn_df = ingest_data()
    train, test, train_data_profile, test_data_profile = data_splitter(customer_churn_df)
    visualize_statistics()
    visualize_whylabs_statistics()
    return customer_churn_df, train, test
