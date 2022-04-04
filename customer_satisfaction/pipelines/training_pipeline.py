from zenml.pipelines import pipeline
from zenml.integrations.constants import MLFLOW


@pipeline(enable_cache=False, required_integrations=[MLFLOW])
def train_pipeline(ingest_data, clean_data, model_train, evaluation):
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        model: CatBoostRegressor
        r2_score: float
        rmse: float
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    lgbm_model = model_train(
        x_train, x_test, y_train, y_test
    )
    r2_score, rmse = evaluation(lgbm_model, x_test, y_test)
    return r2_score, rmse
