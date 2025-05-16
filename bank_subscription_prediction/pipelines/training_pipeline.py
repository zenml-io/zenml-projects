from zenml import pipeline
from steps.data_loader import load_data
from steps.data_cleaner import clean_data_step
from steps.data_preprocessor import preprocess_data_step
from steps.data_splitter import split_data_step
from steps.model_trainer import train_xgb_model_with_feature_selection
from steps.model_evaluator import evaluate_model
import logging

# Set up logger
logger = logging.getLogger(__name__)


@pipeline
def bank_subscription_training_pipeline():
    """Pipeline to train a bank subscription prediction model.

    This pipeline doesn't take parameters directly. Instead, it uses
    step parameters from the YAML config file.
    """
    raw_data = load_data()
    cleaned_data = clean_data_step(df=raw_data)
    preprocessed_data = preprocess_data_step(df=cleaned_data)
    X_train, X_test, y_train, y_test = split_data_step(df=preprocessed_data)
    model, feature_selector = train_xgb_model_with_feature_selection(
        X_train=X_train, y_train=y_train
    )
    evaluate_model(
        model=model,
        feature_selector=feature_selector,
        X_test=X_test,
        y_test=y_test,
    )

    logger.info("Bank subscription training pipeline completed.")
