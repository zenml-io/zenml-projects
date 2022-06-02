from google.oauth2 import service_account
from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseStepConfig
from zenml.repository import Repository
import pandas_gbq
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from steps.importer import bigquery_importer
from steps.preparator import preparator
from steps.transformer import transformer
from steps.trainer import trainer
from steps.evaluator import evaluator

@pipeline
def iris_pipeline(
    bigquery_importer,
    preparator,
    transformer,
    trainer,
    evaluator,
):
    """Links all the steps together in a pipeline"""
    data = bigquery_importer() 
    prepared_data = preparator(data=data) #, X_test, y_train, y_test = preparator(data=data)
    X_train, X_test, y_train, y_test = transformer(data=prepared_data) #splitter(data=transformed_data)
    model = trainer(X_train=X_train,y_train=y_train)
    evaluator(X_test=X_test, y_test=y_test, model=model)

pipeline = iris_pipeline(
    bigquery_importer=bigquery_importer(),
    preparator=preparator(),
    transformer=transformer(),
    trainer=trainer(),
    evaluator=evaluator(),
)

pipeline.run()
