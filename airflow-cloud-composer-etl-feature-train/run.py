# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from typing import Dict, Tuple, Annotated
from datetime import datetime, timezone
import pandas as pd
import xgboost as xgb
from zenml import pipeline, step, Model, log_model_metadata, get_step_context
from zenml.io import fileio
from google.cloud import bigquery

# Set up the Model


def ensure_tmp_dir():
    if not os.path.exists("tmp"):
        os.makedirs("tmp")


ensure_tmp_dir()


def get_current_timestamp():
    return datetime.now(timezone.utc).isoformat()


# Data handling steps


@step
def extract_data_local() -> pd.DataFrame:
    return pd.read_csv("data.csv")


@step
def extract_data_cloud() -> pd.DataFrame:
    with fileio.open("gs://your-bucket/data.csv", "r") as f:
        return pd.read_csv(f)


@step
def transform_identity(df: pd.DataFrame) -> pd.DataFrame:
    df["processed"] = 1
    df["load_timestamp"] = get_current_timestamp()
    return df


@step
def load_data_local(df: pd.DataFrame) -> None:
    df.to_csv("tmp/ecb_raw_data.csv", index=False)


@step
def load_data_cloud(df: pd.DataFrame) -> None:
    client = bigquery.Client()
    table_id = "your-project.your_dataset.ecb_raw_data"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()


@step
def load_latest_data_local() -> pd.DataFrame:
    df = pd.read_csv("tmp/ecb_raw_data.csv")
    return df[df["load_timestamp"] == df["load_timestamp"].max()]


@step
def load_latest_data_cloud() -> pd.DataFrame:
    client = bigquery.Client()
    query = """
    SELECT * FROM `your-project.your_dataset.ecb_raw_data`
    WHERE load_timestamp = (SELECT MAX(load_timestamp) FROM `your-project.your_dataset.ecb_raw_data`)
    """
    return client.query(query).to_dataframe()


@step
def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    df["augmented_rate"] = (
        df[
            "Main refinancing operations - Minimum bid rate/fixed rate (date of changes) - Level (FM.D.U2.EUR.4F.KR.MRR_RT.LEV)"
        ]
        * 2
    )
    df["rate_diff"] = (
        df[
            "Marginal lending facility - date of changes (raw data) - Level (FM.D.U2.EUR.4F.KR.MLFR.LEV)"
        ]
        - df[
            "Deposit facility - date of changes (raw data) - Level (FM.D.U2.EUR.4F.KR.DFR.LEV)"
        ]
    )
    df["augment_timestamp"] = get_current_timestamp()
    return df


@step
def save_augmented_data_local(df: pd.DataFrame) -> None:
    df.to_csv("tmp/ecb_augmented_data.csv", index=False)


@step
def save_augmented_data_cloud(df: pd.DataFrame) -> None:
    client = bigquery.Client()
    table_id = "your-project.your_dataset.ecb_augmented_data"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()


@step
def load_latest_augmented_data_local() -> pd.DataFrame:
    df = pd.read_csv("tmp/ecb_augmented_data.csv")
    return df[df["augment_timestamp"] == df["augment_timestamp"].max()]


@step
def load_latest_augmented_data_cloud() -> pd.DataFrame:
    client = bigquery.Client()
    query = """
    SELECT * FROM `your-project.your_dataset.ecb_augmented_data`
    WHERE augment_timestamp = (SELECT MAX(augment_timestamp) FROM `your-project.your_dataset.ecb_augmented_data`)
    """
    return client.query(query).to_dataframe()


# Model training steps


@step
def train_xgboost_model(
    df: pd.DataFrame,
) -> Tuple[Annotated[xgb.Booster, "model"], Annotated[Dict[str, float], "metrics"]]:
    X = df[["augmented_rate", "rate_diff"]]
    y = df[
        "Main refinancing operations - Minimum bid rate/fixed rate (date of changes) - Level (FM.D.U2.EUR.4F.KR.MRR_RT.LEV)"
    ]

    dtrain = xgb.DMatrix(X, label=y)
    params = {"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror"}
    model = xgb.train(params, dtrain, num_boost_round=100)

    predictions = model.predict(dtrain)
    mse = ((predictions - y) ** 2).mean()
    r2 = 1 - (((y - predictions) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    return model, {"mse": float(mse), "r2": float(r2)}


@step
def promote_model(metrics: Dict[str, float]):
    log_model_metadata(
        model_name="ecb_interest_rate_model",
        metadata={
            "metrics": metrics,
            "training_timestamp": get_current_timestamp(),
        },
    )

    if metrics["r2"] > 0.8:
        get_step_context().model.set_stage("production", force=True)


# Pipelines


@pipeline(enable_cache=False, model=Model(name="ecb_interest_rate_model"))
def etl_pipeline(mode: str = "develop"):
    if mode == "develop":
        raw_data = extract_data_local()
        transformed_data = transform_identity(raw_data)
        load_data_local(transformed_data)
    else:
        raw_data = extract_data_cloud()
        transformed_data = transform_identity(raw_data)
        load_data_cloud(transformed_data)


@pipeline(
    enable_cache=False, model=Model(name="ecb_interest_rate_model", version="latest")
)
def feature_engineering_pipeline(mode: str = "develop"):
    if mode == "develop":
        raw_data = load_latest_data_local()
        augmented_data = augment_data(raw_data)
        save_augmented_data_local(augmented_data)
    else:
        raw_data = load_latest_data_cloud()
        augmented_data = augment_data(raw_data)
        save_augmented_data_cloud(augmented_data)


@pipeline(
    enable_cache=False, model=Model(name="ecb_interest_rate_model", version="latest")
)
def model_training_pipeline(mode: str = "develop"):
    if mode == "develop":
        augmented_data = load_latest_augmented_data_local()
    else:
        augmented_data = load_latest_augmented_data_cloud()

    _, metrics = train_xgboost_model(augmented_data)
    promote_model(metrics)


if __name__ == "__main__":
    etl_pipeline()
    feature_engineering_pipeline()
    model_training_pipeline()
