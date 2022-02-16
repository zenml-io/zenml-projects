import os
import sys

from steps.post_processor import data_post_processor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

from datetime import date, datetime, timedelta
from zenml.pipelines import Schedule

from steps.splitter import date_based_splitter, SplitConfig
from steps.analyzer import analyze_drift
from steps.discord_bot import discord_alert, discord_post_prediction
from steps.encoder import data_encoder
from steps.evaluator import tester
from steps.feature_engineer import feature_engineer
from steps.importer import game_data_importer, fake_data_importer
from steps.profiler import evidently_drift_detector
from steps.splitter import (
    sklearn_splitter,
    SklearnSplitterConfig,
    reference_data_splitter,
    TrainingSplitConfig,
)
from steps.trainer import random_forest_trainer
from steps.encoder import encode_columns_and_clean
from steps.importer import import_season_schedule, SeasonScheduleConfig
from steps.model_picker import model_picker
from steps.predictor import predictor
from steps.splitter import get_coming_week_data, TimeWindowConfig


from pipelines.data_analysis_pipeline import data_analysis_pipeline
from pipelines.training_pipeline import training_pipeline
from pipelines.prediction_pipeline import inference_pipeline


last_week = date.today() - timedelta(days=365)
ONE_WEEK_AGO = last_week.strftime("%Y-%m-%d")
CURRY_FROM_DOWNTOWN = "2016-02-27"


def run_analysis():
    """Create an analysis pipeline run."""
    # Initialize the pipeline
    eda_pipeline = data_analysis_pipeline(
        importer=game_data_importer(),
        drift_splitter=date_based_splitter(
            SplitConfig(date_split=CURRY_FROM_DOWNTOWN, columns=["FG3M"])
        ),
        drift_detector=evidently_drift_detector,
        drift_analyzer=analyze_drift(),
    )

    eda_pipeline.run()



def run_training():
    """Create a training pipeline run."""
    # Initialize the pipeline
    train_pipe = training_pipeline(
        importer=fake_data_importer(),
        # Train Model
        feature_engineer=feature_engineer(),
        encoder=data_encoder(),
        ml_splitter=sklearn_splitter(
            SklearnSplitterConfig(
                ratios={"train": 0.6, "test": 0.2, "validation": 0.2}
            )
        ),
        trainer=random_forest_trainer(),
        tester=tester(),
        # Drift detection
        drift_splitter=reference_data_splitter(
            TrainingSplitConfig(
                new_data_split_date=ONE_WEEK_AGO,
                start_reference_time_frame=CURRY_FROM_DOWNTOWN,
                end_reference_time_frame="2019-02-27",
                columns=["FG3M"],
            )
        ),
        drift_detector=evidently_drift_detector,
        drift_alert=discord_alert(),
    )

    train_pipe.run(
        schedule=Schedule(start_time=datetime.now(), interval_second=600)
    )


def run_inference():
    """Create an inference pipeline run."""
    # Initialize the pipeline
    inference_pipe = inference_pipeline(
        importer=import_season_schedule(
            SeasonScheduleConfig(current_season="2021-22")
        ),
        preprocessor=encode_columns_and_clean(),
        extract_next_week=get_coming_week_data(
            TimeWindowConfig(time_window=7)
        ),
        model_picker=model_picker(),
        predictor=predictor(),
        post_processor=data_post_processor(),
        prediction_poster=discord_post_prediction(),
    )

    inference_pipe.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pipeline", type=str, choices=["drift", "train", "predict"]
    )
    args = parser.parse_args()

    if args.pipeline == "drift":
        run_analysis()
    elif args.pipeline == "train":
        run_training()
    elif args.pipeline == "predict":
        run_inference()
