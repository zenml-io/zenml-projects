from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation
from pipelines.training_pipeline import train_pipeline
from materializer.custom_materializer import cs_materializer


def run_training():
    training = train_pipeline(
        ingest_data(),
        clean_data().with_return_materializers(cs_materializer),
        train_model(),
        evaluation(),
    )

    training.run()


if __name__ == "__main__":
    run_training()

