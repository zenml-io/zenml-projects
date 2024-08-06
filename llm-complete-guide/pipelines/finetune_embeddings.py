from constants import (
    DATASET_NAME_ARGILLA_EMBEDDINGS,
    DATASET_NAME_DISTILABEL_EMBEDDINGS,
    EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS,
    EMBEDDINGS_MODEL_NAME_BASELINE,
    EMBEDDINGS_MODEL_NAME_FINE_TUNED,
    EMBEDDINGS_MODEL_NAME_ZENML,
)
from steps.finetune_embeddings import (
    evaluate_base_model,
    evaluate_finetuned_model,
    finetune,
    prepare_load_data,
)
from zenml import Model, pipeline
from zenml.model.model import ModelStages

model_definition = Model(
    name=EMBEDDINGS_MODEL_NAME_ZENML,
    version=ModelStages.LATEST,
)


@pipeline(
    model=model_definition,
)
def finetune_embeddings():
    data = prepare_load_data(
        dataset_name_argilla=DATASET_NAME_ARGILLA_EMBEDDINGS,
        dataset_name_hf=DATASET_NAME_DISTILABEL_EMBEDDINGS,
    )
    evaluate_base_model(
        dataset=data,
        model_original=EMBEDDINGS_MODEL_NAME_BASELINE,
        matryoshka_dims=EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS
    )
    finetune(
        dataset=data,
        model_orginal=EMBEDDINGS_MODEL_NAME_BASELINE,
        model_fine_tuned=EMBEDDINGS_MODEL_NAME_FINE_TUNED,
        matryoshka_dims=EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS
    )
    evaluate_finetuned_model(
        dataset=data,
        model_fine_tuned=EMBEDDINGS_MODEL_NAME_FINE_TUNED,
        matryoshka_dims=EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS,
        after="finetune"
    )


if __name__ == "__main__":
    finetune_embeddings()
