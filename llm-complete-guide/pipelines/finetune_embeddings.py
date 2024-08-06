from steps.finetune_embeddings import (
    evaluate_base_model,
    evaluate_finetuned_model,
    finetune,
    prepare_load_data,
)
from zenml import Model, pipeline
from zenml.model.model import ModelStages

model_definition = Model(
    name="finetuned-zenml-docs-embeddings",
    version=ModelStages.LATEST,
)


@pipeline(
    model=model_definition,
)
def finetune_embeddings():
    data = prepare_load_data()
    evaluate_base_model(data)
    finetune(data)
    evaluate_finetuned_model(data, after="finetune")


if __name__ == "__main__":
    finetune_embeddings()
