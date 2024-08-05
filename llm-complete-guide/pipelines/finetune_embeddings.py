from steps.finetune_embeddings import (
    evaluate_base_model,
    # evaluate_finetuned_model,
    finetune,
    prepare_load_data,
)
from zenml import pipeline


@pipeline
def finetune_embeddings():
    data = prepare_load_data()
    evaluate_base_model(data)
    finetuned_model = finetune(data)
    # evaluate_finetuned_model(finetuned_model)


if __name__ == "__main__":
    finetune_embeddings()
