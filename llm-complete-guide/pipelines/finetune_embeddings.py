from steps.finetune_embeddings import evaluate_base_model, prepare_load_data
from zenml import pipeline


@pipeline
def finetune_embeddings():
    data = prepare_load_data()
    evaluate_base_model(data)


if __name__ == "__main__":
    finetune_embeddings()
