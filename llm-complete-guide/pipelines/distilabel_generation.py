from steps.distilabel_generate_queries import generate_synthetic_queries
from steps.hf_dataset_loader import load_hf_dataset
from steps.push_to_argilla import push_to_argilla
from steps.push_to_hf import push_to_hf
from zenml import pipeline


@pipeline
def generate_synthetic_data():
    train_dataset, test_dataset = load_hf_dataset()
    train_with_queries, test_with_queries = generate_synthetic_queries(
        train_dataset, test_dataset
    )
    push_to_hf(train_with_queries, test_with_queries)
    push_to_argilla(train_with_queries, test_with_queries)


if __name__ == "__main__":
    generate_synthetic_data()
