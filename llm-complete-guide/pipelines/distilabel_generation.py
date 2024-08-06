from datetime import datetime

from steps.distilabel_generate_queries import generate_synthetic_queries
from steps.hf_dataset_loader import load_hf_dataset
from steps.push_to_argilla import push_to_argilla
from steps.push_to_hf import push_to_hf
from zenml import Model, pipeline

model_definition = Model(
    name="finetuned-zenml-docs-embeddings",
    license="Apache",
    description="A fine-tuned embeddings model for ZenML documentation. Used for RAG retrieval.",
    use_cases="RAG retrieval",
    audience="ZenML users",
    tags=[
        "rag",
        "embeddings",
        "finetuning",
        "llm",
        "internal",
        "synthetic-data",
    ],
    limitations="Only works for ZenML documentation. Not generalizable to other domains. Entirely build with synthetic data. The data is also quite noisy on account of how the chunks were split.",
    trade_offs="Focused on a specific RAG retrieval use case. Not generalizable to other domains.",
    ethics="The data is entirely synthetic.",
    version=f"experiment-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
)


@pipeline(model=model_definition)
def generate_synthetic_data():
    train_dataset, test_dataset = load_hf_dataset()
    train_with_queries, test_with_queries = generate_synthetic_queries(
        train_dataset, test_dataset
    )
    push_to_hf(train_with_queries, test_with_queries)
    push_to_argilla(train_with_queries, test_with_queries)


if __name__ == "__main__":
    generate_synthetic_data()
