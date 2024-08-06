from datetime import datetime

from constants import (
    DATASET_NAME_ARGILLA_EMBEDDINGS,
    DATASET_NAME_DISTILABEL_EMBEDDINGS,
    DATASET_NAME_EMBEDDINGS,
    EMBEDDINGS_MODEL_NAME_VECTOR_SEARCH,
    EMBEDDINGS_MODEL_NAME_ZENML,
    OPENAI_MODEL_EMBEDDINGS,
    OPENAI_MODEL_GEN_KWARGS_EMBEDDINGS,
)
from steps.distilabel_generate_queries import generate_synthetic_queries
from steps.hf_dataset_loader import load_hf_dataset
from steps.push_to_argilla import push_to_argilla
from steps.push_to_hf import push_to_hf
from zenml import Model, pipeline

model_definition = Model(
    name=EMBEDDINGS_MODEL_NAME_ZENML,
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
    version=f"argilla-webinar-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
)


@pipeline(model=model_definition)
def generate_synthetic_data():
    train_dataset, test_dataset = load_hf_dataset(
        dataset_nama=DATASET_NAME_EMBEDDINGS
    )
    train_with_queries, test_with_queries = generate_synthetic_queries(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        dataset_name=DATASET_NAME_EMBEDDINGS,
        model=OPENAI_MODEL_EMBEDDINGS,
        generation_kwargs=OPENAI_MODEL_GEN_KWARGS_EMBEDDINGS
    )
    push_to_hf(
        train_dataset=train_with_queries,
        test_dataset=test_with_queries,
        dataset_name=DATASET_NAME_DISTILABEL_EMBEDDINGS
    )
    push_to_argilla(
        train_dataset=train_with_queries,
        test_dataset=test_with_queries,
        dataset_name=DATASET_NAME_ARGILLA_EMBEDDINGS,
        model_name_embeddings=EMBEDDINGS_MODEL_NAME_VECTOR_SEARCH,
        model_name_generation=OPENAI_MODEL_EMBEDDINGS
    )


if __name__ == "__main__":
    generate_synthetic_data()
