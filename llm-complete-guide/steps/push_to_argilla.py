import argilla as rg
import torch
from argilla._exceptions import ConflictError
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from zenml import step
from zenml.client import Client


def format_data(batch):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    model = SentenceTransformer(
        model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    def get_embeddings(batch_column):
        # Filter out None values from the batch_column
        batch_column = [item for item in batch_column if item is not None]
        if not batch_column:
            # If all values are None, return an empty list
            return []
        vectors = model.encode(batch_column)
        return [vector.tolist() for vector in vectors]

    # Get indices of non-None values in the anchor column
    valid_indices = [
        i for i, item in enumerate(batch["anchor"]) if item is not None
    ]

    # Filter out None values from the batch columns using valid_indices
    batch["anchor"] = [batch["anchor"][i] for i in valid_indices]
    batch["positive"] = [batch["positive"][i] for i in valid_indices]
    batch["negative"] = [batch["negative"][i] for i in valid_indices]

    batch["anchor-vector"] = get_embeddings(batch["anchor"])
    batch["positive-vector"] = get_embeddings(batch["positive"])
    batch["negative-vector"] = get_embeddings(batch["negative"])
    batch["question-vector"] = batch[
        "positive-vector"
    ]  # Assuming 'positive' is the question

    def get_similarities(a, b):
        similarities = []
        for pos_vec, neg_vec in zip(a, b):
            similarity = cosine_similarity([pos_vec], [neg_vec])[0][0]
            similarities.append(similarity)
        return similarities

    batch["similarity-positive-negative"] = get_similarities(
        batch["positive-vector"], batch["negative-vector"]
    )
    batch["similarity-anchor-positive"] = get_similarities(
        batch["anchor-vector"], batch["positive-vector"]
    )
    batch["similarity-anchor-negative"] = get_similarities(
        batch["anchor-vector"], batch["negative-vector"]
    )
    return batch


@step
def push_to_argilla(train_dataset: Dataset, test_dataset: Dataset) -> None:
    # get secrets for argilla connection
    zenml_client = Client()
    api_key = zenml_client.get_secret("argilla_hf").secret_values["api_key"]
    api_url = zenml_client.get_secret("argilla_hf").secret_values["api_url"]

    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    model = SentenceTransformer(
        model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    client = rg.Argilla(api_url=api_url, api_key=api_key)

    settings = rg.Settings(
        fields=[rg.TextField("anchor")],
        questions=[rg.TextQuestion("positive"), rg.TextQuestion("negative")],
        metadata=[
            rg.TermsMetadataProperty("parent_section"),
            rg.IntegerMetadataProperty("token_count"),
            rg.FloatMetadataProperty("similarity-positive-negative"),
            rg.FloatMetadataProperty("similarity-anchor-positive"),
            rg.FloatMetadataProperty("similarity-anchor-negative"),
        ],
        vectors=[
            rg.VectorField(
                "anchor-vector",
                dimensions=model.get_sentence_embedding_dimension(),
            )
        ],
    )

    ds = rg.Dataset(
        name="rag_qa_embedding_questions_0_60_0_distilabel", settings=settings
    )

    # skip if dataset already exists
    try:
        ds.create()
    except ConflictError:
        pass

    # process original HF dataset
    dataset = train_dataset.map(format_data, batched=True, batch_size=1000)

    # log records to argilla
    records = []
    for idx, entry in enumerate(dataset):
        records.append(
            rg.Record(
                id=idx,
                fields={"anchor": entry["anchor"]},
                suggestions=[
                    rg.Suggestion("positive", value=entry["positive"]),
                    rg.Suggestion("negative", value=entry["negative"]),
                ],
                metadata={
                    "parent_section": entry["parent_section"],
                    "token_count": entry["token_count"],
                    "similarity-positive-negative": entry[
                        "similarity-positive-negative"
                    ],
                    "similarity-anchor-positive": entry[
                        "similarity-anchor-positive"
                    ],
                    "similarity-anchor-negative": entry[
                        "similarity-anchor-negative"
                    ],
                },
                vectors={"question-vector": entry["question-vector"]},
            )
        )
    ds.records.log(records)
