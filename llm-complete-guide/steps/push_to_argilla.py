#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.


import argilla as rg
import torch
from constants import (
    DATASET_NAME_ARGILLA,
    EMBEDDINGS_MODEL_ID_BASELINE,
    OPENAI_MODEL_GEN,
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from zenml import step
from zenml.client import Client


def format_data(batch):
    model = SentenceTransformer(
        EMBEDDINGS_MODEL_ID_BASELINE,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    def get_embeddings(batch_column):
        vectors = model.encode(batch_column)
        return [vector.tolist() for vector in vectors]

    batch["anchor-vector"] = get_embeddings(batch["anchor"])
    batch["question-vector"] = get_embeddings(batch["anchor"])
    batch["positive-vector"] = get_embeddings(batch["positive"])
    batch["negative-vector"] = get_embeddings(batch["negative"])

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
    api_key = zenml_client.get_secret("argilla_secrets").secret_values["api_key"]
    api_url = zenml_client.get_secret("argilla_secrets").secret_values["api_url"]
    hf_token = zenml_client.get_secret("argilla_secrets").secret_values["hf_token"]

    model = SentenceTransformer(
        EMBEDDINGS_MODEL_ID_BASELINE,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    client = rg.Argilla(api_url=api_url, api_key=api_key, headers={"Authorization": f"Bearer {hf_token}"})

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
        name=DATASET_NAME_ARGILLA,
        settings=settings,
        workspace=client.workspaces.default,
    )

    # skip if dataset already exists
    #try:
    ds.create()
    # except ConflictError:
    #    ds = client.datasets(DATASET_NAME_ARGILLA)

    # process original HF dataset
    try:
        model = SentenceTransformer(
            EMBEDDINGS_MODEL_ID_BASELINE,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except Exception:
        pass

    dataset = train_dataset.map(format_data, batched=True, batch_size=1000)

    # log records to argilla
    records = []
    for idx, entry in enumerate(dataset):
        records.append(
            rg.Record(
                id=idx,
                fields={"anchor": entry["anchor"]},
                suggestions=[
                    rg.Suggestion(
                        "positive",
                        value=entry["positive"],
                        agent=OPENAI_MODEL_GEN,
                        type="model",
                    ),
                    rg.Suggestion(
                        "negative",
                        value=entry["negative"],
                        agent=OPENAI_MODEL_GEN,
                        type="model",
                    ),
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
                vectors={"anchor-vector": entry["anchor-vector"]},
            )
        )

    ds.records.log(records)
