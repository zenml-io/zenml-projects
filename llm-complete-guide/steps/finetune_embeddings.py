import os
import tempfile
from typing import Annotated

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from zenml import step


@step
def prepare_load_data() -> Annotated[DatasetDict, "full_dataset"]:
    # Load dataset from the hub
    dataset = load_dataset(
        "zenml/rag_qa_embedding_questions_0_60_0_distilabel", split="train"
    )

    # Add an id column to the dataset
    dataset = dataset.add_column("id", range(len(dataset)))

    # split dataset into a 10% test set
    dataset = dataset.train_test_split(test_size=0.1)

    return dataset


@step
def evaluate_base_model(dataset: DatasetDict):
    # make temp dir
    temp_dir = tempfile.TemporaryDirectory()
    train_dataset_path = os.path.join(temp_dir.name, "train_dataset.json")
    test_dataset_path = os.path.join(temp_dir.name, "test_dataset.json")

    # save datasets to disk
    dataset["train"].to_json(train_dataset_path, orient="records")
    dataset["test"].to_json(test_dataset_path, orient="records")

    model_id = (
        "sentence-transformers/all-MiniLM-L6-v2"  # Hugging Face model ID
    )
    matryoshka_dimensions = [384, 256, 128, 64]  # Important: large to small

    # Load a model
    model = SentenceTransformer(
        model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # load test dataset
    test_dataset = load_dataset(
        "json", data_files=test_dataset_path, split="train"
    )
    train_dataset = load_dataset(
        "json", data_files=train_dataset_path, split="train"
    )
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Convert the datasets to dictionaries
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)
    queries = dict(
        zip(test_dataset["id"], test_dataset["anchor"])
    )  # Our queries (qid => question)

    # Create a mapping of relevant document (1 in our case) for each query
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [q_id]

    matryoshka_evaluators = []
    # Iterate over the different dimensions
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)

    # Create a sequential evaluator
    evaluator = SequentialEvaluator(matryoshka_evaluators)

    # Evaluate the model
    results = evaluator(model)

    # # COMMENT IN for full results
    print(results)

    # Print the main score
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        print
        print(f"{key}: {results[key]}")


@step
def finetune_embeddings():
    pass


@step
def evaluate_finetuned_model():
    pass


@step
def promote_model():
    pass
