import os
import tempfile
from typing import Annotated, Dict

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim
from zenml import ArtifactConfig, log_model_metadata, step
from zenml.client import Client
from zenml.utils.cuda_utils import cleanup_gpu_memory

# MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# FINETUNED_MODEL_ID = "finetuned-all-MiniLM-L6-v2"
MODEL_ID = "Snowflake/snowflake-arctic-embed-m"
FINETUNED_MODEL_ID = "finetuned-snowflake-arctic-embed-m"

MATRYOSHKA_DIMENSIONS = [384, 256, 128, 64]  # Important: large to small


@step
def prepare_load_data(
    use_argilla_annotations: bool = False,
) -> Annotated[DatasetDict, "full_dataset"]:
    """Load and prepare the dataset for training and evaluation."""
    if use_argilla_annotations:
        zenml_client = Client()
        annotator = zenml_client.active_stack.annotator
        if not annotator:
            raise RuntimeError("No annotator found in the active stack.")
        dataset = annotator.get_labeled_data(
            dataset_name="rag_qa_embedding_questions_0_60_0_distilabel"
        )
    else:
        # Load dataset from the hub
        dataset = load_dataset(
            "zenml/rag_qa_embedding_questions_0_60_0_distilabel", split="train"
        )
        # Add an id column to the dataset
        dataset = dataset.add_column("id", range(len(dataset)))

    # split dataset into a 10% test set
    dataset = dataset.train_test_split(test_size=0.1)

    return dataset


def get_evaluator(
    dataset: DatasetDict, model: SentenceTransformer
) -> SequentialEvaluator:
    """Create a SequentialEvaluator for the given dataset and model."""
    temp_dir = tempfile.TemporaryDirectory()
    train_dataset_path = os.path.join(temp_dir.name, "train_dataset.json")
    test_dataset_path = os.path.join(temp_dir.name, "test_dataset.json")

    # save datasets to disk
    dataset["train"].to_json(train_dataset_path, orient="records")
    dataset["test"].to_json(test_dataset_path, orient="records")

    # load datasets
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
    relevant_docs: Dict[str, list] = {q_id: [q_id] for q_id in queries}

    matryoshka_evaluators = [
        InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
        )
        for dim in MATRYOSHKA_DIMENSIONS
    ]

    return SequentialEvaluator(matryoshka_evaluators)


def evaluate_model(
    dataset: DatasetDict, model: SentenceTransformer
) -> Dict[str, float]:
    """Evaluate the given model on the dataset."""
    cleanup_gpu_memory(force=True)

    evaluator = get_evaluator(dataset, model)
    results = evaluator(model)

    for dim in MATRYOSHKA_DIMENSIONS:
        key = f"dim_{dim}_cosine_ndcg@10"
        print(f"{key}: {results[key]}")

    return results


@step
def evaluate_base_model(
    dataset: DatasetDict,
) -> Annotated[Dict[str, float], "evaluation_results"]:
    """Evaluate the base model on the given dataset."""
    model = SentenceTransformer(
        MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    results = evaluate_model(dataset, model)

    # Convert numpy.float64 values to regular Python floats
    # (needed for serialization)
    base_model_eval = {
        f"dim_{dim}_cosine_ndcg@10": float(
            results[f"dim_{dim}_cosine_ndcg@10"]
        )
        for dim in MATRYOSHKA_DIMENSIONS
    }

    log_model_metadata(
        metadata={"base_model_eval": base_model_eval},
    )

    return results


@step
def evaluate_finetuned_model(
    dataset: DatasetDict,
) -> Annotated[Dict[str, float], "evaluation_results"]:
    """Evaluate the finetuned model on the given dataset."""
    fine_tuned_model = SentenceTransformer(
        f"zenml/{FINETUNED_MODEL_ID}",
        device="cuda" if torch.cuda.is_available() else "cpu",
        revision="main",
    )

    results = evaluate_model(dataset, fine_tuned_model)

    # Convert numpy.float64 values to regular Python floats
    # (needed for serialization)
    # Extract and log only the desired results
    finetuned_model_eval = {
        f"dim_{dim}_cosine_ndcg@10": float(
            results[f"dim_{dim}_cosine_ndcg@10"]
        )
        for dim in MATRYOSHKA_DIMENSIONS
    }

    log_model_metadata(
        metadata={"finetuned_model_eval": finetuned_model_eval},
    )

    return results


@step
def finetune(
    dataset: DatasetDict,
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    optimizer: str = "adamw_torch_fused",
) -> Annotated[
    SentenceTransformer,
    ArtifactConfig(
        is_model_artifact=True,
        name="finetuned-model",
    ),
]:
    """Finetune the model on the given dataset."""
    cleanup_gpu_memory(force=True)

    # load model with SDPA for using Flash Attention 2
    model = SentenceTransformer(
        MODEL_ID,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"zenml/{FINETUNED_MODEL_ID}",
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model, inner_train_loss, matryoshka_dims=MATRYOSHKA_DIMENSIONS
    )

    temp_dir = tempfile.TemporaryDirectory()
    train_dataset_path = os.path.join(temp_dir.name, "train_dataset.json")
    dataset["train"].to_json(train_dataset_path, orient="records")
    train_dataset = load_dataset(
        "json", data_files=train_dataset_path, split="train"
    )

    evaluator = get_evaluator(dataset, model)

    args = SentenceTransformerTrainingArguments(
        output_dir=FINETUNED_MODEL_ID,  # output directory and hugging face model ID
        num_train_epochs=epochs,  # number of epochs
        per_device_train_batch_size=batch_size,  # train batch size
        gradient_accumulation_steps=16,  # for a global batch size of 512
        per_device_eval_batch_size=16,  # evaluation batch size
        warmup_ratio=0.1,  # warmup ratio
        learning_rate=learning_rate,  # learning rate, 2e-5 is a good value
        lr_scheduler_type="cosine",  # use constant learning rate scheduler
        optim=optimizer,  # use fused adamw optimizer
        tf32=True,  # use tf32 precision
        bf16=True,  # use bf16 precision
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="epoch",  # evaluate after each epoch
        save_strategy="epoch",  # save after each epoch
        logging_steps=10,  # log every 10 steps
        save_total_limit=3,  # save only the last 3 models
        load_best_model_at_end=True,  # load the best model when training ends
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
        report_to="none",  # turn off wandb tracking
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,  # training arguments
        train_dataset=train_dataset.select_columns(
            ["positive", "anchor"]
        ),  # training dataset
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()
    trainer.model.push_to_hub(f"zenml/{FINETUNED_MODEL_ID}", exist_ok=True)

    log_model_metadata(
        metadata={
            "training_params": {
                "num_train_epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "base_model": MODEL_ID,
                "matryoshka_dims": MATRYOSHKA_DIMENSIONS,
                "optimizer": optimizer,
            },
            "hardware": {
                "accelerator": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU",
                "accelerator_type": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else ["CPU"],
                "accelerator_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.0f}GB"
                if torch.cuda.is_available()
                else "N/A",
            },
        }
    )

    # handle materialization error with this workaround:
    # Save the model to a temporary file
    temp_dir = tempfile.TemporaryDirectory()
    temp_model_path = os.path.join(temp_dir.name, "model.pt")
    torch.save(trainer.model.state_dict(), temp_model_path)

    # Load the model from the temporary file
    rehydrated_model = SentenceTransformer(MODEL_ID)
    rehydrated_model.load_state_dict(torch.load(temp_model_path))

    # Clean up the temporary directory
    temp_dir.cleanup()

    return rehydrated_model
