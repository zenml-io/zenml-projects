import os
import tempfile
from typing import Annotated

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
from zenml import step

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MATRYOSHKA_DIMENSIONS = [384, 256, 128, 64]  # Important: large to small


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


def get_evaluator(
    dataset: DatasetDict, model: SentenceTransformer
) -> SequentialEvaluator:
    temp_dir = tempfile.TemporaryDirectory()
    train_dataset_path = os.path.join(temp_dir.name, "train_dataset.json")
    test_dataset_path = os.path.join(temp_dir.name, "test_dataset.json")

    # save datasets to disk
    dataset["train"].to_json(train_dataset_path, orient="records")
    dataset["test"].to_json(test_dataset_path, orient="records")

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
    for dim in MATRYOSHKA_DIMENSIONS:
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
    return SequentialEvaluator(matryoshka_evaluators)


@step
def evaluate_base_model(dataset: DatasetDict):
    model = SentenceTransformer(
        MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create a sequential evaluator
    evaluator = get_evaluator(dataset, model)

    # Evaluate the model
    results = evaluator(model)

    # # COMMENT IN for full results
    print(results)

    # Print the main score
    for dim in MATRYOSHKA_DIMENSIONS:
        key = f"dim_{dim}_cosine_ndcg@10"
        print
        print(f"{key}: {results[key]}")


@step
def finetune(
    dataset: DatasetDict,
) -> None:  # TODO: return the model
    # load model with SDPA for using Flash Attention 2
    model = SentenceTransformer(
        MODEL_ID,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="finetuned-matryoshka",
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model, inner_train_loss, matryoshka_dims=MATRYOSHKA_DIMENSIONS
    )

    temp_dir = tempfile.TemporaryDirectory()
    train_dataset_path = os.path.join(temp_dir.name, "train_dataset.json")

    # save datasets to disk
    dataset["train"].to_json(train_dataset_path, orient="records")

    # load train dataset again
    train_dataset = load_dataset(
        "json", data_files=train_dataset_path, split="train"
    )

    evaluator = get_evaluator(dataset, model)

    # define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir="finetuned-matryoshka",  # output directory and hugging face model ID
        num_train_epochs=4,  # number of epochs
        per_device_train_batch_size=32,  # train batch size
        gradient_accumulation_steps=16,  # for a global batch size of 512
        per_device_eval_batch_size=16,  # evaluation batch size
        warmup_ratio=0.1,  # warmup ratio
        learning_rate=2e-5,  # learning rate, 2e-5 is a good value
        lr_scheduler_type="cosine",  # use constant learning rate scheduler
        optim="adamw_torch_fused",  # use fused adamw optimizer
        tf32=True,  # use tf32 precision
        bf16=True,  # use bf16 precision
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="epoch",  # evaluate after each epoch
        save_strategy="epoch",  # save after each epoch
        logging_steps=10,  # log every 10 steps
        save_total_limit=3,  # save only the last 3 models
        load_best_model_at_end=True,  # load the best model when training ends
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
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

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save the best model
    trainer.save_model()

    # push model to hub
    # trainer.model.push_to_hub("finetuned-matryoshka")


@step
def evaluate_finetuned_model():
    pass


@step
def promote_model():
    pass
