import argparse

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from typing_extensions import Annotated
from zenml import pipeline, step, log_metadata
from zenml.integrations.huggingface.materializers.huggingface_datasets_materializer import (
    HFDatasetMaterializer
)


@step(output_materializers=HFDatasetMaterializer)
def prepare_data(
    base_model_id: str,
    dataset_name: str,
    dataset_size: int,
    max_length: int,
) -> Annotated[Dataset, "tokenized_dataset"]:
    """
    Prepare and tokenize the dataset for fine-tuning.

    This step loads a specified dataset, tokenizes it with a given base model's
    tokenizer, and prepares it for training by formatting the input as
    question-answer prompts.

    Args:
        base_model_id (str): Identifier of the base model to use for
            tokenization.
        dataset_name (str): Name of the dataset to load from Hugging Face
            datasets.
        dataset_size (int): Number of samples to use from the dataset.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        Annotated[Dataset, "tokenized_dataset"]: Tokenized dataset ready for
            training.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(dataset_name, split=f"train[:{dataset_size}]")

    def tokenize_function(example):
        """
        Tokenize a single example by formatting it as a question-answer prompt.

        Args:
            example (dict): A single dataset example.

        Returns:
            dict: Tokenized input with input_ids, attention_mask, etc.
        """
        prompt = f"Question: {example['question']}\n" \
                 f"Answer: {example['answers']['text'][0]}"
        return tokenizer(prompt, truncation=True, padding="max_length",
                         max_length=max_length)

    tokenized_data = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names
    )
    log_metadata(
        metadata={
            "dataset_size": len(tokenized_data),
            "max_length": max_length
        },
        infer_model=True,
    )
    return tokenized_data


@step
def finetune(
    base_model_id: str,
    tokenized_dataset: Dataset,
    num_train_epochs: int,
    per_device_train_batch_size: int
) -> None:
    """
    Fine-tune a pre-trained language model on the prepared dataset.

    This step loads the base model, sets up training arguments, and performs
    fine-tuning using the Hugging Face Trainer.

    Args:
        base_model_id (str): Identifier of the base model to fine-tune.
        tokenized_dataset (Dataset): Tokenized dataset prepared for training.
        num_train_epochs (int): Number of training epochs.
        per_device_train_batch_size (int): Batch size per device during training.
    """
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float32,  # Changed from float16 to float32
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                      mlm=False),
    )

    train_result = trainer.train()
    log_metadata(
        metadata={
            "metrics": {"train_loss": train_result.metrics.get("train_loss")}
        },
        infer_model=True,
    )
    trainer.save_model("finetuned_model")


@pipeline
def llm_finetune_pipeline(base_model_id: str):
    """
    ZenML pipeline for fine-tuning a language model.

    This pipeline orchestrates the data preparation and fine-tuning steps
    for a language model on a specified dataset.

    Args:
        base_model_id (str): Identifier of the base model to fine-tune.
    """
    tokenized_dataset = prepare_data(base_model_id)
    finetune(base_model_id, tokenized_dataset)


if __name__ == "__main__":
    """
    Entry point for the script that allows configuration via command-line argument.

    Expects a YAML configuration file path to be provided.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML config file'
    )
    args = parser.parse_args()
    llm_finetune_pipeline.with_options(config_path=args.config)()
