import argparse

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from typing_extensions import Annotated
from zenml import log_model_metadata, pipeline, step
from zenml.integrations.huggingface.materializers.huggingface_datasets_materializer import (
    HFDatasetMaterializer,
)


@step(output_materializers=HFDatasetMaterializer)
def prepare_data(
    base_model_id: str, dataset_name: str, dataset_size: int, max_length: int
) -> Annotated[Dataset, "tokenized_dataset"]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(dataset_name, split=f"train[:{dataset_size}]")

    def tokenize_function(example):
        prompt = f"Question: {example['question']}\nAnswer: {example['answers']['text'][0]}"
        return tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_data = dataset.map(
        tokenize_function, remove_columns=dataset.column_names
    )
    log_model_metadata(
        metadata={
            "dataset_size": len(tokenized_data),
            "max_length": max_length,
        }
    )
    return tokenized_data


@step
def finetune(
    base_model_id: str,
    tokenized_dataset: Dataset,
    num_train_epochs: int,
    per_device_train_batch_size: int,
) -> None:
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
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
    )

    train_result = trainer.train()
    log_model_metadata(
        metadata={
            "metrics": {"train_loss": train_result.metrics.get("train_loss")}
        }
    )
    trainer.save_model("finetuned_model")


@pipeline
def llm_finetune_pipeline(base_model_id: str):
    tokenized_dataset = prepare_data(base_model_id)
    finetune(base_model_id, tokenized_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    llm_finetune_pipeline.with_options(config_path=args.config)()
