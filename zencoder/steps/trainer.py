"""
Fine-Tune StarCoder on code/text dataset

Based off Sayak Paul (https://github.com/sayakpaul) and Sourab Mangrulkar (https://github.com/pacman100) codebase: https://github.com/pacman100/DHS-LLM-Workshop/tree/main/
All credit to them for their amazing work!
"""

import functools
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from materializers import HFTrainerMaterializer
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from pydantic import BaseModel
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from typing_extensions import Annotated
from zenml import ArtifactConfig, log_metadata, save_artifact, step
from zenml.client import Client
from zenml.enums import ArtifactType


# this is expensive so we cache it
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        (
            FIM_PREFIX,
            FIM_MIDDLE,
            FIM_SUFFIX,
            FIM_PAD,
        ) = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok]
            for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            None,
            None,
            None,
            None,
        )
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample: List[int],
    np_rng: np.random.RandomState,
    suffix_tok_id: Optional[int],
    prefix_tok_id: Optional[int],
    middle_tok_id: Optional[int],
    pad_tok_id: Optional[int],
    fim_rate: float = 0.5,
    fim_spm_rate: float = 0.5,
    truncate_or_pad: bool = False,
) -> Tuple[List[int], np.random.RandomState]:
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    if np_rng.binomial(1, fim_rate):
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(
            sample[boundaries[0] : boundaries[1]], dtype=np.int64
        )
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            new_length = (
                suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            )
            diff = new_length - len(sample)
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate(
                    [suffix, np.full((-1 * diff), pad_tok_id)]
                )

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't do FIM preproc
        new_sample = sample

    return list(new_sample), np_rng


class Configuration(BaseModel):
    model_path: str = "bigcode/starcoderplus"
    dataset_name: str = "smangrul/hf-stack-v1"
    subset: str = "data"
    split: str = "train"
    size_valid_set: int = 4000
    test_size: float = 0.005
    streaming: bool = False
    shuffle_buffer: int = 5000
    data_column: str = "content"

    seq_length: int = 8192
    max_steps: int = 10000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    eos_token_id: int = 49152

    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    weight_decay: float = 0.05

    local_rank: int = 0
    no_fp16: bool = True
    bf16: bool = False
    no_gradient_checkpointing: bool = True
    seed: int = 0
    num_workers: Optional[int] = None
    output_dir: str = "./checkpoints"
    log_freq: int = 1
    eval_freq: int = 1000
    save_freq: int = 1000

    fim_rate: float = 0
    fim_spm_rate: float = 0

    use_peft_lora: bool = False
    lora_r: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0
    lora_target_modules: Optional[str] = None

    use_flash_attn: bool = False

    use_4bit_qunatization: bool = False
    use_nested_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"

    use_8bit_qunatization: bool = False

    push_to_hub: bool = False
    output_peft_repo_id: str = "zenml/peft-lora-zencoder15B-personal-copilot"


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(
        zip(range(nb_examples), iter(dataset)), total=nb_examples
    ):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for processing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permutations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)[
                "input_ids"
            ]
            all_token_ids = []

            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(
            buffer_size=args.shuffle_buffer, seed=args.seed
        )
    else:
        dataset = dataset.train_test_split(
            test_size=args.test_size, seed=args.seed, shuffle=True
        )
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )
    chars_per_token = chars_token_ratio(
        train_data, tokenizer, args.data_column
    )
    print(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}"
    )
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=args.seed,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=args.seed,
    )

    return train_dataset, valid_dataset


def create_and_prepare_model(args):
    device_map = None
    bnb_config = None

    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not args.no_gradient_checkpointing,
        trust_remote_code=True,
        use_flash_attention_2=args.use_flash_attn,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    if (
        args.use_4bit_qunatization or args.use_8bit_qunatization
    ) and args.use_peft_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.no_gradient_checkpointing
        )

    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )

        if args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model


def run_training(args: Configuration, train_data, val_data, hf_token):
    train_data.start_iteration = 0

    is_deepspeed_peft_enabled = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true"
        and args.use_peft_lora
    )

    save_strategy = "no" if is_deepspeed_peft_enabled else "steps"

    print(f"Starting main loop")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy=save_strategy,
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
        fp16=args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=f"starcoder-copilot",
        push_to_hub=args.push_to_hub,
        include_tokens_per_second=True,
    )

    print("Loading the model")
    model = create_and_prepare_model(args)
    print(model)
    if args.use_peft_lora:
        model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    # post process for faster training when using PEFT + INT4 Quantization
    if args.use_peft_lora:
        for name, module in trainer.model.named_modules():
            if isinstance(module, LoraLayer):
                if args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if any(
                x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]
            ):
                if hasattr(module, "weight"):
                    if args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    print("Training...")
    trainer.train()
    if args.use_peft_lora:
        print("Saving last checkpoint of the model")
        # Save in ZenML as well
        model.save_pretrained(
            os.path.join(args.output_dir, "final_checkpoint/")
        )
        try:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            save_artifact(unwrapped, "final_checkpoint")
        except Exception as e:
            print(str(e))
            print("Skipped saving final checkpoint to ZenML")
            pass

    if is_deepspeed_peft_enabled:
        trainer.accelerator.wait_for_everyone()
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
        unwrapped_model.save_pretrained(
            args.output_dir,
            state_dict=trainer.accelerator.get_state_dict(trainer.deepspeed),
        )
        trainer.accelerator.wait_for_everyone()

    # Save model and tokenizer
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            "FULL_STATE_DICT"
        )

    try:
        if args.push_to_hub:
            commit_info = trainer.push_to_hub()
            log_metadata(
                metadata={"trainer_commit_info": str(commit_info)},
                infer_model=True,
            )
        else:
            trainer.save_model(args.output_dir)
        trainer.accelerator.print(f"Model saved to {args.output_dir}")

        if args.push_to_hub:
            commit_info = trainer.model.push_to_hub(
                repo_id=args.output_peft_repo_id, token=hf_token
            )
            log_metadata(
                metadata={"model_commit_info": str(commit_info)},
                infer_model=True,
            )
    except Exception as e:
        print("Exception while pushing or saving")
        print(str(e))
        pass
    return trainer


@step
def merge_and_push(
    peft_model_id: str, base_model_name: str = "bigcode/starcoder"
):
    secret = Client().get_secret("huggingface_creds")
    hf_token = secret.secret_values["token"]
    login(token=hf_token)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=None,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # model = model.merge_and_unload()
    if not hasattr(model, "hf_device_map"):
        model.cuda()

    peft_model = PeftModel.from_pretrained(
        model, peft_model_id, adapter_name="personal_copilot"
    )
    peft_model.add_weighted_adapter(
        ["personal_copilot"], [0.8], "best_personal_copilot"
    )
    peft_model.set_adapter("best_personal_copilot")
    final_model = peft_model.merge_and_unload()
    final_model.eval()

    model_id_merged = f"{peft_model_id}-merged"
    commit_info = tokenizer.push_to_hub(model_id_merged, token=hf_token)
    log_metadata(
        metadata={"merged_tokenizer_commit_info": str(commit_info)},
        infer_model=True,
    )
    commit_info = final_model.push_to_hub(model_id_merged, token=hf_token)
    log_metadata(
        metadata={"merged_model_commit_info": str(commit_info)},
        infer_model=True,
    )


@step(output_materializers={"trainer_obj": HFTrainerMaterializer})
def trainer(
    args: Configuration,
) -> Tuple[
    Annotated[
        Trainer,
        ArtifactConfig(name="trainer_obj", artifact_type=ArtifactType.MODEL),
    ],
    Annotated[
        GPT2TokenizerFast,
        ArtifactConfig(name="tokenizer_obj", artifact_type=ArtifactType.MODEL),
    ],
    Annotated[str, "peft_model_id"],
    Annotated[ConstantLengthDataset, "train_dataset"],
    Annotated[ConstantLengthDataset, "eval_dataset"],
]:
    set_seed(args.seed)
    hf_token = None
    if args.push_to_hub:
        # Get token from ZenML
        secret = Client().get_secret("huggingface_creds")
        hf_token = secret.secret_values["token"]
        login(token=hf_token)

    print("Loading tokenizer...")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, token=True, trust_remote_code=True
    )

    print("Creating a dataset...")
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    print("Creating a training...")
    trainer_obj = run_training(args, train_dataset, eval_dataset, hf_token)

    return (
        trainer_obj,
        tokenizer,
        args.output_peft_repo_id,
        train_dataset,
        eval_dataset,
    )
