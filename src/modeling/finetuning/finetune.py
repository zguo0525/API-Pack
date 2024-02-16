import os
import sys
from typing import List
import pyarrow
import fire
import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from datasets import disable_caching
import wandb

wandb.login()
disable_caching()

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

HF_DATASETS_CACHE = '/gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/finetuning/cache'
# Check if the directory exists, and create it if it doesn't
if not os.path.exists(HF_DATASETS_CACHE):
    os.makedirs(HF_DATASETS_CACHE)
# Set the environment variable for the Hugging Face datasets cache
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # state_dict = trainer.model.state_dict()
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        FullStateDictConfig,
        StateDictType,
    )
    model=trainer.model  
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    fsdp: str = "full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap: str = "GPTBigCodeBlock",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out the instruction and input part in loss
    add_eos_token: bool = False,
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"fsdp: {fsdp}\n"
            f"fsdp_transformer_layer_cls_to_wrap: {fsdp_transformer_layer_cls_to_wrap}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "cpu"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:
        total_num_gpus = int(os.environ.get('TOTAL_NUM_GPUS', 1))
        #device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // total_num_gpus
        print("gradient_accumulation_steps", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = False
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map=device_map,
    )

    model = model.to(dtype=torch.float16)
    print("not expanding positional embeddings for this model")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    train_data = load_from_disk(data_path)
    val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            #deepspeed="ds_zero3_cpu_offload.json",
            fsdp=fsdp,
            fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls_to_wrap,
            optim="adafactor",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            report_to=None,
            run_name=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    if resume_from_checkpoint is True:
        resume_from_checkpoint = None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

if __name__ == "__main__":
    fire.Fire(train)