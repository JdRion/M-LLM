import argparse
import os
import re
from datetime import datetime, timedelta
import torch
import transformers
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, load_peft_weights
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb
from omegaconf import OmegaConf
from task_vector import TaskVector
from datamodule import *
from utils import *

time_ = datetime.now() + timedelta(hours=9)
time_now = time_.strftime("%m%d%H%M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"/home/elicer/M-LLM/config/{args.config}.yaml")

    login()
    wandb.login()
    wandb.init(
        project=cfg.wandb.wandb_project,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    ds = preprare_train_dataset(cfg.path, tokenizer)

    # for i in range(len(ds)):
    #     peft_model, tokenizer = peft_model_pipeline(cfg.model.model_name)

    #     args = TrainingArguments(
    #     num_train_epochs = cfg.train.epoch,
    #     per_device_train_batch_size=cfg.train.batch_size,
    #     gradient_accumulation_steps=cfg.train.gradient_accumulation,
    #     learning_rate=cfg.optimizer.learning_rate,
    #     fp16=True,
    #     logging_steps=cfg.train.logging_step,
    #     output_dir="outputs",
    #     optim=cfg.optimizer.optimizer_name,
    #     lr_scheduler_type=cfg.optimizer.scheduler_name,
    #     report_to="wandb",
    #     run_name=f"{cfg.model.saved_name}_{cfg.huggingface[i]}_{time_now}"
    #     )

    #     trainer = Trainer(
    #         model=peft_model,
    #         train_dataset=ds[i],
    #         args=args,
    #         data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    #     )
    #     peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    #     trainer.train()
    #     wandb.finish()
    #     peft_model.push_to_hub(f"{cfg.huggingface[i]}")

    task_vectors = [
        TaskVector(cfg.model.model_name, peft_id, lora=True)
        for peft_id in cfg.huggingface
    ]

    task_vector_sum = sum(task_vectors)
    peft_model, tokenizer = task_vector_sum.apply_to_lora(cfg.model.model_name)

    # 각 데이터셋에서 10개의 행을 추출
    sample_from_ds0 = ds[0].select(range(10))
    sample_from_ds1 = ds[1].select(range(10))
    sample_from_ds2 = ds[2].select(range(10))

    # 추출된 데이터를 하나의 데이터셋으로 병합
    merged_dataset = concatenate_datasets([sample_from_ds0, sample_from_ds1, sample_from_ds2])

    args = TrainingArguments(
    num_train_epochs = cfg.train.epoch,
    per_device_train_batch_size=1,
    learning_rate=cfg.optimizer.learning_rate,
    fp16=True,
    output_dir="outputs",
    optim=cfg.optimizer.optimizer_name,
    report_to="wandb",
    run_name=f"{cfg.model.saved_name}_{time_now}"
    )

    trainer = Trainer(
        model=peft_model,
        train_dataset=merged_dataset,
        args=args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    wandb.finish()
    peft_model.push_to_hub(f"JD97/TaskVector_LLM")
