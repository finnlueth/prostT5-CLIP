import gc
import os
import re
import random
from datetime import datetime
import yaml

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
)
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import (
    AutoConfig,
    AutoTokenizer,
    T5Tokenizer,
    TrainingArguments,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from src.model.configuration_protein_clip import ProtT5CLIPConfig
from src.model.data_collator_multi_input import DataCollatorForProtT5CLIP
from src.model.modeling_protein_clip import ProtT5CLIP
from src.model.trainer_protein_subset import ProteinSampleSubsetTrainer
from src.model.metrics import metrics_factory
import src.model.utils as utils


def setup_environment(train_config):
    """Setup training environment and configs"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    project_name = train_config["project_name"]
    custom_run_name = train_config["custom_run_name"]
    model_name_identifier = (
        project_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + (f"-{custom_run_name}" if custom_run_name else "")
    )

    USE_WANDB = train_config["weights_and_biases"]["enabled"]
    if USE_WANDB:
        import wandb

        run = wandb.init(project=project_name, name=model_name_identifier)

    return model_name_identifier, USE_WANDB


def load_model(train_config, device):
    """Load and configure the model"""
    plm_name = train_config["model"]["protein_encoder_name"]
    llm_name = train_config["model"]["text_encoder_name"]

    plm_config = AutoConfig.from_pretrained(plm_name)
    llm_config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)

    model_config = ProtT5CLIPConfig(
        name_or_path_plm=plm_name,
        name_or_path_llm=llm_name,
        plm_config=plm_config,
        llm_config=llm_config,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
        projection_dim=train_config["model"]["text_projection_dim"],
        logit_scale_init_value=train_config["model"]["logit_scale_init_value"],
        device=device,
    )

    model = ProtT5CLIP(model_config)

    target_modules = []
    modules_to_save = ["protein_projection", "text_projection", "logit_scale"]

    target_modules += TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["t5"]
    modules_to_save += model.loading_info_plm["missing_keys"]

    target_modules += ["qkv_proj"]
    modules_to_save += model.loading_info_llm["missing_keys"]

    lora_config = LoraConfig(
        inference_mode=False,
        r=train_config["lora"]["r"],
        lora_alpha=train_config["lora"]["lora_alpha"],
        lora_dropout=train_config["lora"]["lora_dropout"],
        target_modules=target_modules,
        bias="none",
        modules_to_save=modules_to_save,
        use_rslora=train_config["lora"]["use_rslora"],
        use_dora=train_config["lora"]["use_dora"],
    )

    model = get_peft_model(model, lora_config)
    return model


def prepare_dataset(train_config, tokenizer_plm, tokenizer_llm):
    """Load and prepare the dataset"""
    dataset_path = "../tmp/data/train_val_GO_skimmed"
    dataset_path_processed = "../tmp/data/train_val_GO_skimmed_processed"

    if not os.path.exists(dataset_path_processed):
        dataset = load_from_disk(dataset_path)

        for split in dataset:
            dataset[split] = dataset[split].filter(lambda x: len(x["sequence"]) < 256)
            processed_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in dataset[split]["sequence"]]
            dataset[split] = dataset[split].add_column("sequence_processed", processed_sequences)

            tknz_plm = tokenizer_plm(text=dataset[split]["sequence_processed"], padding=False, truncation=False)
            tknz_llm = tokenizer_llm(text=dataset[split]["GO Sentence"], padding=False, truncation=False)

            dataset[split] = dataset[split].add_column("input_ids_sequence", tknz_plm["input_ids"])
            dataset[split] = dataset[split].add_column("attention_mask_sequence", tknz_plm["attention_mask"])
            dataset[split] = dataset[split].add_column("input_ids_text", tknz_llm["input_ids"])
            dataset[split] = dataset[split].add_column("attention_mask_text", tknz_llm["attention_mask"])

        dataset.save_to_disk(dataset_path_processed)
    else:
        dataset = load_from_disk(dataset_path_processed)

    return dataset


def main():
    with open("../configs/model.yaml", "r") as f:
        train_config = yaml.safe_load(f)

    accelerator = Accelerator()

    set_seed(train_config["seed"])

    model_name_identifier, USE_WANDB = setup_environment(train_config)

    tokenizer_plm = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=train_config["model"]["protein_encoder_name"],
        do_lower_case=False,
        use_fast=True,
        legacy=False,
    )

    tokenizer_llm = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=train_config["model"]["text_encoder_name"],
    )

    model = load_model(train_config, accelerator.device)

    dataset = prepare_dataset(train_config, tokenizer_plm, tokenizer_llm)

    data_collator = DataCollatorForProtT5CLIP(
        tokenizer_plm=tokenizer_plm, tokenizer_llm=tokenizer_llm, padding=True, pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir=f"../tmp/models/checkpoints/{model_name_identifier}",
        run_name=model_name_identifier if USE_WANDB else None,
        report_to="wandb" if USE_WANDB else None,
        learning_rate=train_config["trainer"]["learning_rate"],
        per_device_train_batch_size=train_config["trainer"]["train_batch_size"],
        num_train_epochs=train_config["trainer"]["num_epochs"],
        eval_strategy=train_config["trainer"]["eval_strategy"],
        eval_steps=train_config["trainer"]["eval_steps"],
        per_device_eval_batch_size=train_config["trainer"]["eval_batch_size"],
        eval_on_start=train_config["trainer"]["eval_on_start"],
        batch_eval_metrics=train_config["trainer"]["batch_eval_metrics"],
        save_strategy=train_config["trainer"]["save_strategy"],
        save_steps=train_config["trainer"]["save_steps"],
        save_total_limit=train_config["trainer"]["save_total_limit"],
        remove_unused_columns=train_config["trainer"]["remove_unused_columns"],
        label_names=["input_ids_sequence", "attention_mask_sequence", "input_ids_text", "attention_mask_text"],
        logging_strategy="steps",
        logging_steps=train_config["trainer"]["logging_steps"],
        seed=train_config["seed"],
    )

    trainer = ProteinSampleSubsetTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].select(range(512)),
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=metrics_factory(),
        eval_sample_size=train_config["trainer"]["eval_sample_size"],
    )

    model, trainer = accelerator.prepare(model, trainer)

    trainer.train()
    trainer.evaluate()

    if accelerator.is_main_process:
        model_save_path = f"../tmp/models/{model_name_identifier}"
        model.save_pretrained(model_save_path)

        pd.DataFrame(trainer.state.log_history).to_csv(f"{model_save_path}/training_log.csv", index=False)

        with open(f"{model_save_path}/train_config.yaml", "w") as f:
            yaml.dump(train_config, f, sort_keys=False)

        print("Model, config, and log saved to:", model_save_path)


if __name__ == "__main__":
    main()
