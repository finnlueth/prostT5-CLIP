import gc
import os
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
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

import wandb
import src.model.utils as utils

from src.plots.train_plots import plot_training_history
import matplotlib.pyplot as plt

# from accelerate.distributed import DistributedDataParallelKwargs
from src.model.configuration_protein_clip import ProtT5CLIPConfig
from src.model.data_collator_multi_input import DataCollatorForProtT5CLIP
from src.model.metrics import metrics_factory
from src.model.modeling_protein_clip import ProtT5CLIP
from src.model.trainer_protein_subset import ProteinSampleSubsetTrainer


def load_config():
    with open("../configs/model.yaml", "r") as f:
        train_config = yaml.safe_load(f)
    return train_config


def setup_environment(train_config):
    """Setup training environment and configs"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Increase shared memory limit
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(profile="default")

    VERBOSE = train_config["verbose"]
    SEED = train_config["seed"]

    project_name = train_config["project_name"]
    custom_run_name = train_config["custom_run_name"]
    model_name_identifier = (
        project_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + (f"-{custom_run_name}" if custom_run_name else "")
    )

    USE_WANDB = train_config["weights_and_biases"]["enabled"]
    report_to = train_config["weights_and_biases"]["report_to"]

    if USE_WANDB:
        run = wandb.init(project=project_name, name=model_name_identifier)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    print("Model identifier:", model_name_identifier)

    return model_name_identifier, device, report_to, (run if USE_WANDB else None), USE_WANDB, SEED


def load_clip_model(train_config, device):
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
    model.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("Loaded model...")
    utils.check_model_on_cuda(model)
    return model


def apply_lora_to_model(model, train_config):
    target_modules = []
    modules_to_save = ["protein_projection", "text_projection", "logit_scale"]

    target_modules += TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["t5"]
    modules_to_save += model.loading_info_plm["missing_keys"]

    # target_modules += ["qkv_proj"]
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

    print("target_modules:", target_modules)
    print("modules_to_save:", modules_to_save)
    model.print_trainable_parameters()

    return model


def freeze_base_models(model):
    """Freezes the LLM and PLM base models to prevent backpropagation during training"""
    for param in model.model_llm.parameters():
        param.requires_grad = False

    for param in model.model_plm.parameters():
        param.requires_grad = False

    print("Base LLM and PLM models frozen")


def load_tokenizers(train_config):
    tokenizer_plm = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=train_config["model"]["protein_encoder_name"],
        do_lower_case=False,
        use_fast=True,
        legacy=False,
    )

    tokenizer_llm = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=train_config["model"]["text_encoder_name"],
    )
    return tokenizer_plm, tokenizer_llm


def prepare_dataset(train_config, tokenizer_plm, tokenizer_llm):
    """Load and prepare the dataset"""
    dataset_path = "../tmp/data/train_val_GO_skimmed"
    dataset_path_processed = "../tmp/data/train_val_GO_skimmed_processed"

    if not os.path.exists(dataset_path_processed):
        print("Processing dataset...")
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
        print("Loading dataset from disk...")
        dataset = load_from_disk(dataset_path_processed)
    return dataset


def setup_trainer(model, dataset, train_config, model_name_identifier, USE_WANDB, tokenizer_plm, tokenizer_llm):
    """Setup data collator and trainer with training arguments"""
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
        train_dataset=dataset["train"],  # .select(range(512)), #!!!
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=metrics_factory(),
        eval_sample_size=train_config["trainer"]["eval_sample_size"],
    )

    return trainer


def train_model(trainer):
    all_output_keys = [
        "logits_per_protein",
        "logits_per_text",
        "text_embeds",
        "protein_embeds",
        "text_outputs",
        "protein_outputs",
        "proj_protein_embeds",
        "proj_text_embeds",
    ]
    keep_output_keys = ["proj_protein_embeds", "proj_text_embeds"]
    ignore_output_keys = [i for i in all_output_keys if i not in keep_output_keys]

    trainer.train(ignore_keys_for_eval=ignore_output_keys)
    trainer.evaluate(ignore_keys=ignore_output_keys)


def save_model_and_logs(model, trainer, model_name_identifier, train_config):
    model_save_path = f"../tmp/models/{model_name_identifier}"
    model.save_pretrained(model_save_path)

    pd.DataFrame(trainer.state.log_history).to_csv(f"{model_save_path}/training_log.csv", index=False)

    with open(f"{model_save_path}/train_config.yaml", "w") as f:
        yaml.dump(train_config, f, sort_keys=False)

    fig = plot_training_history(log_history=pd.DataFrame(trainer.state.log_history), train_config=train_config)
    fig.savefig(f"{model_save_path}/training_history.png")
    plt.close(fig)

    print("Model, config, and log saved to:", model_save_path)
