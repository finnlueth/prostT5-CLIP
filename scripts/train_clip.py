import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from src.model.model import ProtT5CLIP
from src.model.data_collator import DataCollatorForProtT5CLIP

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_from_disk

import torch
import re
import pandas as pd
import numpy as np
import gc
from datetime import datetime

from transformers import T5Tokenizer, Trainer, TrainingArguments, AutoConfig, CLIPConfig, PretrainedConfig

from peft import (
    LoraConfig,
    get_peft_model,
)
import wandb

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
USE_WANDB = True


def main():
    if USE_WANDB:
        run = wandb.init(project="protT5-CLIP", name=f"protT5-CLIP-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

    report_to = "wandb" if USE_WANDB else None

    plm_config = AutoConfig.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    llm_config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

    model_config = PretrainedConfig(
        name_or_path_plm="Rostlab/prot_t5_xl_uniref50",
        name_or_path_llm="microsoft/Phi-3.5-mini-instruct",
        plm_config=plm_config,
        llm_config=llm_config,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
        frozen_plm=False,
        frozen_llm=False,
        projection_dim=1024,
        logit_scale_init_value=2.6592,
    )

    model = ProtT5CLIP(model_config)
    model.to(device)
    model.to(torch.bfloat16)

    target_modules = []
    modules_to_save = ["protein_projection", "text_projection"]
    if not model_config.frozen_plm:
        target_modules += ["q", "k", "v", "o"]
        modules_to_save += model.loading_info_plm["missing_keys"]
    if not model_config.frozen_llm:
        target_modules += ["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        modules_to_save += model.loading_info_llm["missing_keys"]

    lora_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        modules_to_save=modules_to_save,
        # use_rslora=True,
        # use_dora=True,
    )

    model = get_peft_model(model, lora_config)
    print("target_modules:", target_modules)
    print("modules_to_save:", modules_to_save)
    model.print_trainable_parameters()

    tokenizer_plm = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=model_config.name_or_path_plm,
        do_lower_case=False,
        use_fast=True,
        legacy=False,
    )

    tokenizer_llm = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_config.name_or_path_llm,
    )

    overwrite = False

    processed_dataset_path = "tmp/data/processed_train_val_GO_FULL"
    if not overwrite and os.path.exists(processed_dataset_path):
        print("Loading processed dataset from disk...")
        dataset = load_from_disk(processed_dataset_path)
    else:
        print("Processing dataset...")
        dataset = load_from_disk("tmp/data/train_val_GO")
        dataset = DatasetDict(
            {
                "train": dataset["train"],  # .select(range(10000)),
                "valid": dataset["test"],  # .select(range(3000))
            }
        )

        for split in dataset:
            dataset[split] = dataset[split].filter(lambda x: len(x["sequence"]) < 256)

            dataset[split] = dataset[split].map(lambda x: {"sequence": " ".join(list(re.sub(r"[UZOB]", "X", x["sequence"])))})
            dataset[split] = dataset[split].remove_columns(
                ["identifier", "term", "aspect", "GO Name", "species", "__index_level_0__"]
            )

            tknz_plm = tokenizer_plm(text=dataset[split]["sequence"], padding=False, truncation=False)
            tknz_llm = tokenizer_llm(text=dataset[split]["GO Sentence"], padding=False, truncation=False)

            dataset[split] = dataset[split].add_column(
                "input_ids", [{"sequence": seq, "text": txt} for seq, txt in zip(tknz_plm["input_ids"], tknz_llm["input_ids"])]
            )
            dataset[split] = dataset[split].add_column(
                "attention_mask",
                [{"sequence": seq, "text": txt} for seq, txt in zip(tknz_plm["attention_mask"], tknz_llm["attention_mask"])],
            )

        dataset = dataset.remove_columns(["sequence", "GO Sentence"])

        print("Saving processed dataset to disk...")
        dataset.save_to_disk(processed_dataset_path)

    print(dataset)
    print(dataset["train"][0])

    data_collator = DataCollatorForProtT5CLIP(
        tokenizer_plm=tokenizer_plm, tokenizer_llm=tokenizer_llm, padding=True, pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir="tmp/models/checkpoints/",
        run_name=run.name if USE_WANDB else None,
        learning_rate=1e-3,
        per_device_train_batch_size=26,
        # per_device_eval_batch_size=16,
        num_train_epochs=1,
        logging_steps=1,
        # do_train=False,
        # do_eval=False,
        # eval_steps=300,
        # save_strategy="steps",
        # save_steps=300,
        remove_unused_columns=False,
        # label_names=["labels"],
        seed=69420,
        report_to=report_to,
    )

    def compute_metrics(eval_preds):
        return {
            "loss": 1.0,
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],  # .select(range(1000)), # CHANGE !!!!!!
        # eval_dataset=dataset['valid'],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    gc.collect()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    torch.set_printoptions(profile="full")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    trainer.train()

    save_path = f"tmp/models/protT5-CLIP-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    model.save_pretrained(save_path)


if __name__ == "__main__":
    main()
