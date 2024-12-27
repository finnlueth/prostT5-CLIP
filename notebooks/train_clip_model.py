#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('jupyter nbconvert --to script train_clip_model.ipynb')


# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

from src.model.modeling_protein_clip import ProtT5CLIP
from src.model.data_collator_multi_input import DataCollatorForProtT5CLIP
from src.model.trainer_protein_subset import ProteinSampleSubsetTrainer
from src.model.configuration_protein_clip import ProtT5CLIPConfig

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_from_disk

import torch
import re
import pandas as pd
import numpy as np
import gc
from datetime import datetime

from transformers import (
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    AutoConfig,
    CLIPConfig,
)
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from peft import (
    LoraConfig,
    get_peft_model,
)

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_name_identifier = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

USE_WANDB = False

if USE_WANDB:
    import wandb

    run = wandb.init(project="protT5-CLIP", name=f"protT5-CLIP-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

report_to = "wandb" if USE_WANDB else None

print("Using device:", device)


# In[2]:


plm_config = AutoConfig.from_pretrained("Rostlab/prot_t5_xl_uniref50")
llm_config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

model_config = ProtT5CLIPConfig(
    name_or_path_plm="Rostlab/prot_t5_xl_uniref50",
    name_or_path_llm="microsoft/Phi-3.5-mini-instruct",
    plm_config=plm_config,
    llm_config=llm_config,
    output_hidden_states=True,
    output_attentions=True,
    return_dict=True,
    projection_dim=1024,
    logit_scale_init_value=2.6592,
)

model = ProtT5CLIP(model_config)
print("Loaded model...")


# In[3]:


model


# In[4]:


target_modules = []
modules_to_save = ["protein_projection", "text_projection"]


target_modules += TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['t5']
# target_modules += ["q", "k", "v", "o"]
# target_modules += [f"model_plm.encoder.block.{x}" for x in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['t5']]
modules_to_save += model.loading_info_plm["missing_keys"]


target_modules += ["qkv_proj"]
# target_modules += TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['phi']
# target_modules += ["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
# target_modules += [f"model_llm.model.{x}" for x  in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['phi']]
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


# In[5]:


model.base_model.model.model_llm.model.layers[22].self_attn.qkv_proj.weight.shape


# In[6]:


model.base_model.model.text_projection.original_module.weight


# In[7]:


model.base_model.model.text_projection.modules_to_save.default.weight


# In[8]:


model


# In[6]:


tokenizer_plm = T5Tokenizer.from_pretrained(
    pretrained_model_name_or_path=model_config.name_or_path_plm,
    do_lower_case=False,
    use_fast=True,
    legacy=False,
)

tokenizer_llm = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_config.name_or_path_llm,
)


# In[ ]:


dataset_path = "../tmp/data/train_val_GO_skimmed"
dataset_path_processed = "../tmp/data/train_val_GO_skimmed_processed"

if not os.path.exists(dataset_path_processed):
    print("Processing dataset...")
    dataset = load_from_disk(dataset_path)

    for split in ["train"]:  # dataset:
        tknz_plm = tokenizer_plm(text=dataset[split]["sequence"], padding=False, truncation=False)
        tknz_llm = tokenizer_llm(text=dataset[split]["GO Sentence"], padding=False, truncation=False)

        dataset[split] = dataset[split].add_column("input_ids_sequence", tknz_plm["input_ids"])
        dataset[split] = dataset[split].add_column("attention_mask_sequence", tknz_plm["attention_mask"])
        dataset[split] = dataset[split].add_column("input_ids_text", tknz_llm["input_ids"])
        dataset[split] = dataset[split].add_column("attention_mask_text", tknz_llm["attention_mask"])
    
    dataset.save_to_disk(dataset_path_processed)
else:
    print("Loading dataset from disk...")
    dataset = load_from_disk(dataset_path_processed)

print(dataset)
print(dataset["train"][0])


# In[ ]:


model.to(device)
model.to(torch.bfloat16)

data_collator = DataCollatorForProtT5CLIP(
    tokenizer_plm=tokenizer_plm, tokenizer_llm=tokenizer_llm, padding=True, pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir=f"../tmp/models/checkpoints/{model_name_identifier}",
    # run_name=run.name if USE_WANDB else None,
    # report_to=report_to,
    learning_rate=1e-3,
    per_device_train_batch_size=2,#26,
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
)


def compute_metrics(eval_preds):
    return {
        "loss": 1.0,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


trainer = ProteinSampleSubsetTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(100)),
    # eval_dataset=dataset['valid'],
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)


# In[ ]:


gc.collect()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
torch.set_printoptions(profile="full")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

trainer.train()


# ---
# # Model saving

# In[10]:


from src.model.utils import get_model_info, compare_models


# In[ ]:


model_save_path = f"../tmp/models/protT5-CLIP-{model_name_identifier}"
model.save_pretrained(
    model_save_path,
)
print("Model saved to:", model_save_path)


# In[ ]:


reloaded_model = ProtT5CLIP(model_config)
reloaded_model.load_adapter(model_save_path)
reloaded_model.to(device)
reloaded_model.to(torch.bfloat16)
print("Loading adapter from:", model_save_path)


# In[ ]:


reloaded_model_fresh = ProtT5CLIP(model_config)


# In[ ]:


display("model")
display(model.base_model.model)
display("reloaded_model")
display(reloaded_model)
display("reloaded_model_fresh")
display(reloaded_model_fresh)


# In[15]:


# model = model.merge_and_unload()
# reloaded_model = reloaded_model.merge_and_unload()
# model.to("cpu")
# reloaded_model.to("cpu")


# In[16]:


# model.protein_projection.modules_to_save.default.weight[0]
# reloaded_model.protein_projection.modules_to_save.default.weight[0]
# model.text_projection.modules_to_save.default.weight[0]
# reloaded_model.text_projection.modules_to_save.default.weight[0]
# model.logit_scale
# reloaded_model.logit_scale


# In[ ]:


all(model.model_llm.model.layers[0].self_attn.o_proj.weight[0] == reloaded_model.model_llm.model.layers[0].self_attn.o_proj.weight[0])


# In[ ]:


# print(get_model_info(model))
# print("--------------------------------\n####\n--------------------------------")
# print(get_model_info(reloaded_model))

# Print named parameters names of reloaded_model
for index, (name, param) in enumerate(model.named_modules()):
    print(f"{index}: {name}")


# In[ ]:


# Compare the models
print("Comparing original and reloaded models...")
models_match = compare_models(reloaded_model, model.base_model.model, verbose=True)


# ### Mismatching

# In[ ]:


reloaded_model_fresh.text_projection.weight[0]


# In[ ]:


model.base_model.model.text_projection.modules_to_save.default.weight[0]


# In[ ]:


reloaded_model.text_projection.modules_to_save.default.weight[0]


# In[ ]:


reloaded_model.text_projection.modules_to_save.weight[0]


# ### Matching

# In[ ]:


model.model_plm.encoder.block[19].layer[0].SelfAttention.v.lora_A.default.weight[0]


# In[ ]:


reloaded_model_fresh.model_plm.encoder.block[23].layer[0].SelfAttention.q.weight[0]


# In[ ]:


reloaded_model.model_plm.encoder.block[23].layer[0].SelfAttention.q.lora_A.default.weight[0]


# In[ ]:


model.model_plm.encoder.block[23].layer[0].SelfAttention.q.lora_A.default.weight[0]


# ---
# ## Analysis

# In[10]:


# pd.DataFrame(trainer.state.log_history)


# In[11]:


# import matplotlib.pyplot as plt

# log_df = pd.DataFrame(trainer.state.log_history)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# ax1_twin = ax1.twinx()
# log_df.plot(y='loss', ax=ax1, color='blue', label='Training Loss')
# log_df.plot(y='grad_norm', ax=ax1_twin, color='red', label='Gradient Norm')
# ax1.set_xlabel('Step')
# ax1.set_ylabel('Loss', color='blue')
# ax1_twin.set_ylabel('Gradient Norm', color='red')
# ax1.set_title('Training Loss and Gradient Norm over Time')
# ax1.grid(True)

# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax1_twin.get_legend_handles_labels()
# ax1_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# log_df.plot(y='learning_rate', ax=ax2, color='green', label='Learning Rate')
# ax2.set_xlabel('Step')
# ax2.set_ylabel('Learning Rate')
# ax2.set_title('Learning Rate Schedule')
# ax2.grid(True)
# ax2.legend()

# plt.tight_layout()
# plt.show()


# In[12]:


# import os

# os.makedirs("../tmp/models", exist_ok=True)

# log_df.to_csv("../tmp/models/training_logs.csv", index=True)
# print("Training logs saved to ../tmp/models/training_logs.csv")

