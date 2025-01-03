#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !jupyter nbconvert --to python train_clip_model.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove_cell


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import gc
import os
import re
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
)
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPConfig,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)

from src.model.configuration_protein_clip import ProtT5CLIPConfig
from src.model.data_collator_multi_input import DataCollatorForProtT5CLIP
from src.model.modeling_protein_clip import ProtT5CLIP
from src.model.trainer_protein_subset import ProteinSampleSubsetTrainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
torch.set_printoptions(profile="full")

VERBOSE = True

project_name = "protT5-CLIP"
custom_name = ""
model_name_identifier = (
    project_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + (f"-{custom_name}" if custom_name else "")
)

USE_WANDB = False
report_to = "wandb"
if USE_WANDB:
    import wandb

    run = wandb.init(project=project_name, name=model_name_identifier)

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


# In[ ]:


plm_name = "Rostlab/prot_t5_xl_uniref50"
llm_name = "microsoft/Phi-3.5-mini-instruct"

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
    projection_dim=1024,
    logit_scale_init_value=2.6592,
)

model = ProtT5CLIP(model_config)
model.to(device)
# model.to(torch.bfloat16)

if VERBOSE:
    for name, param in model.named_parameters():
        print(f"{name:<96} {param.device}, {param.dtype}, {param.nelement() * param.element_size() / (1024**2):.2f} MB, {param.requires_grad}")
    

print("Loaded model...")


# In[5]:


target_modules = []
modules_to_save = ["protein_projection", "text_projection", "logit_scale"]

target_modules += TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["t5"]
# target_modules += ["q", "k", "v", "o"]
# target_modules += [f"model_plm.encoder.block.{x}" for x in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['t5']]
modules_to_save += model.loading_info_plm["missing_keys"]

target_modules += ["qkv_proj"]
# target_modules += ["qkv_proj", "o_proj"]
# target_modules += TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['phi']
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
    use_rslora=True,
    # layers_to_transform
    # use_dora=True,
)

model = get_peft_model(model, lora_config)

if VERBOSE:
    for name, param in model.named_parameters():
        print(f"{name:<96} {param.device}, {param.dtype}, {param.nelement() * param.element_size() / (1024**2):.2f} MB, {param.requires_grad}")

print("target_modules:", target_modules)
print("modules_to_save:", modules_to_save)
model.print_trainable_parameters()


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

    for split in dataset:
        print(f"Processing {split}, {len(dataset[split])} items.")

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

print(dataset)
print(dataset["train"][0])


# In[ ]:


data_collator = DataCollatorForProtT5CLIP(
    tokenizer_plm=tokenizer_plm, tokenizer_llm=tokenizer_llm, padding=True, pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir=f"../tmp/models/checkpoints/{model_name_identifier}",
    run_name=run.name if USE_WANDB else None,
    report_to=report_to,
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    logging_steps=1,
    # do_train=True,
    # do_eval=True,
    per_device_eval_batch_size=16,
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=5,
    load_best_model_at_end=True,
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
    eval_dataset=dataset["valid"].select(random.sample(range(len(dataset["valid"])), 300)),
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# ---
# # Model saving

# In[ ]:


model_save_path = f"../tmp/models/{model_name_identifier}"
model.save_pretrained(
    model_save_path,
)
print("Model saved to:", model_save_path)


# ---
# ## Model Sanity Checks

# In[24]:


from src.model.utils import compare_models, get_model_info


# In[ ]:


model_save_path = "../tmp/models/protT5-CLIP-2024-12-27-22-32-46"

reloaded_model = ProtT5CLIP(model_config)
reloaded_model.load_adapter(model_save_path)
reloaded_model.to(device)
reloaded_model.to(torch.bfloat16)
print("Loading adapter from:", model_save_path)


# In[ ]:


reloaded_model.logit_scale


# In[ ]:


reloaded_model_fresh = ProtT5CLIP(model_config)
reloaded_model_fresh.to(device)
reloaded_model_fresh.to(torch.bfloat16)
print("Loading fresh model.")


# In[ ]:


display("model")
display(model.base_model.model)
display("reloaded_model")
display(reloaded_model)
display("reloaded_model_fresh")
display(reloaded_model_fresh)


# In[29]:


# model = model.merge_and_unload()
# reloaded_model = reloaded_model.merge_and_unload()
# model.to("cpu")
# reloaded_model.to("cpu")


# In[30]:


# model.protein_projection.modules_to_save.default.weight[0]
# reloaded_model.protein_projection.modules_to_save.default.weight[0]
# model.text_projection.modules_to_save.default.weight[0]
# reloaded_model.text_projection.modules_to_save.default.weight[0]
# model.logit_scale
# reloaded_model.logit_scale


# In[ ]:


all(
    model.model_llm.model.layers[0].self_attn.o_proj.weight[0]
    == reloaded_model.model_llm.model.layers[0].self_attn.o_proj.weight[0]
)


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


# In[ ]:


model.base_model.model.text_projection.modules_to_save.default.weight


# In[ ]:


model.base_model.model.text_projection.original_module.weight


# In[ ]:


dummy_text = "This is a test protein sequence text"
dummy_protein = "MLKFVVVLAAVLSLYAYAPAFEVHNKKNVLMQRVGETLRISDRYLYQTLSKPYKVTLKTLDGHEIFEVVGEAPVTFRFKDKERPVVVASPEHVVGIVAVHNGKIYARNLYIQNISIVSAGGQHSYSGLSWRYNQPNDGKVTDYF"
print(len(dummy_protein))
dummy_protein = " ".join(list(re.sub(r"[UZOB]", "X", dummy_protein)))
print(len(dummy_protein))

text_tokens = tokenizer_llm(dummy_text, return_tensors="pt", padding=False, truncation=False)
protein_tokens = tokenizer_plm(dummy_protein, return_tensors="pt", padding=False, truncation=False)

text_tokens = {k: v.to(model.device) for k, v in text_tokens.items()}
protein_tokens = {k: v.to(model.device) for k, v in protein_tokens.items()}

print(text_tokens["input_ids"])
print(protein_tokens)

model.eval()
with torch.no_grad():
    text_emb_orig = model(input_ids_text=text_tokens["input_ids"], attention_mask_text=text_tokens["attention_mask"])
    protein_emb_orig = model(
        input_ids_sequence=protein_tokens["input_ids"], attention_mask_sequence=protein_tokens["attention_mask"]
    )

reloaded_model_fresh.eval()
with torch.no_grad():
    text_emb_reload = reloaded_model_fresh(
        input_ids_text=text_tokens["input_ids"], attention_mask_text=text_tokens["attention_mask"]
    )
    protein_emb_reload = reloaded_model_fresh(
        input_ids_sequence=protein_tokens["input_ids"], attention_mask_sequence=protein_tokens["attention_mask"]
    )


text_match = torch.allclose(text_emb_orig.proj_text_embeds, text_emb_reload.proj_text_embeds, rtol=1e-4, atol=1e-4)
protein_match = torch.allclose(
    protein_emb_orig.proj_protein_embeds, protein_emb_reload.proj_protein_embeds, rtol=1e-4, atol=1e-4
)

text_exact_match = torch.equal(text_emb_orig.proj_text_embeds, text_emb_reload.proj_text_embeds)
protein_exact_match = torch.equal(protein_emb_orig.proj_protein_embeds, protein_emb_reload.proj_protein_embeds)

print(f"Text embeddings match: {text_match}")
print(f"Protein embeddings match: {protein_match}")
print(f"Text embeddings exact match: {text_exact_match}")
print(f"Protein embeddings exact match: {protein_exact_match}")

print("\nSample text embeddings (first 5 dimensions):")
print("Original:", text_emb_orig.proj_text_embeds[0, :2, :10])
print("Reloaded:", text_emb_reload.proj_text_embeds[0, :2, :10])

print("\nSample protein embeddings (first 5 dimensions):")
print("Original:", protein_emb_orig.proj_protein_embeds[0, :2, :10])
print("Reloaded:", protein_emb_reload.proj_protein_embeds[0, :2, :10])


# In[ ]:


# Calculate cosine similarity between protein and text projections
def cosine_similarity(a, b):
    # Mean pool across sequence length dimension if needed
    if len(a.shape) > 2:
        a = torch.mean(a, dim=1)
    if len(b.shape) > 2:
        b = torch.mean(b, dim=1)

    # Normalize vectors
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)

    # Calculate similarity
    similarity = torch.matmul(a_norm, b_norm.t())
    return similarity


# Calculate similarities for original model
orig_similarity = cosine_similarity(text_emb_orig.proj_text_embeds, protein_emb_orig.proj_protein_embeds)

# Calculate similarities for reloaded model
reload_similarity = cosine_similarity(text_emb_reload.proj_text_embeds, protein_emb_reload.proj_protein_embeds)

print("\nCosine similarities:")
print("Original model:", orig_similarity.item())
print("Reloaded model:", reload_similarity.item())
print("Similarity difference:", (orig_similarity - reload_similarity).item())


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

