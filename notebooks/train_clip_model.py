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

from src.model.configuration_protein_clip import ProtT5CLIPConfig
from src.model.data_collator_multi_input import DataCollatorForProtT5CLIP
from src.model.modeling_protein_clip import ProtT5CLIP
from src.model.trainer_protein_subset import ProteinSampleSubsetTrainer
from src.model.metrics import metrics_factory
import src.model.utils as utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

with open("../configs/model.yaml", "r") as f:
    train_config = yaml.safe_load(f)

# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# torch.set_printoptions(profile="full")
torch.set_printoptions(profile="default")

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
    import wandb
    run = wandb.init(project=project_name, name=model_name_identifier)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


# In[3]:


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
# model.to(torch.bfloat16)

if VERBOSE:
    for name, param in model.named_parameters():
        print(
            f"{name:<96} {param.device}, {param.dtype}, {param.nelement() * param.element_size() / (1024**2):.2f} MB, {param.requires_grad}"
        )

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("Loaded model...")

utils.check_model_on_cuda(model)


# In[7]:


target_modules = ["k"]
modules_to_save = ["protein_projection", "text_projection", "logit_scale"]

# target_modules += TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["t5"]
# target_modules += ["q", "k", "v", "o"]
# target_modules += [f"model_plm.encoder.block.{x}" for x in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['t5']]
modules_to_save += model.loading_info_plm["missing_keys"]

# target_modules += ["qkv_proj"]
# target_modules += ["qkv_proj", "o_proj"]
# target_modules += TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['phi']
# target_modules += [f"model_llm.model.{x}" for x  in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['phi']]
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
    # layers_to_transform
    use_dora=train_config["lora"]["use_dora"],
)

model = get_peft_model(model, lora_config)

if VERBOSE:
    for name, param in model.named_parameters():
        print(
            f"{name:<96} {param.device}, {param.dtype}, {param.nelement() * param.element_size() / (1024**2):.2f} MB, {param.requires_grad}"
        )

print("target_modules:", target_modules)
print("modules_to_save:", modules_to_save)
model.print_trainable_parameters()


# In[5]:


# print("\n".join([x for x, y in model.named_parameters()]))


# In[ ]:


tokenizer_plm = T5Tokenizer.from_pretrained(
    pretrained_model_name_or_path=model_config.name_or_path_plm,
    do_lower_case=False,
    use_fast=True,
    legacy=False,
)

tokenizer_llm = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_config.name_or_path_llm,
)

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
    report_to=report_to if USE_WANDB else None,
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
    # load_best_model_at_end=True,
    # metric_for_best_model="???"
    # greater_is_better=True,
    remove_unused_columns=train_config["trainer"]["remove_unused_columns"],
    label_names=["input_ids_sequence", "attention_mask_sequence", "input_ids_text", "attention_mask_text"],
    logging_strategy="steps",
    # logging_first_step=True,
    logging_steps=train_config["trainer"]["logging_steps"],
    seed=SEED,
)

trainer = ProteinSampleSubsetTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(512)),
    eval_dataset=dataset["test"],  # .select(random.sample(range(len(dataset["test"])), 20)),
    data_collator=data_collator,
    compute_metrics=metrics_factory(),
    eval_sample_size=train_config["trainer"]["eval_sample_size"],
)

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


# In[ ]:


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

trainer.train(ignore_keys_for_eval=ignore_output_keys)
trainer.evaluate(ignore_keys=ignore_output_keys)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# In[ ]:


pd.DataFrame(trainer.state.log_history)


# In[ ]:


from src.plots.train_plots import plot_training_history
import matplotlib.pyplot as plt

fig = plot_training_history(log_history=pd.DataFrame(trainer.state.log_history), train_config=train_config)
plt.show()


# In[ ]:


model.logit_scale.modules_to_save.default.scale


# In[ ]:


model.logit_scale.original_module.scale


# ---
# # Model saving

# In[ ]:


model_save_path = f"../tmp/models/{model_name_identifier}"
model.save_pretrained(
    model_save_path,
)

pd.DataFrame(trainer.state.log_history).to_csv(f"{model_save_path}/training_log.csv", index=False)

with open(f"{model_save_path}/train_config.yaml", "w") as f:
    yaml.dump(train_config, f, sort_keys=False)

print("Model, config, and log saved to:", model_save_path)


# ---
# ## Model Sanity Checks

# In[14]:


from src.model.utils import compare_model_parameters_state_dicts, get_model_info


# In[ ]:


# model_save_path = "../tmp/models/protT5-CLIP-2024-12-28-03-12-32"
# model_save_path = '../tmp/models/protT5-CLIP-2025-01-02-22-58-37'

reloaded_model = ProtT5CLIP(model_config)
reloaded_model.load_adapter(model_save_path)
reloaded_model.to(device)
# reloaded_model.to(torch.bfloat16)
print("Loading adapter from:", model_save_path)


# In[16]:


# del reloaded_model
# gc.collect()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
# if torch.backends.mps.is_available():
#     torch.mps.empty_cache()


# In[ ]:


reloaded_model.logit_scale.modules_to_save.default.scale


# In[ ]:


reloaded_model.logit_scale.original_module.scale


# In[19]:


# reloaded_model_fresh = ProtT5CLIP(model_config)
# reloaded_model_fresh.to(device)
# # reloaded_model_fresh.to(torch.bfloat16)
# print("Loading fresh model.")


# In[20]:


if VERBOSE:
    display("model")
    display(model.base_model.model)
    display("reloaded_model")
    display(reloaded_model)
    display("reloaded_model_fresh")
    display(reloaded_model_fresh)


# In[21]:


# model = model.merge_and_unload()
# reloaded_model = reloaded_model.merge_and_unload()
# model.to("cpu")
# reloaded_model.to("cpu")


# In[22]:


# model.protein_projection.modules_to_save.default.weight[0]
# reloaded_model.protein_projection.modules_to_save.default.weight[0]
# model.text_projection.modules_to_save.default.weight[0]
# reloaded_model.text_projection.modules_to_save.default.weight[0]
# model.logit_scale
# reloaded_model.logit_scale


# In[ ]:


all(
    model.model_llm.model.layers[0].self_attn.qkv_proj.weight[0]
    == reloaded_model.model_llm.model.layers[0].self_attn.qkv_proj.weight[0]
)


# In[24]:


# print(get_model_info(model))
# print("--------------------------------\n####\n--------------------------------")
# print(get_model_info(reloaded_model))

# Print named parameters names of reloaded_model
if VERBOSE:
    for index, (name, param) in enumerate(model.named_modules()):
        print(f"{index}: {name}")


# In[ ]:


reloaded_model


# In[ ]:


model


# In[ ]:


reloaded_model.model_llm.to(torch.bfloat16)
model.model_llm.to(torch.bfloat16)

print("Comparing original and reloaded models...")
models_match = compare_model_parameters_state_dicts(reloaded_model, model.base_model.model, verbose=True)


# In[ ]:


print("Protein projection parameter count:", sum(p.numel() for p in model.protein_projection.parameters()))


# In[ ]:


model.base_model.model.text_projection.modules_to_save.default.weight


# In[ ]:


model.base_model.model.text_projection.original_module.weight


# In[ ]:


dummy_texts = ["This is a test protein sequence text", "This is a protein test sequence test"]
dummy_proteins = [
    "MLKFVVVLAAVLSLYAYAPAFEVHNKKNVLMQRVGETLRISDRYLYQTLSKPYKVTLKTLDGHEIFEVVGEAPVTFRFKDKERPVVVASPEHVVGIVAVHNGKIYARNLYIQNISIVSAGGQHSYSGLSWRYNQPNDGKVTDYF",
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGE"
]

print("dummy_protein length:", [len(x) for x in dummy_proteins])
dummy_proteins = [" ".join(list(re.sub(r"[UZOB]", "X", x))) for x in dummy_proteins]
print("dummy_protein length after processing:", [len(x) for x in dummy_proteins])

text_tokens = tokenizer_llm(dummy_texts, return_tensors="pt", padding=True, truncation=False)
protein_tokens = tokenizer_plm(dummy_proteins, return_tensors="pt", padding=True, truncation=False)

text_tokens = {k: v.to(model.device) for k, v in text_tokens.items()}
protein_tokens = {k: v.to(model.device) for k, v in protein_tokens.items()}

print(text_tokens["input_ids"])
print(protein_tokens["input_ids"])

model.eval()
with torch.no_grad():
    text_emb_orig = model(input_ids_text=text_tokens["input_ids"], attention_mask_text=text_tokens["attention_mask"])
    protein_emb_orig = model(
        input_ids_sequence=protein_tokens["input_ids"], attention_mask_sequence=protein_tokens["attention_mask"]
    )

reloaded_model.eval()
with torch.no_grad():
    text_emb_reload = reloaded_model(
        input_ids_text=text_tokens["input_ids"], attention_mask_text=text_tokens["attention_mask"]
    )
    protein_emb_reload = reloaded_model(
        input_ids_sequence=protein_tokens["input_ids"], attention_mask_sequence=protein_tokens["attention_mask"]
    )


text_match = torch.allclose(text_emb_orig.proj_text_embeds, text_emb_reload.proj_text_embeds, rtol=1e-4, atol=1e-4)
protein_match = torch.allclose(
    protein_emb_orig.proj_protein_embeds, protein_emb_reload.proj_protein_embeds, rtol=1e-4, atol=1e-4
)

text_exact_match = torch.equal(text_emb_orig.proj_text_embeds, text_emb_reload.proj_text_embeds)
protein_exact_match = torch.equal(protein_emb_orig.proj_protein_embeds, protein_emb_reload.proj_protein_embeds)

print()
print(f"Text embeddings shape: {text_emb_orig.proj_text_embeds.shape}")
print(f"Protein embeddings shape: {protein_emb_orig.proj_protein_embeds.shape}")
print()
print(f"Text embeddings match: {text_match}")
print(f"Protein embeddings match: {protein_match}")
print(f"Text embeddings exact match: {text_exact_match}")
print(f"Protein embeddings exact match: {protein_exact_match}")
print()
print("Sample text embeddings (first 5 dimensions):")
print("Original:", text_emb_orig.proj_text_embeds[0, :2, :10])
print("Reloaded:", text_emb_reload.proj_text_embeds[0, :2, :10])
print()
print("\nSample protein embeddings (first 5 dimensions):")
print("Original:", protein_emb_orig.proj_protein_embeds[0, :2, :10])
print("Reloaded:", protein_emb_reload.proj_protein_embeds[0, :2, :10])


# In[ ]:


def cosine_similarity(a, b):
    if len(a.shape) > 2:
        a = torch.mean(a, dim=1)
    if len(b.shape) > 2:
        b = torch.mean(b, dim=1)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)

    similarity = torch.matmul(a_norm, b_norm.t())
    
    return similarity


def cosine_similarity_v2(a, b):
    if len(a.shape) > 2:
        a = torch.mean(a, dim=1)
    if len(b.shape) > 2:
        b = torch.mean(b, dim=1)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)

    similarities = torch.sum(a_norm * b_norm, dim=-1)
    
    return similarities


orig_similarity_v1 = cosine_similarity(text_emb_orig.proj_text_embeds, protein_emb_orig.proj_protein_embeds)
reload_similarity_v1 = cosine_similarity(text_emb_reload.proj_text_embeds, protein_emb_reload.proj_protein_embeds)

orig_similarity_v2 = cosine_similarity_v2(text_emb_orig.proj_text_embeds, protein_emb_orig.proj_protein_embeds)
reload_similarity_v2 = cosine_similarity_v2(text_emb_reload.proj_text_embeds, protein_emb_reload.proj_protein_embeds)

print("\nCosine similarities:")
print("Original model v1:", orig_similarity_v1.tolist())
print("Reloaded model v1:", reload_similarity_v1.tolist())
print("Original model v2:", orig_similarity_v2.tolist())
print("Reloaded model v2:", reload_similarity_v2.tolist())
# print("Similarity difference:", (orig_similarity - reload_similarity).item())


# ---
# ## Analysis

# In[32]:


# pd.DataFrame(trainer.state.log_history)


# In[33]:


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


# In[34]:


# import os

# os.makedirs("../tmp/models", exist_ok=True)

# log_df.to_csv("../tmp/models/training_logs.csv", index=True)
# print("Training logs saved to ../tmp/models/training_logs.csv")

