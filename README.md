# pro(s)tT5 + CLIP

Combining Protein and LLM Embeddings with CLIP for Protein Design and Function Prediction

Protein design and function prediction are crucial tasks in computational biology with significant implications for drug discovery, enzyme engineering, and protein understanding. Recent advancements in protein language models (pLMs) and large language models (LLMs) have shown promise in capturing intricate patterns in protein sequences and their associated textual descriptions.

This project explores the potential of combining protein embeddings from pLMs (specifically ProtT5) with text embeddings from LLMs (using Microsoft’s Phi-3.5 model) in a unified framework inspired by CLIP (Contrastive Language-Image Pre-training).

While CLIP was originally designed to align image and text embeddings, we propose adapting this approach to create a shared embedding space for protein sequences and their textual descriptions.

The ultimate goals of this project are twofold: 1. Protein Design: Infer protein sequences from textual descriptions of their properties or functions. 2. Function Prediction: Predict protein descriptions from a given sequences.

By leveraging the strengths of both protein and language models, we aim to develop a powerful tool for bidirectional protein-text understanding and generation.

## Training

for singe gpu training

```sh
cd scripts
python train_clip.py
nohup python train_clip.py &
```

or for distributed (multi gpu) training

```sh
cd scripts
accelerate launch train_clip_ddp.py

nohup accelerate launch --config_file ../configs/accelerate_default_config.yaml train_clip_ddp.py &
```

## Conda/Micromamba

```sh
git submodule init

micromamba env create --file env_base.yml --prefix ./.venv -y
micromamba activate --prefix ./.venv
micromamba env remove --prefix ./.venv -y
micromamba deactivate
```

## Docker

```sh
docker container run -it --cpus 8 --memory 32G --gpus all -d --env-file ~/.docker_config/env.list \
-v $(pwd -P)/:/home/lfi/mnt/dev/ \
-v /mnt/project/data/lfi/huggingface:/home/lfi/.cache/huggingface \
-v /mnt/project/data/lfi/.cursor-server:/home/lfi/.cursor-server \
--name finn-container-prostt5-clip finn-image 

docker container start finn-container-prostt5-clip
docker container exec -it finn-container-prostt5-clip "/bin/bash" 

docker container stop finn-container-prostt5-clip

docker container rm finn-container-prostt5-clip
```

## Other Resources

* Training Code: https://github.com/openai/CLIP/issues/83
* https://github.com/wukevin/proteinclip
* https://github.com/pan-emily/protein-clip

## Good Checkpoints

- First good train run, full lora on plm and llm: protT5-CLIP-2025-01-02-22-58-37
- Second good train run, full lora on plm and llm: protT5-CLIP-2025-01-12-14-13-15-ddp
- No Lora, no plm, no llm:

## Drop PLM Layer for LLM only inference

```python
import gc
model.model_plm = None

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
```