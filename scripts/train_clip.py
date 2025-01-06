import random

import accelerate
import pandas as pd
import torch
import transformers
import yaml
import os

from src._shared import (
    load_config,
    load_model_with_lora,
    load_tokenizers,
    prepare_dataset,
    setup_environment,
    setup_trainer,
    train_model,
    save_model_and_logs,
)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    train_config = load_config()
        
    model_name_identifier, device, report_to, run, USE_WANDB, SEED = setup_environment(train_config)
    
    accelerate.utils.set_seed(SEED)
    transformers.set_seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    tokenizer_plm, tokenizer_llm = load_tokenizers(train_config)
    
    model = load_model_with_lora(train_config, device)
    
    dataset = prepare_dataset(train_config, tokenizer_plm, tokenizer_llm)
    
    print(dataset)
    print(dataset["train"][0])
    
    trainer = setup_trainer(model, dataset, train_config, model_name_identifier, USE_WANDB, tokenizer_plm, tokenizer_llm)
    
    train_model(trainer)
    
    save_model_and_logs(model, trainer, model_name_identifier, train_config)


if __name__ == "__main__":
    main()
