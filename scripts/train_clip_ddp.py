import random

import accelerate
import torch
import transformers
from accelerate import Accelerator

from src._shared import (
    apply_lora_to_model,
    load_config,
    load_model_with_lora,
    load_tokenizers,
    prepare_dataset,
    save_model_and_logs,
    setup_environment,
    setup_trainer,
    train_model,
    freeze_base_models,
)


def main():
    accelerator = Accelerator()
    
    train_config = load_config()
    
    # todo: add continue training from checkpoint
    
    model_name_identifier, device, report_to, run, USE_WANDB, SEED = setup_environment(train_config)
    model_name_identifier = model_name_identifier + "-ddp"
    
    accelerator.wait_for_everyone()
    
    accelerate.utils.set_seed(SEED+1)
    transformers.set_seed(SEED+2)
    torch.manual_seed(SEED+3)
    random.seed(SEED+4)

    tokenizer_plm, tokenizer_llm = load_tokenizers(train_config)
    dataset = prepare_dataset(train_config, tokenizer_plm, tokenizer_llm)

    model = load_model_with_lora(train_config, accelerator.device)

    if train_config.lora.enabled:
        model = apply_lora_to_model(model, train_config)
    else:
        freeze_base_models(model)
    
    if accelerator.is_main_process:
        print(dataset)
        print(dataset["train"][0])
    
    trainer = setup_trainer(model, dataset, train_config, model_name_identifier, USE_WANDB, tokenizer_plm, tokenizer_llm)

    model, trainer = accelerator.prepare(model, trainer)

    train_model(trainer)

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_model_and_logs(unwrapped_model, trainer, model_name_identifier, train_config)


if __name__ == "__main__":
    main()
