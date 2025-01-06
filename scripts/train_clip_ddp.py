import random

import accelerate
import pandas as pd
import torch
import transformers
import yaml
from accelerate import Accelerator

# from accelerate.distributed import DistributedDataParallelKwargs
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
    accelerator = Accelerator()
    # accelerator = Accelerator(
    #     kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    #     # Try using gloo backend instead of nccl if issues persist
    #     kwargs_handlers=[DistributedDataParallelKwargs(backend="gloo")]
    # )
    
    train_config = load_config()
    
    model_name_identifier, device, report_to, run, USE_WANDB, SEED = setup_environment(train_config)

    accelerate.utils.set_seed(SEED)
    transformers.set_seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    tokenizer_plm, tokenizer_llm = load_tokenizers(train_config)

    model = load_model_with_lora(train_config, accelerator.device)

    dataset = prepare_dataset(train_config, tokenizer_plm, tokenizer_llm)
    
    print(dataset)
    print(dataset["train"][0])

    trainer = setup_trainer(model, dataset, train_config, model_name_identifier, USE_WANDB, tokenizer_plm, tokenizer_llm)

    model, trainer = accelerator.prepare(model, trainer)

    train_model(trainer)

    if accelerator.is_main_process:
        save_model_and_logs(model, trainer, model_name_identifier, train_config)


if __name__ == "__main__":
    main()
