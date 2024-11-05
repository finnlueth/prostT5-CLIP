import copy
import inspect
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import random
import os

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    KLDivLoss,
    MSELoss,
)
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5Config,
    T5EncoderModel,
    T5ForTokenClassification,
    T5PreTrainedModel,
    T5Tokenizer,
    set_seed,
)

SEED = 69420
BASE_MODEL_PLM = "Rostlab/prot_t5_xl_uniref50"
# BASE_MODEL_PLM = "facebook/esm2_t6_8M_UR50D"
BASE_MODEL_LLM = "microsoft/Phi-3.5-mini-instruct"
# BASE_MODEL_LLM = "meta-llama/Llama-3.2-1B-Instruct"


VERBOSE = True
FILE_PATHS = {
    "models": "../tmp/models/",
    "data": "../tmp/data/",
}

TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'batch_size': 2,
    'num_epochs': 10,
    'logging_steps': 1,
    'eval_steps': 300,
    'save_steps': 9999999,
}

for x in FILE_PATHS.values():
    os.makedirs(x, exist_ok=True)

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# device = 'cpu'

print(f"Using device:\t {device}")