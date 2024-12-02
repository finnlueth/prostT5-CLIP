from collections import OrderedDict
from typing import Tuple, Union, Literal, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from open_clip.loss import ClipLoss, SigLipLoss




# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""

#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()

#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

#         for name in ["text_projection", "proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 if attr is not None:
#                     attr.data = attr.data.half()

#     model.apply(_convert_weights_to_fp16)

BASE_MODEL_PLM = "Rostlab/prot_t5_xl_uniref50"
BASE_MODEL_LLM = "microsoft/Phi-3.5-mini-instruct"

class PLMEncoder(nn.Module):
    """wraps PLM encoders"""
    def __init__(
        self,
        model_name: Literal["protT5, prostT5, esm2"] | str,
        device: torch.device,
        lora: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        
        if model_name == "protT5":
            self.mod_type = "pt"
            self.model = T5EncoderModel.from_pretrained(
            BASE_MODEL_PLM,
            device_map=device,
            torch_dtype='auto',
            cache_dir="/mnt/volume/mathias/pretrained_models"
            )
            self.tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL_PLM)
            
        if lora:
            print("IF LORA DOES COOL THINGS")
            
    def freeze(self):
        """Freeze model params"""
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, x, emb_type):
        inputs = self.tokenizer(
            x,
            return_tensors = "pt",
            max_length=10_000,
            truncation=True,
            padding=True,
            add_special_tokens=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs).last_hidden_state.cpu()
        # if emb_type == "per_res":
        #     if self.mod_type in ("pt", "ank"):
        #         outputs = outputs[:-1, :]
        #     elif self.mod_type == "esm":
        #         output = np.squeeze(outputs, axis=0)[:-1, :]
        #     return outputs
        
        if emb_type == "per_prot":
            return outputs.mean(axis=1).flatten()
        else:
            raise ValueError("Input valid embedding type")            
        
        
class LLMEncoder(nn.Module):
    """wraps LLM encoders"""
    def __init__(
        self,
        model_name: Literal["phi3.5"] | str,
        device: torch.device,
        lora: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.device = device
        if model_name == "phi3.5":
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_LLM,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
                cache_dir="/mnt/volume/mathias/pretrained_models"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_LLM)
         
        if lora:
            print("IF LORA: DO COOL THINGS")
            
        
    def freeze(self):
        """Freeze model params"""
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x, sentence_level=True):
        """Forward pass, extract token or sentence embeddings"""
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        if sentence_level:
            # Average over tokens to get sentence embedding
            embeddings = last_hidden_state.mean(dim=1)
        else:
            # Keep per-token embeddings
            embeddings = last_hidden_state.squeeze(0)
        
        return embeddings.detach().cpu().float().numpy()   # detach to numpy, remove detach() and numpy() if needed for further computation
    
    
class ProtCLIP(nn.Module):
    def __init__(
        self,
        plm_name: Literal["protT5"],
        llm_name: Literal["phi3.5"],
        loss: Literal["CLIP", "SIGLIP"],
        device: torch.device,
        lora: bool = False,
        
    ):
        super().__init__()
        # REMOVE FREEZE
        self.plm_encoder = PLMEncoder(model_name=plm_name, device=device, lora=lora)
        self.llm_encoder = LLMEncoder(model_name=llm_name, device=device, lora=lora)
        if loss == "CLIP":  
            self.loss = ClipLoss()
            
        print(self.plm_encoder)
        print(self.llm_encoder)
    
    def forward(self, batch: dict[str, Tensor]):
        prot_embed = self.plm_encoder(batch["seq"], "per_prot")
        txt_embed = self.llm_encoder(batch["text"], sentence_level=True) # sentence level true correct?
        # add normalization
        return prot_embed, txt_embed
    
    def compute_loss(self, prot_embed, txt_embed):
        return self.loss(prot_embed, txt_embed) # logit_scale temperature in init?        