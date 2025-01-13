import torch
import torch.nn as nn
import typing as T
from transformers import (
    AutoModelForCausalLM,
    T5EncoderModel,
    PreTrainedModel,
    modeling_utils,
)

from typing import Optional

from transformers.models.clip.modeling_clip import (
    clip_loss,
    _get_vector_norm,
)

from .configuration_protein_clip import ProtT5CLIPConfig
from .output_protein_clip import ProteinTextOutput


def _switch_phi_padding_side_old(hidden_states, attention_mask):
    """
    Adjusts embeddings from Phi models to move meaningful tokens to the start.
    Args:
        hidden_states: tensor of shape (batch_size, seq_length, hidden_dim)
        attention_mask: tensor of shape (batch_size, seq_length)
    Returns:
        Adjusted hidden states with same shape but meaningful tokens at start
    """
    batch_size = hidden_states.shape[0]

    adjusted_hidden_states = []
    for i in range(batch_size):
        actual_length = attention_mask[i].sum().item()

        sequence_embeddings = hidden_states[i]

        meaningful_embeddings = sequence_embeddings[-actual_length:]
        padding_embeddings = sequence_embeddings[:-actual_length]

        adjusted_sequence = torch.cat([meaningful_embeddings, padding_embeddings])
        adjusted_hidden_states.append(adjusted_sequence)

    adjusted_hidden_states = torch.stack(adjusted_hidden_states)
    return adjusted_hidden_states


def _switch_phi_padding_side(hidden_states, attention_mask):
    """
    Adjusts embeddings from Phi models to move meaningful tokens to the start.
    Args:
        hidden_states: tensor of shape (batch_size, seq_length, hidden_dim)
        attention_mask: tensor of shape (batch_size, seq_length)
    Returns:
        Adjusted hidden states with same shape but meaningful tokens at start
    """
    pad_lengths = (~attention_mask.bool()).sum(dim=1)
    seq_length = hidden_states.size(1)

    indices = torch.arange(seq_length, device=hidden_states.device)
    indices = indices.expand(hidden_states.size(0), -1)
    rolled_indices = (indices + pad_lengths.unsqueeze(1)) % seq_length
    
    adjusted_hidden_states = torch.gather(hidden_states, 1, rolled_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
    adjusted_attention_mask = torch.gather(attention_mask, 1, rolled_indices)

    return adjusted_hidden_states, adjusted_attention_mask


class LogitScale(nn.Module):
    def __init__(self, init_value, dtype=torch.float32):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=dtype))

    def forward(self, x=None):
        return self.scale


class ProtT5CLIP(PreTrainedModel):
    config_class = ProtT5CLIPConfig
    main_input_name = "input_ids_sequence"

    def __init__(self, config: ProtT5CLIPConfig):
        super().__init__(config=config)

        device_map = config.device if hasattr(config, "device") else "auto"

        self.model_llm, self.loading_info_llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.name_or_path_llm,
            device_map=device_map,
            output_loading_info=True,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        self.model_plm, self.loading_info_plm = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=config.name_or_path_plm,
            device_map=device_map,
            output_loading_info=True,
            torch_dtype="auto",
        )

        self.projection_dim = config.projection_dim
        self.protein_embed_dim = config.plm_config.hidden_size
        self.text_embed_dim = config.llm_config.hidden_size
        self._dtype = torch.float32

        self.protein_projection = nn.Linear(self.protein_embed_dim, self.projection_dim, bias=False, dtype=self._dtype)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False, dtype=self._dtype)
        # self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value, dtype=self._dtype))
        self.logit_scale = LogitScale(config.logit_scale_init_value, self._dtype)
        self.drophout = nn.Dropout(p=0.2)

        for name, init_func in modeling_utils.TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)
        self.post_init()

    def encode_protein(
        self,
        protein_ids=None,
        protein_attention_mask=None,
    ):
        outputs = self.model_plm(
            input_ids=protein_ids,
            attention_mask=protein_attention_mask,
        )
        # todo: for prost t5 add trimming function
        return outputs

    def encode_text(
        self,
        text_ids=None,
        text_attention_mask=None,
    ):
        outputs = self.model_llm(input_ids=text_ids, attention_mask=text_attention_mask, output_hidden_states=True)

        is_phi_model = any("phi" in name.lower() for name in self.model_llm.config.architectures)
        if is_phi_model:
            last_hidden = outputs.hidden_states[-1]
            adjusted_last_hidden, adjusted_attention_mask = _switch_phi_padding_side(hidden_states=last_hidden, attention_mask=text_attention_mask)
            outputs.hidden_states = tuple(list(outputs.hidden_states[:-1]) + [adjusted_last_hidden])
            outputs.attention_mask = adjusted_attention_mask
        return outputs

    def forward(
        self,
        input_ids_sequence: Optional[torch.LongTensor] = None,
        input_ids_text: Optional[torch.LongTensor] = None,
        attention_mask_sequence: Optional[torch.Tensor] = None,
        attention_mask_text: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # print("------------------------------- forward -------------------------------")
        # print("input_ids_sequence", input_ids_sequence)
        # print("input_ids_text", input_ids_text)
        # print("self.logit_scale", self.logit_scale)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits_per_protein = None
        logits_per_text = None
        protein_embeds = None
        text_embeds = None
        protein_outputs = None
        text_outputs = None
        proj_protein_embeds = None
        proj_text_embeds = None

        if input_ids_sequence is not None:
            protein_outputs = self.encode_protein(
                protein_ids=input_ids_sequence,
                protein_attention_mask=attention_mask_sequence,
            )
            protein_embeds = protein_outputs["last_hidden_state"].to(self._dtype)
            protein_embeds = self.drophout(protein_embeds)
            proj_protein_embeds = self.protein_projection(protein_embeds)

        if input_ids_text is not None:
            text_outputs = self.encode_text(
                text_ids=input_ids_text,
                text_attention_mask=attention_mask_text,
            )
            text_embeds = text_outputs["hidden_states"][-1].to(self._dtype)
            text_embeds = self.drophout(text_embeds)
            proj_text_embeds = self.text_projection(text_embeds)

        # TODO: check if this is needed or ask somebody about it
        # if attention_mask is not None:
        #     protein_embeds = protein_embeds * attention_mask["attention_mask_sequence"].unsqueeze(-1)
        #     text_embeds = text_embeds * attention_mask["attention_mask_text"].unsqueeze(-1)

        loss = None
        if proj_text_embeds is not None and proj_protein_embeds is not None:
            pe = torch.mean(proj_protein_embeds, dim=1)
            te = torch.mean(proj_text_embeds, dim=1)

            pe = pe / _get_vector_norm(pe)
            te = te / _get_vector_norm(te)

            logit_scale = self.logit_scale(None).exp()
            logits_per_text = torch.matmul(te, pe.t()) * logit_scale
            logits_per_protein = logits_per_text.t()

            if input_ids_sequence is not None and input_ids_text is not None:
                loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (
                logits_per_protein if self.config.output_logits_per_protein else None,
                logits_per_text if self.config.output_logits_per_text else None,
                text_embeds if self.config.output_text_embeds else None,
                protein_embeds if self.config.output_protein_embeds else None,
                text_outputs if self.config.output_text_outputs else None,
                protein_outputs if self.config.output_protein_outputs else None,
                proj_protein_embeds if self.config.output_proj_protein_embeds else None,
                proj_text_embeds if self.config.output_proj_text_embeds else None,
            )
            return ((loss,) + output) if loss is not None else output
        return ProteinTextOutput(
            loss=loss,
            logits_per_protein=logits_per_protein if self.config.output_logits_per_protein else None,
            logits_per_text=logits_per_text if self.config.output_logits_per_text else None,
            protein_embeds=protein_embeds if self.config.output_protein_embeds else None,
            text_embeds=text_embeds if self.config.output_text_embeds else None,
            protein_outputs=protein_outputs if self.config.output_protein_outputs else None,
            text_outputs=text_outputs if self.config.output_text_outputs else None,
            proj_protein_embeds=proj_protein_embeds if self.config.output_proj_protein_embeds else None,
            proj_text_embeds=proj_text_embeds if self.config.output_proj_text_embeds else None,
        )
