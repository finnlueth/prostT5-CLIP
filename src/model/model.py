import torch
import torch.nn as nn
import typing as T
from transformers import (
    AutoModelForCausalLM,
    T5EncoderModel,
    PretrainedConfig,
    CLIPConfig,
    AutoConfig,
    PreTrainedModel,
    modeling_utils,
)
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional
from src.model.utils import pool_features, postprocess_features
from src.model.loss_openclip import ClipLoss

# from src.model.modules import CLIPModule


# adapted from https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/clip/modeling_clip.py#L629
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# adapted from https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/clip/modeling_clip.py#L629
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    text_loss = contrastive_loss(similarity)
    protein_loss = contrastive_loss(similarity.t())
    return (text_loss + protein_loss) / 2.0


# adapted from https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/clip/modeling_clip.py#L629
def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


@dataclass
class ProteinTextOutput(ModelOutput):
    """
    Output type for protein-text models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Contrastive loss between protein and text embeddings.
        logits_per_protein (`torch.FloatTensor` of shape `(batch_size, batch_size)`):
            Similarity between each protein and all texts in the batch.
        logits_per_text (`torch.FloatTensor` of shape `(batch_size, batch_size)`):
            Similarity between each text and all proteins in the batch.
        protein_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Protein embeddings.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Text embeddings.
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_protein: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    protein_embeds: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    protein_outputs: Optional[torch.FloatTensor] = None
    text_outputs: Optional[torch.FloatTensor] = None


class ProtT5CLIP(PreTrainedModel):
    main_input_name = "input_ids_sequence"

    def __init__(self, config: T.Union[dict, PretrainedConfig, CLIPConfig]):
        super().__init__(config=config)

        self.freeze_llm = config.frozen_llm
        self.freeze_plm = config.frozen_plm

        self.model_llm, self.loading_info_llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.name_or_path_llm,
            device_map="auto",
            output_loading_info=True,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        
        llm_dtype = next(self.model_llm.parameters()).dtype

        self.model_plm, self.loading_info_plm = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=config.name_or_path_plm,
            device_map="auto",
            output_loading_info=True,
            torch_dtype=llm_dtype,
        )

        self.projection_dim = config.projection_dim
        self.protein_embed_dim = config.plm_config.hidden_size
        self.text_embed_dim = config.llm_config.hidden_size

        self.protein_projection = nn.Linear(self.protein_embed_dim, self.projection_dim, bias=False, dtype=llm_dtype)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False, dtype=llm_dtype)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value, dtype=llm_dtype))
        
        for name, init_func in modeling_utils.TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)

    def encode_protein(
        self,
        protein_ids=None,
        protein_attention_mask=None,
    ):
        if self.freeze_plm:
            with torch.no_grad():
                outputs = self.model_plm(
                    input_ids=protein_ids,
                    attention_mask=protein_attention_mask,
                )
        else:
            outputs = self.model_plm(
                input_ids=protein_ids,
                attention_mask=protein_attention_mask,
            )
        return outputs

    def encode_text(
        self,
        text_ids=None,
        text_attention_mask=None,
    ):
        if self.freeze_llm:
            with torch.no_grad():
                outputs = self.model_llm(input_ids=text_ids, attention_mask=text_attention_mask, output_hidden_states=True)
        else:
            outputs = self.model_llm(input_ids=text_ids, attention_mask=text_attention_mask, output_hidden_states=True)
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        protein_outputs = self.encode_protein(
            protein_ids=input_ids_sequence,
            protein_attention_mask=attention_mask_sequence,
        )
        text_outputs = self.encode_text(
            text_ids=input_ids_text,
            text_attention_mask=attention_mask_text,
        )
        
        protein_embeds = protein_outputs["last_hidden_state"]
        protein_embeds = self.protein_projection(protein_embeds)

        text_embeds = text_outputs["hidden_states"][-1]
        text_embeds = self.text_projection(text_embeds)

        protein_embeds = torch.mean(protein_embeds, dim=1)
        text_embeds = torch.mean(text_embeds, dim=1)
        
        # TODO: check if this is needed or ask somebody about it
        # if attention_mask is not None:
        #     protein_embeds = protein_embeds * attention_mask["attention_mask_sequence"].unsqueeze(-1)
        #     text_embeds = text_embeds * attention_mask["attention_mask_text"].unsqueeze(-1)

        protein_embeds = protein_embeds / _get_vector_norm(protein_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)
        
        # print("shape of protein_embeds: ", protein_embeds.shape)
        # print("shape of text_embeds: ", text_embeds.shape)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, protein_embeds.t().to(text_embeds.device)) * logit_scale.to(
            text_embeds.device
        )
        logits_per_protein = logits_per_text.t()

        loss = None
        if input_ids_sequence is not None and input_ids_text is not None:
            loss = clip_loss(logits_per_text)
        
        
        if not return_dict:
            output = (logits_per_protein, logits_per_text, text_embeds, protein_embeds, text_outputs, protein_outputs)
            return ((loss,) + output) if loss is not None else output
        # print("loss: ", loss)
        return ProteinTextOutput(
            loss=loss,
            logits_per_protein=logits_per_protein,
            logits_per_text=logits_per_text,
            protein_embeds=protein_embeds,
            text_embeds=text_embeds,
            protein_outputs=protein_outputs,
            text_outputs=text_outputs,
        )
