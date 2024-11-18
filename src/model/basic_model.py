import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    T5EncoderModel,
)
from src.model.utils import pool_features, postprocess_features


class ProtT5CLIP(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        self.model_llm, self.loading_info_llm = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model_llm"],
            # device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            output_loading_info=True,
        )

        self.model_plm, self.loading_info_plm = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=model_cfg["base_model_plm"],
            # device_map='auto',
            output_loading_info=True,
        )

    def encode_protein(
        self,
        protein_ids=None,
        protein_attention_mask=None,
    ):
        if self.freeze_llm:
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
                outputs = self.model_llm(
                    input_ids=text_ids,
                    attention_mask=text_attention_mask,
                )
        else:
            outputs = self.model_llm(
                input_ids=text_ids,
                attention_mask=text_attention_mask,
            )
        return outputs

    def forward(
        self,
        input_ids_sequence=None,
        attention_mask_sequence=None,
        # input_ids_text=None,
        # attention_mask_text=None,
        *args,
        **kwargs,
    ):
        print('test')
        # print(kwargs)
        # protein_features = self.encode_protein(
        #     protein_ids=protein_ids,
        #     protein_attention_mask=protein_attention_mask,
        # )

        # text_features = self.encode_text(
        #     text_ids=text_ids,
        #     text_attention_mask=text_attention_mask,
        # )

        # print(text_features.last_hidden_state.shape, text_features.last_hidden_state)
        # print(text_features.pooler_output.shape, text_features.pooler_output)
