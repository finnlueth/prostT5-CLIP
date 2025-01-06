from transformers import PretrainedConfig
from dataclasses import dataclass


@dataclass(slots=True)
class ProtT5CLIPConfig(PretrainedConfig):
    model_type = "pro(s)t_T5_CLIP"

    def __init__(
        self,
        name_or_path_llm=None,
        name_or_path_plm=None,
        projection_dim=1024,
        logit_scale_init_value=2.6592,
        llm_config=None,
        plm_config=None,
        output_logits_per_protein=True,
        output_logits_per_text=True, 
        output_protein_embeds=True,
        output_text_embeds=True,
        output_proj_protein_embeds=True,
        output_proj_text_embeds=True,
        output_protein_outputs=False,
        output_text_outputs=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name_or_path_llm = name_or_path_llm
        self.name_or_path_plm = name_or_path_plm
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.llm_config = llm_config
        self.plm_config = plm_config
        self.output_logits_per_protein = output_logits_per_protein
        self.output_logits_per_text = output_logits_per_text
        self.output_protein_embeds = output_protein_embeds 
        self.output_text_embeds = output_text_embeds
        self.output_proj_protein_embeds = output_proj_protein_embeds
        self.output_proj_text_embeds = output_proj_text_embeds
        self.output_protein_outputs = output_protein_outputs
        self.output_text_outputs = output_text_outputs
