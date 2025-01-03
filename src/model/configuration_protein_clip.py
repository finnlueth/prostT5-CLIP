from transformers import PretrainedConfig
from dataclasses import dataclass


@dataclass
class ProtT5CLIPConfig(PretrainedConfig):
    model_type = "prot_t5_clip"

    def __init__(
        self,
        name_or_path_llm=None,
        name_or_path_plm=None,
        projection_dim=1024,
        logit_scale_init_value=2.6592,
        llm_config=None,
        plm_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name_or_path_llm = name_or_path_llm
        self.name_or_path_plm = name_or_path_plm
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.llm_config = llm_config
        self.plm_config = plm_config
