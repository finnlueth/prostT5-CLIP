from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional
import torch


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
    proj_protein_embeds: Optional[torch.FloatTensor] = None
    proj_text_embeds: Optional[torch.FloatTensor] = None
