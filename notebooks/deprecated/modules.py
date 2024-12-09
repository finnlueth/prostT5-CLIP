import torch
import torch.nn as nn


class CLIPModule(nn.Module):
    """
    CLIP module that can be used as a submodule in the main model.
    Handles the similarity computation between protein and text embeddings.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/temperature)))

    def forward(self, protein_features, protein_attention_mask, text_features, text_attention_mask):
        """
        Forward pass for CLIP module
        Args:
            protein_features: Tensor of shape (batch_size, protein_dim)
            text_features: Tensor of shape (batch_size, text_dim) 
        Returns:
            logits_per_protein: Tensor of shape (batch_size, batch_size)
            logits_per_text: Tensor of shape (batch_size, batch_size)
        """
        protein_features = protein_features / protein_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        print(protein_features.shape)
        print(text_features.shape)

        logit_scale = self.logit_scale.exp()
        logits_per_protein = logit_scale * protein_features @ text_features.t()
        logits_per_text = logits_per_protein.t()

        return logits_per_protein, logits_per_text


__all__ = ['CLIPModule']


