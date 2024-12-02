def pool_features(features, attention_mask):
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1)
        pooled_features = (features * mask).sum(dim=1) / mask.sum(dim=1)
    else:
        pooled_features = features.mean(dim=1)
    return pooled_features


def postprocess_features(model, features, attention_mask):
    model_name = model.config.name_or_path
    if model_name == "Rostlab/prot_t5_xl_uniref50":
        return features, attention_mask
    elif model_name == "Rostlab/ProstT5-XL-UniRef100":
        return features, attention_mask
    elif model_name == "microsoft/Phi-3.5-mini-instruct":
        return features, attention_mask
    else:
        raise ValueError(f"Unknown model: {model_name}")
