import torch
import re
from transformers import AutoTokenizer, T5Tokenizer


def get_model_info(model):
    info = []

    info.append(f"Model device: {next(model.parameters()).device}")
    info.append(f"Model PLM device: {next(model.model_plm.parameters()).device}")
    info.append(f"Model LLM device: {next(model.model_llm.parameters()).device}")

    info.append("\nProtein Model (T5) Parameter dtypes:")
    for name, param in model.model_plm.named_parameters():
        info.append(f"{name}: {param.dtype}")

    info.append("\nText Model (Phi) Parameter dtypes:")
    for name, param in model.model_llm.named_parameters():
        info.append(f"{name}: {param.dtype}")

    info.append("\nProjection Layer Parameter dtypes:")
    for name, param in model.protein_projection.named_parameters():
        info.append(f"protein_projection.{name}: {param.dtype}")
    for name, param in model.text_projection.named_parameters():
        info.append(f"text_projection.{name}: {param.dtype}")

    info.append(f"\nLogit Scale dtype: {model.logit_scale.dtype}")

    return "\n".join(info)


def compare_model_parameters_state_dicts(model1, model2, should_match=True, verbose=False):
    """
    Compare two models parameter by parameter.

    Args:
        model1: First model
        model2: Second model
        verbose: If True, print details about each parameter comparison

    Returns:
        bool: True if models are identical, False otherwise
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1

    if missing_in_2 or missing_in_1:
        print("Found mismatch in parameters\n")
        print(f"Models are identical: {False}")

        if missing_in_2:
            print("Keys missing from reloaded model:")
            for key in sorted(missing_in_2):
                print(f"Key {key} missing from reloaded model")

        if missing_in_1:
            print("Keys missing from original model:")
            for key in sorted(missing_in_1):
                print(f"Key {key} missing from original model")

        return False

    parameters_match = True
    mismatched_params = []

    for key in list(state_dict1.keys()):
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        if param1.shape != param2.shape:
            parameters_match = False
            mismatched_params.append((key, "shape mismatch", param1.shape, param2.shape))
            continue

        if not torch.allclose(param1.float(), param2.float(), rtol=1e-5, atol=1e-8):
            parameters_match = False
            mismatched_params.append((key, "value mismatch", torch.max(torch.abs(param1 - param2)).item()))

    if verbose and not parameters_match:
        print("Parameter mismatches:")
        for index, param_info in enumerate(mismatched_params):
            if len(param_info) == 4:
                key, msg, shape1, shape2 = param_info
                print(f"{index}: {key}: {msg} - shape1: {shape1}, shape2: {shape2}")
            else:
                key, msg, diff = param_info
                print(f"{index}: {key}: {msg} - max difference: {diff}")

    print(f"State dictionaries are identical: {parameters_match}")

    return parameters_match


def compare_model_embeddings(
    model, reloaded_model, train_config, tokenizer_llm=None, tokenizer_plm=None, dummy_texts=None, dummy_proteins=None
):
    """Compare embeddings between original and reloaded models using sample inputs.

    Args:
        model: Original model
        reloaded_model: Reloaded model to compare against
        tokenizer_llm: Text tokenizer
        tokenizer_plm: Protein sequence tokenizer
        dummy_texts: List of sample text inputs. Defaults to two test sentences.
        dummy_proteins: List of sample protein sequences. Defaults to two test sequences.

    Returns:
        tuple: (text_match, protein_match, text_exact_match, protein_exact_match)
    """
    if dummy_texts is None:
        dummy_texts = ["This is a test protein sequence text", "This is a different protein test sequence"]

    if dummy_proteins is None:
        dummy_proteins = [
            "MLKFVVVLAAVLSLYAYAPAFEVHNKKNVLMQRVGETLRISDRYLYQTLSKPYKVTLKTLDGHEIFEVVGEAPVTFRFKDKERPVVVASPEHVVGIVAVHNGKIYARNLYIQNISIVSAGGQHSYSGLSWRYNQPNDGKVTDYF",
            "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGE",
        ]

    dummy_proteins = [" ".join(list(re.sub(r"[UZOB]", "X", x))) for x in dummy_proteins]

    if tokenizer_llm is None:
        tokenizer_llm = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=train_config["model"]["text_encoder_name"],
        )

    if tokenizer_plm is None:
        tokenizer_plm = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=train_config["model"]["protein_encoder_name"],
            do_lower_case=False,
            use_fast=True,
            legacy=False,
        )
    text_tokens = tokenizer_llm(dummy_texts, return_tensors="pt", padding=True, truncation=False)
    protein_tokens = tokenizer_plm(dummy_proteins, return_tensors="pt", padding=True, truncation=False)

    text_tokens = {k: v.to(model.device) for k, v in text_tokens.items()}
    protein_tokens = {k: v.to(model.device) for k, v in protein_tokens.items()}

    model.eval()
    with torch.no_grad():
        text_emb_orig = model(input_ids_text=text_tokens["input_ids"], attention_mask_text=text_tokens["attention_mask"])
        protein_emb_orig = model(
            input_ids_sequence=protein_tokens["input_ids"], attention_mask_sequence=protein_tokens["attention_mask"]
        )

    reloaded_model.eval()
    with torch.no_grad():
        text_emb_reload = reloaded_model(
            input_ids_text=text_tokens["input_ids"], attention_mask_text=text_tokens["attention_mask"]
        )
        protein_emb_reload = reloaded_model(
            input_ids_sequence=protein_tokens["input_ids"], attention_mask_sequence=protein_tokens["attention_mask"]
        )

    text_match = torch.allclose(text_emb_orig.proj_text_embeds, text_emb_reload.proj_text_embeds, rtol=1e-4, atol=1e-4)
    protein_match = torch.allclose(
        protein_emb_orig.proj_protein_embeds, protein_emb_reload.proj_protein_embeds, rtol=1e-4, atol=1e-4
    )

    text_exact_match = torch.equal(text_emb_orig.proj_text_embeds, text_emb_reload.proj_text_embeds)
    protein_exact_match = torch.equal(protein_emb_orig.proj_protein_embeds, protein_emb_reload.proj_protein_embeds)

    print(f"Text embeddings shape: {text_emb_orig.proj_text_embeds.shape}")
    print(f"Protein embeddings shape: {protein_emb_orig.proj_protein_embeds.shape}")
    print(f"Text embeddings match: {text_match}")
    print(f"Protein embeddings match: {protein_match}")
    print(f"Text embeddings exact match: {text_exact_match}")
    print(f"Protein embeddings exact match: {protein_exact_match}")

    return text_match, protein_match, text_exact_match, protein_exact_match


def check_model_on_cuda(model):
    """Check if all model parameters are on CUDA device."""
    if torch.cuda.is_available():
        cuda_check_failed = False
        for name, param in model.named_parameters():
            if not param.is_cuda:
                print(f"WARNING: Parameter {name} is not on CUDA")
                cuda_check_failed = True
        if not cuda_check_failed:
            print("All model parameters are on CUDA")
        else:
            print("Some parameters are not on CUDA - see warnings above")
    else:
        print("CUDA is not available")


def check_model_parameters_requires_grad(model):
    """Check if all model parameters require gradients."""
    grad_check_failed = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # print(f"WARNING: Parameter {name} does not require gradients")
            grad_check_failed = True
    if not grad_check_failed:
        print("All model parameters require gradients")
    else:
        print("Some parameters do not require gradients - see warnings above")
