import torch

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


def compare_model_parameters_state_dicts(model1, model2, verbose=False):
    """
    Compare two models parameter by parameter.
    
    Args:
        model1: First model
        model2: Second model
        verbose: If True, print details about each parameter comparison
        
    Returns:
        bool: True if models are identical, False otherwise
    """
    # Get state dictionaries
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # Compare keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    # Check for missing keys
    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1
    
    if missing_in_2 or missing_in_1:
        print("Found mismatch in parameters\n")
        print(f"Models are identical: {False}")
        
        if missing_in_2:
            print("\nKeys missing from reloaded model:")
            for key in sorted(missing_in_2):
                print(f"Key {key} missing from reloaded model")
                
        if missing_in_1:
            print("\nKeys missing from original model:")
            for key in sorted(missing_in_1):
                print(f"Key {key} missing from original model")
        
        return False
    
    # Compare parameter values
    parameters_match = True
    mismatched_params = []
    
    for key in list(state_dict1.keys()):
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        # print(f"Key: {key}")
        
        # Check if parameters have same shape
        if param1.shape != param2.shape:
            parameters_match = False
            mismatched_params.append((key, "shape mismatch", param1.shape, param2.shape))
            continue
            
        # Check if parameters have same values
        if not torch.allclose(param1.float(), param2.float(), rtol=1e-5, atol=1e-8):
            parameters_match = False
            mismatched_params.append((key, "value mismatch", 
                                    torch.max(torch.abs(param1 - param2)).item()))
    
    if verbose and not parameters_match:
        print("\nParameter mismatches:")
        for index, param_info in enumerate(mismatched_params):
            if len(param_info) == 4:
                key, msg, shape1, shape2 = param_info
                print(f"{index}: {key}: {msg} - shape1: {shape1}, shape2: {shape2}")
            else:
                key, msg, diff = param_info
                print(f"{index}: {key}: {msg} - max difference: {diff}")
    
    print(f"\nState dictionaries are identical: {parameters_match}")
    
    return parameters_match