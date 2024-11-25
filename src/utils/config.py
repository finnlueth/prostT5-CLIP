import dvc.api


def get_params(key=None):
    """Get parameters from DVC"""
    params = dvc.api.params_show()
    if key is None:
        return params
    return params.get(key, None)
