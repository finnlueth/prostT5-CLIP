import dvc.api


def get_params(key=None):
    """Get parameters from DVC"""
    return dvc.api.params_show() if key is None else dvc.api.params_show()[key]
