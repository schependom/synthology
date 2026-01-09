import torch


def get_device() -> torch.device:
    """Get the available device (GPU if available, otherwise CPU).

    Returns:
        torch.device: The available device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
