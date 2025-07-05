from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Union
from torch import dtype
from google.colab import userdata

import torch

# Load the model and tokenizer.
def load_model(model_name: str, device: str = 'cuda', dtype: Optional[Union[str, dtype]] = torch.float32, cache_dir: Optional[str] = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the model and tokenizer.

    Args:
    - model_name: Name of the model to load.
    - dtype: Data type to use.
    - cache_dir: Directory to cache the model.

    Returns:
    - model: Model loaded.
    - tokenizer: Tokenizer loaded.
    """
    #Load the model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, cache_dir=cache_dir, device_map=device, token = userdata.get('HF_TOKEN'))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer