from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Union
from torch import dtype
import os

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
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, cache_dir=cache_dir, device_map=device, token = os.getenv("HF_TOKEN"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token = os.getenv("HF_TOKEN"))

    tokenizer.chat_template = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"

    return model, tokenizer