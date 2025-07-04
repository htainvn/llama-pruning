from torch import nn


# Count the number of parameters in a model.
# Note: This method is copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
