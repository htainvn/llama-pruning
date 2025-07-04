import torch
from typing import Optional


# Maximum Absolute Weight:
# The maximum absolute weight in a neuron might indicate its significance.
# Note: This method is copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
def get_importance(gate_weight: torch.Tensor, up_weight: torch.Tensor, weights: Optional[list] = [1.0, 1.0]) -> torch.Tensor:
    """
    Calculate the importance score for the weight matrix.

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer
    - weights: Weights for each layer.

    Returns:
    - importance: Importance score for the weight matrix.
    """
    gate_max_abs = torch.max(gate_weight, dim=1).values + torch.abs(
        torch.min(gate_weight, dim=1).values
    ) * weights[0]
    up_max_abs = torch.max(up_weight, dim=1).values + torch.abs(
        torch.min(up_weight, dim=1).values
    ) * weights[1]

    return gate_max_abs.float() + up_max_abs.float()


# Adjusted Importance:
# The adjusted importance score is calculated by dividing the sum of the maximum absolute weight,
# the mean of the weights, and the maximum absolute weight of the weights that are less than the mean,
# by the number of weights that are less than the mean.
def get_adjusted_importance(
    gate_weight: torch.Tensor, up_weight: torch.Tensor, weights: list = [1.0, 1.0]
) -> torch.Tensor:
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.
    - weights: Weights for each layer.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_importance = get_adjusted_weight_importance(gate_weight) * weights[0]
    up_importance = get_adjusted_weight_importance(up_weight) * weights[1]

    return gate_importance + up_importance


# Adjusted Importance:
# The adjusted importance score is calculated by dividing the sum of the maximum absolute weight,
# the mean of the weights, and the maximum absolute weight of the weights that are less than the mean,
# by the number of weights that are less than the mean.
def get_adjusted_weight_importance(weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate the adjusted importance score for the weight matrix.

    Args:
    - weight: Weight matrix from the layer.

    Returns:
    - adjusted_importance: Adjusted importance score for the weight matrix.
    """
    mask = weight > 0.0
    positive_max = (weight * mask.float()).max(dim=1).values
    negative_max_abs = (torch.abs(weight * (~mask).float())).max(dim=1).values

    weight_adjusted_mean = (
        torch.abs(weight).sum(dim=1) - positive_max - negative_max_abs
    ) / (weight.size(1) - 2)

    weight_small_count = (torch.abs(weight) <= weight_adjusted_mean.unsqueeze(1)).sum(
        dim=1
    )

    adjusted_importance = (
        (positive_max.float() + negative_max_abs.float() + weight_adjusted_mean.float())
        * weight.size(1)
        / weight_small_count
    )

    return adjusted_importance
