import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
from logging import getLogger

from copy import deepcopy

from src.method.mk_prune import prune_neuron_pairs

logger = getLogger()


# Iterates throught the model layers and applies pruning.
# Note: This method was previously copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
# It was modified to include new ways to calculate the importance score, include the target_size parameter and take normalizations into account.
def update_model(
    model: nn.Module,
    prune_percent: float,
    prune_method: str = "mk_prune",
    use_normalized_weights: bool = False,
    use_layer_norm_tweaks: bool = False,
    layer_norm_scale: Optional[float] = 4.0,
    device: Optional[str] = "cuda",
    target_size: Optional[int] = None,
    gate_up_weight_weights: Optional[list] = [1.0, 1.0],
    deepcopy_model: bool = False,
) -> nn.Module:
    """
    It modifies each mlp layer present in model, to retain only the most
    important neurons. Creating new smaller versions of each layer pruned.

    Args:
    - model: Model to prune.
    - prune_percent: Percentage of neurons to prune.
    - prune_method: Method to calculate the importance score. Currently, only "mk_prune" (alias: "mk") and "mk_prune_adjusted" (alias: "mka") are supported. (default: mk_prune)
    - use_normalized_weights: Use normalized weights to calculate the final weights.
    - use_layer_norm_tweaks: Use layer normalization tweaks.
    - layer_norm_scale: Layer normalization scale. Only used if use_layer_norm_tweaks is True. (default: 4.0)
    - device: Device to use.
    - target_size: Target size for the intermediate layer. (prune_percent will be ignored)
    - gate_up_weight_weights: Weights for the gate and up weights. (default: [1.0, 1.0])
    - deepcopy_model: If True, the model will be copied before pruning. (default: False)

    Returns:
    - model: New pruned model.
    """
    new_intermediate_size = None

    logger.info(
        f"Pruning model, using {prune_method} method, "
        f"normalizing weights: {use_normalized_weights}, "
        f"layer norm tweaks: {use_layer_norm_tweaks}, "
        f"layer norm scale: {layer_norm_scale} "
        f"target size: {target_size}, "
        f"gate up weight weights: {gate_up_weight_weights}\n"
    )

    if deepcopy_model:
        model = deepcopy(model)

    # loop for each model layer.
    for idx, layer in tqdm(
        enumerate(model.model.layers), total=len(model.model.layers)
    ):
        # Since each layer is a LlamaDecoderLayer it contains multiple components
        # Attention, MLP and Layer norms. We're targetting MLP component
        # by accesing layer.mlp.
        mlp = layer.mlp

        if prune_method == "mk_prune":
            pass
        elif prune_method == "mk_prune_adjusted":
            pass
        else:
            raise ValueError(f"Unknown method: {prune_method}")

        # Call the prune_neiron_pairs with the layers and receiving the pruned.
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(
            mlp,
            prune_percent,
            prune_method=prune_method,
            use_normalized_weights=use_normalized_weights,
            device=device,
            target_size=target_size,
            gate_up_weight_weights=gate_up_weight_weights,
        )

        if use_layer_norm_tweaks:
            # Update the layer normalization weights starting from the second layer.
            if idx < len(model.model.layers) - 1:
                last_layer_original_sum = (
                    torch.abs(mlp.down_proj.weight.data).sum()
                    + torch.abs(mlp.up_proj.weight.data).sum()
                    + torch.abs(mlp.gate_proj.weight.data).sum()
                )
                last_layer_pruned_sum = (
                    torch.abs(new_down_proj.weight.data).sum()
                    + torch.abs(new_up_proj.weight.data).sum()
                    + torch.abs(new_gate_proj.weight.data).sum()
                )

                # Update the next layer normalization weights.
                model.model.layers[idx + 1].input_layernorm.weight.data *= (
                    1.0
                    + (1.0 - torch.abs(last_layer_pruned_sum / last_layer_original_sum))
                    / layer_norm_scale
                )

        # Replace the Origiginal Layers with Pruned Layers.
        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj

        torch.cuda.empty_cache()

        # new_intermediate_size only needs to be set once
        if new_intermediate_size is None:
            new_intermediate_size = new_size

    if use_layer_norm_tweaks:
        # Update the last layer normalization weights.
        model.model.norm.weight.data *= (
            1.0
            + (1.0 - torch.abs(last_layer_pruned_sum / last_layer_original_sum))
            / layer_norm_scale
        )

    # Update the model config file.
    model.config.intermediate_size = new_intermediate_size

    # return final_model
    return model
