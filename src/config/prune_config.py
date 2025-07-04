from dataclasses import dataclass, field
from typing import Optional, List
from config.prune_grid_search_config import PruneGridSearchConfig

import logging
import os

logger = logging.getLogger(__name__)


# Dataclass to store the prune configuration.
@dataclass
class PruneConfig:
    model_name: str = "meditsolutions/Llama-3.2-SUN-1B-chat"
    dtype: str = "float32"
    device: str = "cuda"
    output: str = "results/pruned_model"
    apply_chat_template: bool = False
    prompt: str = "What is the capital of France?"
    apply_chat_template: bool = False
    max_new_tokens: int = 50
    prune_percent: float = 0.2
    prune_method: str = "mk_prune"
    use_normalized_weights: bool = False
    use_layer_norm_tweaks: bool = False
    layer_norm_scale: float = 4.0
    log_dir: str = "logs"
    stop_logging: bool = False
    test_only: bool = False
    print_summary: bool = False
    quiet: bool = False
    cache_dir: Optional[str] = None
    target_size: Optional[int] = None
    gate_up_weight_weights: Optional[List[float]] = field(
        default_factory=lambda: [1.0, 1.0]
    )
    eval_dataset: Optional[str] = None
    eval_dataset_size: Optional[int] = None
    grid_search: Optional[PruneGridSearchConfig] = None

    def __post_init__(self):
        # Check if the prune method is valid.
        if self.prune_method not in ["mk_prune", "mk", "mk_prune_adjusted", "mka"]:
            raise ValueError(f"Unknown prune method: {self.prune_method}")
        elif self.prune_method in ["mk", "mk_prune"]:
            self.prune_method = "mk_prune"
        elif self.prune_method in ["mka", "mk_prune_adjusted"]:
            self.prune_method = "mk_prune_adjusted"
        else:
            pass

        # Validate parameters
        if self.use_layer_norm_tweaks and self.layer_norm_scale <= 0:
            raise ValueError("layer_norm_scale must be greater than 0.")

        if self.gate_up_weight_weights is not None:
            if len(self.gate_up_weight_weights) != 2:
                raise ValueError("gate_up_weight_weights must have 2 values.")
            if any(
                [weight < 0 or weight > 1.0 for weight in self.gate_up_weight_weights]
            ):
                raise ValueError("gate_up_weight_weights must be between 0 and 1")

        if self.target_size is not None:
            if self.target_size <= 0:
                raise ValueError("target_size must be greater than 0.")

        if self.prune_percent < 0 or self.prune_percent > 1.0:
            raise ValueError("prune_percent must be between 0 and 1")

        if self.max_new_tokens <= 10:
            raise ValueError("max_new_tokens must be greater than 10")

        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                raise ValueError("cache_dir does not exist.")

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        if self.eval_dataset is not None and self.grid_search is None:
            raise ValueError("Grid search (AutoML) requires an eval_dataset.")

        if self.eval_dataset is not None and self.eval_dataset_size is None:
            logger.warning("eval_dataset_size is not set. Defaulting to 20.")
            self.eval_dataset_size = 20
