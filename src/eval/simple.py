from dataclasses import dataclass
from src.model.generate import get_output
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from logging import getLogger

logger = getLogger()

import torch


@dataclass
class SimpleEvaluator:
    model: AutoModelForCausalLM
    pruned_model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str = "cuda"
    apply_chat_template: bool = False
    max_new_tokens: int = 50
    quiet: bool = False

    original_logits: Optional[torch.Tensor] = None
    pruned_logits: Optional[torch.Tensor] = None

    def generate(
        self,
        prompt: str,
        is_pruned: bool = False,
    ) -> None:
        """
        Evaluate the model using the prompt.

        Args:
        - prompt: Prompt to evaluate the model.
        - is_pruned: If True, the pruned model will be used. (default: False)

        Returns:
        - response: Generated response.
        """

        if is_pruned:
            # Get sample output from the pruned model.
            logger.info("Generating output from the pruned model...")

            _, self.pruned_logits = get_output(
                prompt,
                self.pruned_model,
                self.tokenizer,
                device=self.device,
                apply_chat_template=self.apply_chat_template,
                max_new_tokens=self.max_new_tokens,
                quiet=self.quiet,
            )

        else:
            # Get sample output from the original model.
            logger.info("Generating output from the original model...")

            _, self.original_logits = get_output(
                prompt,
                self.model,
                self.tokenizer,
                device=self.device,
                apply_chat_template=self.apply_chat_template,
                max_new_tokens=self.max_new_tokens,
                quiet=self.quiet,
            )

    def evaluate(self) -> None:
        """
        Evaluate the model using the generated logits.

        Returns:
        - original_logits: Original logits.
        - pruned_logits: Pruned logits.
        """
        if self.original_logits is None:
            raise ValueError(
                "Original logits are missing. Run generate() method first."
            )

        if self.pruned_logits is None:
            raise ValueError("Pruned logits are missing. Run generate() method first.")

        logger.info("Evaluating the model...")

        # Calculate the difference between the logits of the original and pruned model using KL divergence.
        diff = torch.nn.functional.kl_div(
            self.original_logits.log_softmax(dim=-1),
            self.pruned_logits.softmax(dim=-1),
            reduction="batchmean",
        ).item() / self.original_logits.size(1)

        logger.info(f"\n\nKL divergence: {diff:.2f}\n")

        original_max = self.original_logits.max().item()
        pruned_max = self.pruned_logits.max().item()

        original_min = self.original_logits.min().item()
        pruned_min = self.pruned_logits.min().item()

        original_mean = self.original_logits.mean().item()
        pruned_mean = self.pruned_logits.mean().item()

        logger.info(
            f"\n\nLogits statistics:\nOriginal: Max: {original_max:.2f} | Min: {original_min:.2f} | Mean: {original_mean:.2f}\nPruned: Max: {pruned_max:.2f} | Min: {pruned_min:.2f} | Mean: {pruned_mean:.2f}\n"
        )
