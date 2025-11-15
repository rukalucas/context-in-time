"""Instructed timing task with sequence of trials and block structure."""

import torch
from .utils import BaseTask


class InstructedTask(BaseTask):
    """
    Instructed timing task with block structure. BPTT through sequence of trials.

    Trial Structure:
    1. Rule report epoch (variable delay + 700ms response) - reports current rule
    2. Timing decision epoch (variable delay + pulses + 700ms response) - timing-based decision
    3. ITI - 1000ms with reward feedback, handled during training

    Block Structure:
    - Block length = block_min + geometric(1/block_mean)
    - First 4 trials of block: always instructed
    - Remaining trials: instructed with probability rule_cue_prob
    - Rules alternate between blocks
    """

    # Override metric_names to include instructed/not instructed split
    metric_names = [
        "decision_accuracy",
        "rule_accuracy",
        "decision_accuracy_instructed",
        "decision_accuracy_not_instructed",
        "rule_accuracy_instructed",
        "rule_accuracy_not_instructed",
    ]

    def __init__(self, **kwargs):
        """Initialize instructed timing task."""
        super().__init__(**kwargs)
        self.trials_per_sequence = kwargs.get("trials_per_sequence", 40)
        self.name = "Instructed"

    def compute_accuracy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        eval_mask: torch.Tensor,
        batch: dict,
    ) -> dict[str, float]:
        """Compute accuracy metrics with instructed/not instructed split.

        Args:
            outputs: [B, T, 2] network outputs
            targets: [B, T, 2] target outputs
            eval_mask: [B, T, 2] evaluation mask
            batch: Batch data (list of dicts)

        Returns:
            Dictionary with 6 metrics: overall + instructed/not instructed split
        """
        # Get per-trial accuracies [N, B]
        per_trial_metrics = super().compute_accuracy(
            outputs, targets, eval_mask, batch, reduce=False
        )
        decision_accs = per_trial_metrics["decision_accuracy"]  # [N, B]
        rule_accs = per_trial_metrics["rule_accuracy"]  # [N, B]

        with torch.no_grad():
            # Extract has_instruction for each trial
            has_instruction_list = [
                trial_dict["metadata"]["has_instruction"] for trial_dict in batch
            ]
            has_instruction = torch.stack(
                [torch.from_numpy(hi) for hi in has_instruction_list]
            )  # [N, B]

            # Group by instructed/not instructed
            instructed_mask = has_instruction
            not_instructed_mask = ~has_instruction

            # Compute overall and conditional means
            return {
                "decision_accuracy": decision_accs.mean().item(),
                "rule_accuracy": rule_accs.mean().item(),
                "decision_accuracy_instructed": decision_accs[instructed_mask]
                .mean()
                .item()
                if instructed_mask.any()
                else float("nan"),
                "decision_accuracy_not_instructed": decision_accs[not_instructed_mask]
                .mean()
                .item()
                if not_instructed_mask.any()
                else float("nan"),
                "rule_accuracy_instructed": rule_accs[instructed_mask].mean().item()
                if instructed_mask.any()
                else float("nan"),
                "rule_accuracy_not_instructed": rule_accs[not_instructed_mask]
                .mean()
                .item()
                if not_instructed_mask.any()
                else float("nan"),
            }
