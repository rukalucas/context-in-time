"""Inferred Timing Task with rule inference from reward history."""

import numpy as np
from .utils import BaseTask


class InferredTask(BaseTask):
    """
    Inferred rule timing task. Same as InstructedTask but without rule cues.
    """

    def __init__(self, **kwargs):
        """Initialize inferred timing task."""
        super().__init__(**kwargs)
        self.trials_per_sequence = kwargs.get("trials_per_sequence", 40)
        self.name = "Inferred Task"

    def _generate_block_structure(self, num_trials: int):
        """Generate block structure without any instruction cues."""
        rules, block_starts, _, block_ids = super()._generate_block_structure(
            num_trials
        )
        # Override: no trials have instruction cues in InferredTask
        has_instruction = np.zeros(num_trials, dtype=bool)
        return rules, block_starts, has_instruction, block_ids
