"""Transition Task - Gradual transition from instructed to inferred via rule_cue_prob parameter."""

import numpy as np
from .utils import BaseTask


class TransitionTask(BaseTask):
    """
    Transition task with adjustable cue probability for curriculum learning.

    Uses simple per-trial cue_probability to control what fraction of trials receive
    explicit rule cues. This allows smooth interpolation between fully instructed
    (cue_probability=1.0) and fully inferred (cue_probability=0.0) tasks.

    Block Structure:
    - Block length = block_min + geometric(1/block_mean)
    - Each trial shows cue with probability cue_probability (independent per trial)
    - Rules alternate between blocks
    """

    def __init__(self, rule_cue_prob: float = 0.9, **kwargs):
        """Initialize transition task with adjustable cue probability."""
        kwargs["rule_cue_prob"] = rule_cue_prob
        super().__init__(**kwargs)
        self.trials_per_sequence = kwargs.get("trials_per_sequence", 40)
        self.name = "Transition Task"

    def _generate_block_structure(self, num_trials: int) -> tuple:
        """Override: Generate block structure with per-trial rule_cue_prob."""
        rules, block_starts, _, block_ids = super()._generate_block_structure(
            num_trials
        )
        # Override: each trial shows cue with probability rule_cue_prob
        has_instruction = np.random.rand(num_trials) < self.rule_cue_prob
        return rules, block_starts, has_instruction, block_ids
