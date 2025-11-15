"""Single-trial instructed (cued context-dependent) timing decision task."""

from .utils import BaseTask


class SingleTrialTask(BaseTask):
    """
    Single-trial instructed context-dependent timing decision task. Only one trial, then BPTT.

    Inputs (5 channels):
    1. Center fixation: Tonic at 0.2, flashes to 1.0 for 50ms at pulses
    2. Horizontal cue: Flashes left(-1) or right(+1) for 50ms at second pulse
    3. Rule cue: +1 for Rule 1, -1 for Rule 2 (during rule report epoch)
    4. Vertical cue: Binary (0/1), active during rule report response period
    5. Reward cue: Always 0 (single trial, no ITI)

    Outputs (2 channels):
    1. Horizontal eye position: Decision output (rule-dependent timing decision)
    2. Vertical eye position: Rule report output

    Rules:
    - Rule 1: Short (Δt < 850ms) → Pro; Long (Δt ≥ 850ms) → Anti
    - Rule 2: Short (Δt < 850ms) → Anti; Long (Δt ≥ 850ms) → Pro

    Trial Structure:
    1. Rule presentation and report epoch (variable delay + 700ms response)
    2. Timing decision epoch (variable delay + pulses + 700ms response)
    """

    def __init__(self, **kwargs):
        """Initialize single-trial instructed timing task."""
        if "trials_per_sequence" in kwargs and kwargs["trials_per_sequence"] != 1:
            raise ValueError(
                "SingleTrialTask only supports single trials (trials_per_sequence=1)."
            )
        kwargs["trials_per_sequence"] = 1  # Single trial only
        kwargs["inter_trial_interval"] = 0.0  # No ITI
        kwargs["reward_duration"] = 0.0
        kwargs["rule_cue_prob"] = 1.0  # Always show rule cue

        super().__init__(**kwargs)
        self.name = "Single-Trial"
