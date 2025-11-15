"""Shared utility functions and base class for task generation."""

import numpy as np
import torch
from typing import Optional
import matplotlib.pyplot as plt


class BaseTask:
    """
    Base class for timing decision tasks.

    Provides common functionality for:
    - Sampling stimulus times with Weber fraction noise
    - Computing decisions based on rules and measured times
    - Common parameters and initialization

    Subclasses can override:
    - get_input_output_dims(): Return (5, 2) for all tasks
    - compute_accuracy(): Task-specific accuracy computation
    - metric_names: List of metric names returned by compute_accuracy()
    """

    # Metrics returned by compute_accuracy() - subclasses can override
    metric_names = ["decision_accuracy", "rule_accuracy"]

    def __init__(
        self,
        dt: float = 20.0,
        pulse_width: float = 50.0,
        decision_threshold: float = 850.0,
        delta_t_min: float = 530.0,
        delta_t_max: float = 1170.0,
        fixation_delay_min: float = 400.0,
        fixation_delay_max: float = 900.0,
        rule_report_period: float = 700.0,
        response_period: float = 700.0,
        grace_period: float = 400.0,
        fixation_grace_period: float = 200.0,
        input_noise_std: float = 0.1,
        w_m: float = 0.05,
        discrete_eval: bool = False,
        # Sequence-specific parameters
        inter_trial_interval: float = 1500.0,
        reward_duration: float = 500.0,
        block_min: int = 10,
        block_mean: int = 6,
        rule_cue_prob: float = 0.7,
        trials_per_sequence: int = 1,
    ):
        """Initialize base task parameters."""
        # Core timing parameters
        self.dt = dt  # Timestep size in milliseconds
        self.pulse_width = pulse_width  # Duration of timing pulse flashes in ms
        self.decision_threshold = (
            decision_threshold  # Time boundary for short/long decisions (850ms)
        )
        self.delta_t_range = (
            delta_t_min,
            delta_t_max,
        )  # Range of stimulus intervals to sample from
        self.input_noise_std = (
            input_noise_std  # Std dev of Gaussian noise added to inputs
        )
        self.w_m = (
            w_m  # Weber fraction for scalar timing noise (measured time variance)
        )
        self.discrete_eval = discrete_eval  # If True, sample from discrete intervals; if False, continuous uniform
        self.eval_intervals = np.array(
            [530, 610, 690, 770, 850, 930, 1010, 1090, 1170]
        )  # Discrete eval intervals

        # Trial structure parameters
        self.fixation_delay_range = (
            fixation_delay_min,
            fixation_delay_max,
        )  # Variable fixation delay range
        self.rule_report_period = (
            rule_report_period  # Duration of rule report response period
        )
        self.response_period = response_period  # Duration of decision response period
        self.grace_period = grace_period  # Grace period at start of response (no loss penalty to avoid sharp changes in target)
        self.fixation_grace_period = fixation_grace_period  # Grace period after rule report when returning to fixation

        # Sequence parameters
        self.inter_trial_interval = (
            inter_trial_interval  # Duration between trials in ms
        )
        self.reward_duration = reward_duration  # Duration of reward signal during ITI
        self.block_min = block_min  # Minimum block length (trials)
        self.block_mean = (
            block_mean  # Mean of geometric distribution for additional block trials
        )
        self.rule_cue_prob = rule_cue_prob  # Probability of showing rule cue per trial
        self.trials_per_sequence = trials_per_sequence  # Number of trials per sequence

        # Task name for plots (can be overridden in subclasses)
        self.name = self.__class__.__name__.replace("Task", " Task")

        # Compute max trial length
        max_rule_epoch = int(self.fixation_delay_range[1] / dt) + int(
            rule_report_period / dt
        )
        max_timing_epoch = (
            int(self.fixation_delay_range[1] / dt)
            + int((self.delta_t_range[1]) / dt)
            + int(response_period / dt)
        )
        self.max_timesteps = max_rule_epoch + max_timing_epoch

        # ITI parameters
        self.iti_len = int(inter_trial_interval / dt)
        self.reward_len = int(reward_duration / dt)

    def _sample_stimulus_time(self, t_s: Optional[float] = None) -> tuple[float, float]:
        """Sample stimulus time and measured time with Weber noise. t_m ~ N(t_s, (w_m * t_s)^2)."""
        if t_s is None:
            if self.discrete_eval:
                t_s = float(np.random.choice(self.eval_intervals))
            else:
                t_s = np.random.uniform(*self.delta_t_range)

        t_m = np.random.normal(t_s, t_s * self.w_m)
        t_m = np.clip(t_m, self.delta_t_range[0], self.delta_t_range[1])
        return t_s, t_m

    def _compute_decision(
        self, t_s: float, stim_direction: float, rule: float
    ) -> float:
        """Compute decision based on rule, true stimulus time, and direction."""
        if np.isclose(t_s, self.decision_threshold):
            return np.random.choice([-stim_direction, stim_direction])
        if rule == 1.0:  # C1 in Morteza's paper
            return stim_direction if t_s < self.decision_threshold else -stim_direction
        elif rule == -1.0:  # C2
            return -stim_direction if t_s < self.decision_threshold else stim_direction
        else:
            raise NotImplementedError(f"Unknown rule={rule}")

    def get_input_output_dims(self) -> tuple[int, int]:
        """Return input and output dimensions. All tasks return (5, 2)."""
        return (5, 2)

    def generate_trial(
        self,
        t_s: Optional[float] = None,
        stim_direction: Optional[int] = None,
        rule: Optional[int] = None,
        has_instruction: bool = True,
    ) -> dict:
        """Generate a single trial with rule report, timing, and decision epochs.
        Args:
            t_s: Optional fixed stimulus interval
            stim_direction: Stimulus direction (+1 or -1), random if None
            rule: Rule indicator (+1 or -1), random if None
            has_instruction: Whether to show explicit rule cue

        Returns:
            Dictionary with trial data:
                - inputs: [5, T] input array
                - targets: [2, T] target array
                - loss_mask: [T, 2] loss mask
                - eval_mask: [T, 2] evaluation mask
                - t_s, t_m, stim_direction, rule, decision, trial_length, has_instruction
        """
        if stim_direction is None:
            stim_direction = np.random.choice([-1, 1])
        if rule is None:
            rule = np.random.choice([-1, 1])
        t_s, t_m = self._sample_stimulus_time(
            t_s
        )  # if t_s is None, it is also sampled here

        # Sample variable delays at the beginning of the trial and before timing epoch
        initial_fixation_rule = np.random.uniform(*self.fixation_delay_range)
        initial_fixation_timing = np.random.uniform(*self.fixation_delay_range)

        # Convert to timesteps
        t_initial_fix_rule = int(initial_fixation_rule / self.dt)
        t_rule_report = int(self.rule_report_period / self.dt)
        t_initial_fix_timing = int(initial_fixation_timing / self.dt)
        t_pulse = int(self.pulse_width / self.dt)
        t_inter = int(t_m / self.dt)
        t_response = int(self.response_period / self.dt)
        t_grace = int(
            self.grace_period / self.dt
        )  # grace period before evaluating rule/decision responses
        t_fixation_grace = int(
            self.fixation_grace_period / self.dt
        )  # grace period when returning to fixation

        # Total trial length
        total_t = (
            t_initial_fix_rule
            + t_rule_report
            + t_initial_fix_timing
            + t_inter
            + t_response
        )

        # Initialize 5 input channels (use float32 for PyTorch compatibility)
        center_fixation = np.ones(total_t, dtype=np.float32) * 0.2  # Tonic background
        horizontal_cue = np.zeros(total_t, dtype=np.float32)
        rule_cue = np.zeros(total_t, dtype=np.float32)
        vertical_cue = np.zeros(total_t, dtype=np.float32)
        reward_cue = np.zeros(
            total_t, dtype=np.float32
        )  # Set during ITI in sequence tasks
        # Create masks (loss_mask and eval_mask) as [T, 2]
        loss_mask = np.ones((total_t, 2), dtype=np.float32)
        eval_mask = np.zeros((total_t, 2), dtype=np.float32)
        # Create targets (2 output channels)
        h_position_target = np.zeros(total_t, dtype=np.float32)
        v_position_target = np.zeros(total_t, dtype=np.float32)

        # Build trial structure
        # Initial delay, rule presentation and report
        rule_report_start = t_initial_fix_rule
        rule_report_end = rule_report_start + t_rule_report
        vertical_cue[rule_report_start:rule_report_end] = (
            1.0  # Vertical cue active during response
        )
        if (
            has_instruction
        ):  # Rule cue active throughout entire rule report epoch if instructed
            rule_cue[0:rule_report_end] = rule
        v_position_target[rule_report_start:rule_report_end] = rule
        rule_eval_start = rule_report_start + t_grace
        eval_mask[rule_eval_start:rule_report_end, 1] = 1  # Vertical eval
        loss_mask[rule_report_start:rule_eval_start, 1] = (
            0.0  # No loss during grace period
        )
        loss_mask[rule_eval_start:rule_report_end, 1] = (
            5.0  # 5x weight during rule evaluation
        )
        # Grace period after rule report when returning to fixation
        fixation_grace_end = min(rule_report_end + t_fixation_grace, total_t)
        loss_mask[rule_report_end:fixation_grace_end, 1] = (
            0.0  # No loss during return to fixation
        )

        # Another fixation delay, timing flashes, and decision response
        timing_ready = rule_report_end + t_initial_fix_timing
        timing_set = timing_ready + t_inter
        center_fixation[timing_ready : timing_ready + t_pulse] = (
            1.0  # Ready flash at center
        )
        horizontal_cue[timing_set : timing_set + t_pulse] = (
            stim_direction  # Set pulse + direction
        )
        timing_response_start = timing_set
        timing_response_end = timing_response_start + t_response
        decision = self._compute_decision(t_s, stim_direction, rule)
        h_position_target[timing_response_start:timing_response_end] = decision
        timing_eval_start = timing_response_start + t_grace
        eval_mask[timing_eval_start:timing_response_end, 0] = 1  # Horizontal eval
        loss_mask[timing_response_start:timing_eval_start, 0] = (
            0.0  # No loss during grace period
        )
        loss_mask[timing_eval_start:timing_response_end, 0] = (
            2.0  # 2x weight during decision evaluation
        )

        inputs = np.stack(
            [center_fixation, horizontal_cue, rule_cue, vertical_cue, reward_cue],
            axis=0,
        ).transpose(1, 0)  # [T, 5]
        targets = np.stack([h_position_target, v_position_target], axis=0).transpose(
            1, 0
        )  # [T, 2]

        # Add noise (cast to float32 for consistency)
        inputs_noisy = inputs + (
            self.input_noise_std * np.random.randn(*inputs.shape)
        ).astype(np.float32)

        # Pad to max_timesteps
        if total_t < self.max_timesteps:
            pad_width = self.max_timesteps - total_t
            inputs_noisy = np.pad(
                inputs_noisy, ((0, pad_width), (0, 0)), mode="constant"
            )
            targets = np.pad(targets, ((0, pad_width), (0, 0)), mode="constant")
            loss_mask = np.pad(loss_mask, ((0, pad_width), (0, 0)), mode="constant")
            eval_mask = np.pad(eval_mask, ((0, pad_width), (0, 0)), mode="constant")

        return {
            "inputs": inputs_noisy,
            "targets": targets,
            "loss_mask": loss_mask,
            "eval_mask": eval_mask,
            "t_s": t_s,
            "t_m": t_m,
            "stim_direction": stim_direction,
            "rule": rule,
            "decision": decision,
            "trial_length": total_t,
            "has_instruction": has_instruction,
            "initial_fixation_rule": initial_fixation_rule,
            "initial_fixation_timing": initial_fixation_timing,
        }

    def _generate_block_structure(self, num_trials: int):
        """Generate block structure with rules and instruction signals.
        Args:
            num_trials: Total number of trials

        Returns:
            Tuple of (rules, block_starts, has_instruction, block_ids)
            Each is an array of length num_trials
        """
        rules = np.zeros(num_trials, dtype=int)
        block_starts = np.zeros(num_trials, dtype=bool)
        has_instruction = np.zeros(num_trials, dtype=bool)
        block_ids = np.zeros(num_trials, dtype=int)

        rule = np.random.choice([1.0, -1.0])
        left_in_block = 0
        block_id = -1

        for trial in range(num_trials):
            if left_in_block == 0:  # Start new block
                block_id += 1
                block_len = self.block_min + np.random.geometric(1 / self.block_mean)
                block_starts[trial] = True
                rule = -rule  # invert rule

                if block_len + trial > num_trials:
                    block_len = num_trials - trial  # Truncate last block if needed
                left_in_block = block_len

                if block_len <= 4:  # first four trials always have rule cue
                    has_instruction[trial : trial + block_len] = True
                else:
                    has_instruction[trial : trial + 4] = True
                    remaining = (
                        block_len - 4
                    )  # remaining trials may have rule cue stochastically
                    has_instruction[trial + 4 : trial + block_len] = (
                        np.random.rand(remaining) < self.rule_cue_prob
                    )

            rules[trial] = rule
            block_ids[trial] = block_id
            left_in_block -= 1

        return rules, block_starts, has_instruction, block_ids

    def generate_sequence(self, num_trials: int = None):
        """Generate sequence of trials. Returns list of trial dicts."""
        if not num_trials:
            num_trials = self.trials_per_sequence

        # Check if this is a sequence task with block structure
        if hasattr(self, "block_min"):
            rules, block_starts, has_instruction, block_ids = (
                self._generate_block_structure(num_trials)
            )

            trials = []
            for i in range(num_trials):
                trial = self.generate_trial(
                    rule=int(rules[i]),
                    has_instruction=bool(has_instruction[i]),
                )
                trial["block_id"] = int(block_ids[i])
                trial["trial_in_block"] = i - np.where(block_starts[: i + 1])[0][-1]
                trial["is_switch"] = bool(block_starts[i]) and i > 0
                trial["trial_index"] = i
                trials.append(trial)
            return trials
        else:
            # Simple task - just generate trials
            return [self.generate_trial() for _ in range(num_trials)]

    def generate_batch(self, batch_size: int):
        """Generate batch for both single-trial and sequence tasks.

        Args:
            batch_size: Number of sequences to generate

        Returns:
            List of N trial dicts (where N = trials_per_sequence):
                - inputs: [B, T, 5] tensor
                - targets: [B, T, 2] tensor
                - loss_mask: [B, T, 2] tensor
                - eval_mask: [B, T, 2] tensor
                - trial_lengths: [B] tensor
                - metadata: dict with arrays of length B
        """
        sequences = [self.generate_sequence() for _ in range(batch_size)]
        N = self.trials_per_sequence
        batch = []

        for trial_idx in range(N):
            # Collect trial_idx from all sequences
            trials = [seq[trial_idx] for seq in sequences]
            trial_dict = {}

            # Stack main data arrays (numpy arrays are already float32)
            main_keys = ["inputs", "targets", "loss_mask", "eval_mask", "trial_length"]
            for key in main_keys:
                trial_dict[key] = torch.from_numpy(np.stack([t[key] for t in trials]))

            # Rename trial_length to trial_lengths (plural)
            trial_dict["trial_lengths"] = trial_dict.pop("trial_length")

            # Collect metadata
            metadata = {}
            for key in trials[0].keys():
                if key not in main_keys:
                    metadata[key] = np.array([t.get(key, None) for t in trials])

            trial_dict["metadata"] = metadata
            batch.append(trial_dict)

        return batch

    def compute_accuracy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        eval_mask: torch.Tensor,
        batch: dict,
        reduce: bool = True,
    ) -> dict[str, float] | dict[str, torch.Tensor]:
        """Compute standard accuracy metrics. Can be overridden for custom metrics.

        Args:
            outputs: [B, T, 2] network outputs
            targets: [B, T, 2] target outputs
            eval_mask: [B, T, 2] evaluation mask
            batch: Batch data (list of dicts)
            reduce: If True, return scalar accuracies. If False, return per-trial accuracies [N, B]

        Returns:
            If reduce=True: dict with scalar accuracies
            If reduce=False: dict with [N, B] accuracy tensors
        """
        with torch.no_grad():
            # Compute per-trial accuracies
            decision_accs = []
            rule_accs = []

            t_start = 0
            for trial_dict in batch:
                trial_len = trial_dict["inputs"].shape[1]
                t_end = t_start + trial_len

                # Extract this trial's data
                trial_outputs = outputs[:, t_start:t_end, :]
                trial_targets = targets[:, t_start:t_end, :]
                trial_eval_mask = eval_mask[:, t_start:t_end, :]

                # Decision accuracy per batch element
                h_eval_mask = trial_eval_mask[:, :, 0] > 0
                h_pred = trial_outputs[:, :, 0]
                h_tgt = trial_targets[:, :, 0]
                h_correct = ((h_pred > 0) == (h_tgt > 0)).float() * h_eval_mask
                h_acc_per_batch = h_correct.sum(dim=1) / (h_eval_mask.sum(dim=1) + 1e-8)

                # Rule accuracy per batch element
                v_eval_mask = trial_eval_mask[:, :, 1] > 0
                v_pred = trial_outputs[:, :, 1]
                v_tgt = trial_targets[:, :, 1]
                v_correct = ((v_pred > 0) == (v_tgt > 0)).float() * v_eval_mask
                v_acc_per_batch = v_correct.sum(dim=1) / (v_eval_mask.sum(dim=1) + 1e-8)

                decision_accs.append(h_acc_per_batch)
                rule_accs.append(v_acc_per_batch)

                t_start = t_end

            # Stack into [N, B] tensors
            decision_accs = torch.stack(decision_accs)  # [N, B]
            rule_accs = torch.stack(rule_accs)  # [N, B]

            if reduce:
                # Reduce to scalars
                return {
                    "decision_accuracy": decision_accs.mean().item(),
                    "rule_accuracy": rule_accs.mean().item(),
                }
            else:
                # Return per-trial tensors
                return {
                    "decision_accuracy": decision_accs,
                    "rule_accuracy": rule_accs,
                }

    def create_trial_figure(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        targets: np.ndarray,
        eval_mask: np.ndarray,
        trial_idx: int,
        batch: dict,
        batch_idx: int = 0,
        iti_inputs: Optional[np.ndarray] = None,
        iti_outputs: Optional[np.ndarray] = None,
        loss_mask: Optional[np.ndarray] = None,
        block_overview: bool = False,
        batch_metadata: Optional[dict] = None,
        fig: Optional[plt.Figure] = None,
    ) -> plt.Figure:
        """Create standard trial visualization. Can be overridden for custom figures.

        Args:
            inputs: [T, 5] input array
            outputs: [T, 2] output array
            targets: [T, 2] target array
            eval_mask: [T, 2] evaluation mask
            trial_idx: Trial index
            batch: Batch data (list of dicts)
            batch_idx: Batch element index (default 0)
            iti_inputs: Optional [iti_len, 5] ITI input array
            iti_outputs: Optional [iti_len, 2] ITI output array
            loss_mask: Optional [T, 2] loss mask array
            block_overview: Whether to include block structure overview subplot
            batch_metadata: Full batch metadata (required if block_overview=True)
            fig: Optional existing figure to reuse (will be cleared)
        """
        # Determine number of subplots and create or reuse figure
        if block_overview:
            if fig is None:
                fig = plt.figure(figsize=(14, 7))
            else:
                fig.clear()

            axes = fig.subplots(3, 1, gridspec_kw={"height_ratios": [0.3, 1, 1]})
            ax_block = axes[0]
            ax_inputs = axes[1]
            ax_outputs = axes[2]
        else:
            if fig is None:
                fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            else:
                fig.clear()
                axes = fig.subplots(2, 1, sharex=True)

            ax_inputs = axes[0]
            ax_outputs = axes[1]

        # Get trial length and metadata
        if isinstance(batch, list):
            # For sequence tasks: trial_idx is which trial in sequence, batch_idx is batch element
            trial_length = batch[trial_idx]["trial_lengths"][batch_idx].item()
            metadata = {
                k: v[batch_idx] for k, v in batch[trial_idx]["metadata"].items()
            }
        else:
            # For single-trial tasks: trial_idx is batch element
            trial_length = batch["metadata"]["trial_length"][trial_idx]
            metadata = {k: v[trial_idx] for k, v in batch["metadata"].items()}

        # Handle ITI - either provided separately or already concatenated
        iti_boundary = None

        if iti_inputs is not None and iti_outputs is not None:
            # ITI provided separately - concatenate it
            iti_boundary = trial_length
            iti_len = iti_inputs.shape[0]

            # Concatenate inputs
            inputs = np.concatenate([inputs[:trial_length], iti_inputs], axis=0)
            outputs = np.concatenate([outputs[:trial_length], iti_outputs], axis=0)

            # Extend targets and eval_mask with zeros for ITI
            targets = np.concatenate(
                [targets[:trial_length], np.zeros((iti_len, 2))], axis=0
            )
            eval_mask = np.concatenate(
                [eval_mask[:trial_length], np.zeros((iti_len, 2))], axis=0
            )

            # Extend loss_mask with zeros for ITI if provided
            if loss_mask is not None:
                loss_mask = np.concatenate(
                    [loss_mask[:trial_length], np.zeros((iti_len, 2))], axis=0
                )

            T = trial_length + iti_len
        else:
            # Check if ITI is already concatenated (from generate_data)
            # For sequence tasks, metadata includes 'iti_start' and 'iti_length'
            if "iti_start" in metadata and inputs.shape[0] > trial_length:
                # ITI already included - just mark the boundary
                iti_boundary = trial_length
                T = inputs.shape[0]
            else:
                # No ITI - just use trial data
                T = trial_length

        time_ms = np.arange(T) * self.dt

        # Block overview subplot (if requested)
        if block_overview:
            if batch_metadata is None:
                raise ValueError(
                    "batch_metadata must be provided when block_overview=True"
                )

            # Determine total number of trials
            if isinstance(batch, list):
                num_trials = len(batch)
            else:
                num_trials = len(batch["trial_lengths"])

            # Extract rules and instruction info
            rules = batch_metadata.get("rule", np.ones(num_trials))
            has_instruction = batch_metadata.get("has_instruction", None)

            trial_indices = np.arange(num_trials)
            colors = ["C0" if r == 1 else "C1" for r in rules]

            # Plot all trials with appropriate markers
            if has_instruction is not None:
                for j in range(num_trials):
                    if has_instruction[j]:
                        ax_block.scatter(
                            j,
                            rules[j],
                            c=colors[j],
                            s=100,
                            marker="o",
                            edgecolors="black",
                            linewidths=1.5,
                            alpha=0.6,
                        )
                    else:
                        ax_block.scatter(
                            j,
                            rules[j],
                            c=colors[j],
                            s=100,
                            marker="x",
                            linewidths=2,
                            alpha=0.6,
                        )
            else:
                ax_block.scatter(
                    trial_indices,
                    rules,
                    c=colors,
                    s=100,
                    alpha=0.6,
                    edgecolors="black",
                    linewidths=1,
                )

            # Highlight current trial
            current_trial = trial_idx
            ax_block.scatter(
                current_trial,
                rules[current_trial],
                s=400,
                facecolors="none",
                edgecolors="lime",
                linewidths=3,
                zorder=10,
            )

            ax_block.set_ylabel("Rule", fontsize=10)
            ax_block.set_yticks([-1, 1])
            ax_block.set_yticklabels(["Rule 2", "Rule 1"], fontsize=9)
            if has_instruction is not None:
                ax_block.set_title(
                    f"Batch Overview (○=instructed, ×=uninstructed, trial {trial_idx + 1}/{num_trials} highlighted)",
                    fontsize=10,
                )
            else:
                ax_block.set_title(
                    f"Batch Overview (trial {trial_idx + 1}/{num_trials} highlighted)",
                    fontsize=10,
                )
            ax_block.grid(True, alpha=0.3, axis="y")
            ax_block.set_xlim(-1, num_trials)
            ax_block.set_ylim(-1.5, 1.5)
            ax_block.set_xticks([])

        # Input channels subplot
        ax_inputs.plot(time_ms, inputs[:T, 0], label="Center fixation", linewidth=2)
        ax_inputs.plot(time_ms, inputs[:T, 1], label="Horizontal cue", linewidth=2)
        ax_inputs.plot(time_ms, inputs[:T, 2], label="Rule cue", linewidth=2, alpha=0.6)
        ax_inputs.plot(
            time_ms, inputs[:T, 3], label="Vertical cue", linewidth=2, alpha=0.6
        )
        ax_inputs.plot(
            time_ms, inputs[:T, 4], label="Reward cue", linewidth=2, alpha=0.6
        )
        ax_inputs.set_ylabel("Input value")

        # Build title with metadata
        title_parts = [f"Trial {trial_idx + 1}: {self.name}"]
        if "rule" in metadata:
            rule_name = "Rule 1" if metadata["rule"] == 1 else "Rule 2"
            title_parts.append(rule_name)
        if "t_s" in metadata:
            title_parts.append(f"t_s={metadata['t_s']:.0f}ms")
        if "stim_direction" in metadata:
            title_parts.append(f"stim_dir={metadata['stim_direction']:+.0f}")

        ax_inputs.set_title(", ".join(title_parts))
        ax_inputs.legend(loc="upper right", fontsize=8)
        ax_inputs.grid(True, alpha=0.3)

        # Outputs subplot
        ax = ax_outputs

        # Shade regions where loss_mask is 0 (grace periods) - add first so it's in background
        if loss_mask is not None:
            for ch in [0, 1]:
                # Find contiguous regions where loss_mask is 0
                zero_mask = loss_mask[:T, ch] == 0
                if zero_mask.any():
                    # Find transitions
                    transitions = np.diff(
                        np.concatenate([[False], zero_mask, [False]]).astype(int)
                    )
                    starts = np.where(transitions == 1)[0]
                    ends = np.where(transitions == -1)[0]

                    for start, end in zip(starts, ends):
                        # Shade from start of first zero timestep to end of last zero timestep
                        # time_ms[end] is the start of the next timestep (end of the zero region)
                        end_time = (
                            time_ms[end]
                            if end < len(time_ms)
                            else time_ms[-1] + self.dt
                        )
                        ax.axvspan(
                            time_ms[start], end_time, alpha=0.15, color="gray", zorder=0
                        )

        ax.plot(
            time_ms, outputs[:T, 0], label="Horizontal output", linewidth=2, color="C0"
        )
        ax.plot(
            time_ms,
            targets[:T, 0],
            label="Horizontal target",
            linewidth=2,
            linestyle="--",
            color="C0",
            alpha=0.7,
        )
        ax.plot(
            time_ms, outputs[:T, 1], label="Vertical output", linewidth=2, color="C1"
        )
        ax.plot(
            time_ms,
            targets[:T, 1],
            label="Vertical target",
            linewidth=2,
            linestyle="--",
            color="C1",
            alpha=0.7,
        )

        # Shade eval regions (eval_mask is [T, 2])
        for ch in [0, 1]:
            eval_start = np.where(eval_mask[:T, ch] > 0)[0]
            if len(eval_start) > 0:
                ax.axvspan(
                    time_ms[eval_start[0]],
                    time_ms[eval_start[-1]],
                    alpha=0.1,
                    color="green",
                )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Output value")
        ax.set_title("Outputs")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add ITI boundary marker if present
        if iti_boundary is not None:
            boundary_time = iti_boundary * self.dt
            ax_inputs.axvline(
                boundary_time, color="red", linestyle="--", linewidth=2, alpha=0.7
            )
            ax_outputs.axvline(
                boundary_time, color="red", linestyle="--", linewidth=2, alpha=0.7
            )
        plt.tight_layout()
        return fig

    def _generate_iti_inputs(
        self,
        is_correct: torch.Tensor,
        metadata: dict,
        iti_len: int,
        reward_len: int,
    ) -> torch.Tensor:
        """Generate ITI inputs based on trial correctness.

        Args:
            is_correct: [B] boolean tensor - whether previous trial was correct
            metadata: dict with 'stim_direction' array of length B
            iti_len: Length of ITI period in timesteps
            reward_len: Length of reward signal in timesteps

        Returns:
            iti_inputs: [B, iti_len, 5] tensor
        """
        B = len(is_correct)

        # Initialize [B, iti_len, 5]
        iti_inputs = torch.zeros(B, iti_len, 5)

        # Channel 0: center fixation (tonic at 0.2)
        iti_inputs[:, :, 0] = 0.2

        # Channel 4: reward cue (active if trial was correct)
        # Channel 1: horizontal cue (shows stimulus direction if trial was correct)
        for b in range(B):
            if is_correct[b]:
                stim_direction = metadata["stim_direction"][b]
                iti_inputs[b, :reward_len, 4] = 1.0  # Reward cue
                iti_inputs[b, :reward_len, 1] = (
                    stim_direction  # Horizontal cue shows direction
                )

        # Add noise
        iti_inputs += self.input_noise_std * torch.randn_like(iti_inputs)

        return iti_inputs

    def _evaluate_trial_correctness_batch(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        eval_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Check if both outputs were correct for batch.

        Args:
            outputs: [B, T, 2] network outputs
            targets: [B, T, 2] target outputs
            eval_masks: [B, T, 2] evaluation masks

        Returns:
            [B] boolean tensor indicating trial correctness
        """
        with torch.no_grad():
            B = outputs.shape[0]
            correct = torch.ones(B, dtype=torch.bool, device=outputs.device)

            # Check both channels
            for channel in [0, 1]:
                channel_mask = eval_masks[:, :, channel] > 0
                for b in range(B):
                    if channel_mask[b].any():
                        pred = outputs[b, :, channel][channel_mask[b]]
                        tgt = targets[b, :, channel][channel_mask[b]]
                        channel_correct = ((pred > 0) == (tgt > 0)).all()
                        correct[b] = correct[b] & channel_correct

            return correct
