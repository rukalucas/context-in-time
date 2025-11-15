"""Plotting utilities for analysis."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional

from .utils import _process_single_trial


def plot_psychometric_curve(task, model, num_trials_per_interval=100, save_path=None):
    """
    Plot psychometric curve showing Pro vs Anti accuracy as function of Δt.

    Args:
        task: TimingDecisionTask instance (with discrete_eval=True)
        model: Trained RNN model
        num_trials_per_interval: Number of trials per interval value
        save_path: Optional path to save figure
    """
    model.eval()

    # Use discrete evaluation intervals
    intervals = task.eval_intervals
    accuracies = []
    pro_probs = []

    with torch.no_grad():
        for delta_t in intervals:
            correct = 0
            pro_count = 0
            total = 0

            for _ in range(num_trials_per_interval):
                # Generate trial with fixed Δt
                trial = task.generate_trial(delta_t=delta_t)
                inputs = torch.from_numpy(trial["inputs"])  # [C, T]
                targets = torch.from_numpy(trial["outputs"]).unsqueeze(
                    0
                )  # [1, C_out, T]

                # Forward pass
                outputs = _process_single_trial(model, inputs)

                # Check decision accuracy (during response period)
                T = trial["trial_length"]
                # Last 300ms of trial
                eval_start = T - int(300 / task.dt)
                decision_pred = outputs[0, 0, eval_start:T].mean().item()
                decision_target = targets[0, 0, eval_start:T].mean().item()

                # Check if correct
                if (decision_pred > 0) == (decision_target > 0):
                    correct += 1

                # Check if Pro choice
                if decision_pred > 0 and trial["direction"] > 0:
                    pro_count += 1
                elif decision_pred < 0 and trial["direction"] < 0:
                    pro_count += 1

                total += 1

            accuracies.append(correct / total)
            pro_probs.append(pro_count / total)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax = axes[0]
    ax.plot(intervals, accuracies, "o-", linewidth=2, markersize=8)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3, label="Chance")
    ax.axvline(
        task.decision_threshold, color="r", linestyle="--", alpha=0.3, label="Threshold"
    )
    ax.set_xlabel("Δt (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Decision Accuracy vs Interval Duration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Pro probability
    ax = axes[1]
    ax.plot(intervals, pro_probs, "o-", linewidth=2, markersize=8, color="C1")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3, label="Equal")
    ax.axvline(
        task.decision_threshold, color="r", linestyle="--", alpha=0.3, label="Threshold"
    )
    ax.set_xlabel("Δt (ms)")
    ax.set_ylabel("P(Pro saccade)")
    ax.set_title("Psychometric Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved psychometric curve to {save_path}")

    return fig, (accuracies, pro_probs)


def plot_hidden_state_trajectories(task, model, num_trials=5, save_path=None):
    """
    Plot hidden state trajectories for sample trials.

    Args:
        task: TimingDecisionTask instance
        model: Trained RNN model
        num_trials: Number of trials to plot
        save_path: Optional path to save figure
    """
    model.eval()

    with torch.no_grad():
        # Generate batch
        batch = task.generate_batch(num_trials)
        inputs = batch["inputs"]  # [B, T, C]

        B, T, C = inputs.shape

        # Initialize hidden state
        hidden = torch.zeros(B, model.hidden_size, device=inputs.device)

        # Process timestep by timestep
        outputs_list = []
        hidden_states_list = []

        for t in range(T):
            input_t = inputs[:, t, :]  # [B, C]
            output_t, hidden = model(input_t, hidden)  # [B, C_out], [B, H]
            outputs_list.append(output_t)
            hidden_states_list.append(hidden)

        # Stack to [B, T, C_out] and [B, T, H]
        outputs = torch.stack(outputs_list, dim=1)  # (B, T, C_out)
        hidden_states = torch.stack(hidden_states_list, dim=1)  # (B, T, H)

        # Transpose to [B, C_out, T] and [B, H, T]
        outputs = outputs.transpose(1, 2)  # (B, C_out, T)
        hidden_states = hidden_states.transpose(1, 2)  # (B, H, T)

        # Move to CPU
        hidden_states = hidden_states.cpu().numpy()  # (B, H, T)
        trial_lengths = batch["trial_length"]

        # Plot
        fig, axes = plt.subplots(
            num_trials, 1, figsize=(12, 3 * num_trials), sharex=True
        )
        if num_trials == 1:
            axes = [axes]

        for i in range(num_trials):
            ax = axes[i]
            T = trial_lengths[i]
            time_ms = np.arange(T) * task.dt

            # Plot a subset of hidden units
            num_units_to_plot = min(20, hidden_states.shape[1])
            for j in range(num_units_to_plot):
                ax.plot(time_ms, hidden_states[i, j, :T], alpha=0.5, linewidth=1)

            ax.set_ylabel("Hidden state")
            ax.set_title(
                f"Trial {i + 1}: t_s={batch['t_s'][i]:.0f}ms, t_m={batch['t_m'][i]:.0f}ms, stim_direction={batch['stim_direction'][i]:+.0f}"
            )
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (ms)")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved trajectories to {save_path}")

    return fig


def plot_training_metrics(log_dir, save_path=None):
    """
    Plot training metrics from tensorboard logs.

    Args:
        log_dir: Path to tensorboard log directory
        save_path: Optional path to save figure
    """
    from tensorboard.backend.event_processing import event_accumulator

    # Load events
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get scalars
    loss = ea.Scalars("train/loss")
    decision_acc = ea.Scalars("train/decision_accuracy")
    fixation_acc = ea.Scalars("train/fixation_accuracy")

    # Extract data
    loss_steps = [s.step for s in loss]
    loss_values = [s.value for s in loss]

    decision_steps = [s.step for s in decision_acc]
    decision_values = [s.value for s in decision_acc]

    fixation_steps = [s.step for s in fixation_acc]
    fixation_values = [s.value for s in fixation_acc]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(loss_steps, loss_values, linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(decision_steps, decision_values, linewidth=2, label="Decision")
    ax.axhline(0.9, color="r", linestyle="--", alpha=0.3, label="Target (0.9)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Decision Accuracy")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(fixation_steps, fixation_values, linewidth=2, color="C1")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fixation Accuracy")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training metrics to {save_path}")

    return fig


def plot_rule_specific_psychometric_curves(
    task, model, num_trials_per_interval=100, save_path=None
):
    """
    Plot psychometric curves for SingleTrialTask, separated by rule.

    Args:
        task: SingleTrialTask instance (with discrete_eval=True)
        model: Trained RNN model
        num_trials_per_interval: Number of trials per interval value
        save_path: Optional path to save figure
    """
    model.eval()

    # Use discrete evaluation intervals
    intervals = task.eval_intervals

    # Store results for each rule
    rule1_accuracies = []
    rule1_pro_probs = []
    rule2_accuracies = []
    rule2_pro_probs = []

    with torch.no_grad():
        for delta_t in intervals:
            # Rule 1 results
            rule1_correct = 0
            rule1_pro_count = 0
            rule1_total = 0

            # Rule 2 results
            rule2_correct = 0
            rule2_pro_count = 0
            rule2_total = 0

            for _ in range(num_trials_per_interval):
                # Test both rules
                for rule in [1, -1]:
                    # Generate trial with fixed Δt and rule
                    trial = task.generate_trial(delta_t=delta_t, rule=rule)
                    inputs = torch.from_numpy(trial["inputs"])  # [C, T]
                    targets = torch.from_numpy(trial["outputs"]).unsqueeze(
                        0
                    )  # [1, C_out, T]

                    # Forward pass
                    outputs = _process_single_trial(model, inputs)

                    # Check decision accuracy (during response period)
                    T = trial["trial_length"]
                    # Last 300ms of trial
                    eval_start = T - int(300 / task.dt)
                    decision_pred = outputs[0, 0, eval_start:T].mean().item()
                    decision_target = targets[0, 0, eval_start:T].mean().item()

                    # Check if correct
                    is_correct = (decision_pred > 0) == (decision_target > 0)

                    # Check if Pro choice
                    is_pro = False
                    if decision_pred > 0 and trial["direction"] > 0:
                        is_pro = True
                    elif decision_pred < 0 and trial["direction"] < 0:
                        is_pro = True

                    # Update rule-specific counters
                    if rule == 1:
                        if is_correct:
                            rule1_correct += 1
                        if is_pro:
                            rule1_pro_count += 1
                        rule1_total += 1
                    else:
                        if is_correct:
                            rule2_correct += 1
                        if is_pro:
                            rule2_pro_count += 1
                        rule2_total += 1

            rule1_accuracies.append(rule1_correct / rule1_total)
            rule1_pro_probs.append(rule1_pro_count / rule1_total)
            rule2_accuracies.append(rule2_correct / rule2_total)
            rule2_pro_probs.append(rule2_pro_count / rule2_total)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rule 1 psychometric curve
    ax = axes[0]
    ax.plot(intervals, rule1_pro_probs, "o-", linewidth=2, markersize=8, color="C0")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3, label="Equal")
    ax.axvline(
        task.decision_threshold, color="r", linestyle="--", alpha=0.3, label="Threshold"
    )
    ax.set_xlabel("Δt (ms)")
    ax.set_ylabel("P(Pro saccade)")
    ax.set_title("Rule 1: Short→Pro, Long→Anti")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Add accuracy annotation
    ax.text(
        0.05,
        0.95,
        f"Accuracy: {np.mean(rule1_accuracies):.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    # Rule 2 psychometric curve
    ax = axes[1]
    ax.plot(intervals, rule2_pro_probs, "o-", linewidth=2, markersize=8, color="C1")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3, label="Equal")
    ax.axvline(
        task.decision_threshold, color="r", linestyle="--", alpha=0.3, label="Threshold"
    )
    ax.set_xlabel("Δt (ms)")
    ax.set_ylabel("P(Pro saccade)")
    ax.set_title("Rule 2: Short→Anti, Long→Pro")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Add accuracy annotation
    ax.text(
        0.05,
        0.95,
        f"Accuracy: {np.mean(rule2_accuracies):.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved rule-specific psychometric curves to {save_path}")

    return fig, (rule1_accuracies, rule1_pro_probs, rule2_accuracies, rule2_pro_probs)


def plot_instructed_trial(
    task,
    model,
    delta_t: Optional[float] = None,
    direction: Optional[int] = None,
    rule: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """
    Plot a single trial from SingleTrialTask with 3-subplot structure.

    Args:
        task: SingleTrialTask instance
        model: Trained RNN model
        delta_t: Optional specific interval duration
        direction: Optional specific direction
        rule: Optional specific rule
        save_path: Optional path to save figure
    """
    model.eval()

    with torch.no_grad():
        # Generate trial
        trial = task.generate_trial(delta_t=delta_t, direction=direction, rule=rule)
        inputs_raw = torch.from_numpy(trial["inputs"])  # [C, T]
        targets = torch.from_numpy(trial["outputs"]).unsqueeze(0)  # [1, C_out, T]

        # Forward pass
        outputs = _process_single_trial(model, inputs_raw)

        # Move to CPU
        inputs = inputs_raw.unsqueeze(0).cpu().numpy()  # Add batch dim for consistency
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        T = trial["trial_length"]
        time_ms = np.arange(T) * task.dt

        # Create 3-subplot figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # Top subplot: All 5 input channels
        ax = axes[0]
        ax.plot(time_ms, inputs[0, 0, :T], label="Timing stimulus", linewidth=2)
        ax.plot(time_ms, inputs[0, 1, :T], label="Direction cue", linewidth=2)
        ax.plot(time_ms, inputs[0, 2, :T], label="H fixation signal", linewidth=2)
        ax.plot(time_ms, inputs[0, 3, :T], label="V fixation signal", linewidth=2)
        ax.plot(time_ms, inputs[0, 4, :T], label="Rule cue", linewidth=2)
        ax.set_ylabel("Input value")
        rule_name = "Rule 1" if trial["rule"] == 1 else "Rule 2"
        ax.set_title(
            f"Inputs (t_s={trial['t_s']:.0f}ms, t_m={trial['t_m']:.0f}ms, stim_direction={trial['stim_direction']:+.0f}, {rule_name})"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Middle subplot: Rule-related outputs
        ax = axes[1]
        ax.plot(
            time_ms,
            outputs[0, 2, :T],
            label="Rule report (network)",
            linewidth=2,
            color="C0",
        )
        ax.plot(
            time_ms,
            targets[0, 2, :T],
            label="Rule report target",
            linewidth=2,
            linestyle="--",
            color="C0",
            alpha=0.7,
        )
        ax.plot(
            time_ms,
            outputs[0, 3, :T],
            label="V fixation (network)",
            linewidth=2,
            color="C1",
        )
        ax.plot(
            time_ms,
            targets[0, 3, :T],
            label="V fixation target",
            linewidth=2,
            linestyle="--",
            color="C1",
            alpha=0.7,
        )
        ax.set_ylabel("Output value")
        ax.set_title("Rule Report Outputs")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom subplot: Decision-related outputs
        ax = axes[2]
        ax.plot(
            time_ms,
            outputs[0, 0, :T],
            label="Decision (network)",
            linewidth=2,
            color="C0",
        )
        ax.plot(
            time_ms,
            targets[0, 0, :T],
            label="Decision target",
            linewidth=2,
            linestyle="--",
            color="C0",
            alpha=0.7,
        )
        ax.plot(
            time_ms,
            outputs[0, 1, :T],
            label="H fixation (network)",
            linewidth=2,
            color="C1",
        )
        ax.plot(
            time_ms,
            targets[0, 1, :T],
            label="H fixation target",
            linewidth=2,
            linestyle="--",
            color="C1",
            alpha=0.7,
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Output value")
        ax.set_title("Decision Outputs")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved trial visualization to {save_path}")

    return fig
