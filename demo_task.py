"""Vibe-coded. Demo the Timing Decision Task and Instructed Timing Task in ./tasks.py."""

from absl import app
from absl import flags
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

from tasks import SingleTrialTask, InstructedTask, InferredTask, TransitionTask
from models.rnn import RNN
from trainers.utils import BaseTrainer

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_trials", 1, "Number of task trials to plot.")
flags.DEFINE_bool("discrete", False, "Use discrete evaluation intervals.")
flags.DEFINE_float(
    "rule_cue_prob",
    0.5,
    "Rule cue probability for transition task (0.0=inferred, 1.0=fully instructed).",
)


def demo_instructed_timing_task(num_trials: int, discrete: bool):
    """Demo the SingleTrialTask with new batch structure."""

    task = SingleTrialTask(
        dt=20.0,
        pulse_width=50.0,
        decision_threshold=850.0,
        delta_t_min=530.0,
        delta_t_max=1170.0,
        fixation_delay_min=400.0,
        fixation_delay_max=900.0,
        rule_report_period=700.0,
        response_period=700.0,
        grace_period=400.0,
        input_noise_std=0.1,
        discrete_eval=discrete,
    )

    for _ in range(num_trials):
        # Generate batch - returns list with 1 trial dict for single-trial tasks
        batch = task.generate_batch(batch_size=1)
        trial_dict = batch[0]  # First (and only) trial in sequence

        inputs = trial_dict["inputs"][0].numpy()  # [T, 5]
        targets = trial_dict["targets"][0].numpy()  # [T, 2]
        eval_mask = trial_dict["eval_mask"][0].numpy()  # [T, 2]
        loss_mask = trial_dict["loss_mask"][0].numpy()  # [T, 2]

        # Use task's create_trial_figure method
        # Since outputs = targets for demo (no model), we pass targets as both
        task.create_trial_figure(
            inputs=inputs,
            outputs=targets,
            targets=targets,
            eval_mask=eval_mask,
            trial_idx=0,
            batch=batch,
            batch_idx=0,
            loss_mask=loss_mask,
        )

    plt.show()


def demo_instructed_task(num_sequences: int, discrete: bool):
    """Demo the InstructedTask with interactive navigation."""

    task = InstructedTask(
        dt=20.0,
        pulse_width=50.0,
        decision_threshold=850.0,
        delta_t_min=530.0,
        delta_t_max=1170.0,
        fixation_delay_min=400.0,
        fixation_delay_max=900.0,
        rule_report_period=700.0,
        response_period=700.0,
        grace_period=400.0,
        input_noise_std=0.1,
        inter_trial_interval=1500.0,
        reward_duration=500.0,
        block_min=10,
        block_mean=6,
        rule_cue_prob=0.7,
        trials_per_sequence=40,
        discrete_eval=discrete,
    )

    for _ in range(num_sequences):
        # New batch structure: list of trial dicts
        batch = task.generate_batch(batch_size=1)

        # batch is a list where each element is a trial dict
        num_trials = len(batch)

        # Extract metadata from first trial to build sequence structure
        batch_metadata = {
            "rule": np.array(
                [batch[i]["metadata"]["rule"][0] for i in range(num_trials)]
            ),
            "has_instruction": np.array(
                [batch[i]["metadata"]["has_instruction"][0] for i in range(num_trials)]
            ),
            "t_s": np.array(
                [batch[i]["metadata"]["t_s"][0] for i in range(num_trials)]
            ),
            "t_m": np.array(
                [batch[i]["metadata"]["t_m"][0] for i in range(num_trials)]
            ),
            "stim_direction": np.array(
                [batch[i]["metadata"]["stim_direction"][0] for i in range(num_trials)]
            ),
        }

        # State for interactive navigation
        state = {"current_trial": 0, "fig": None}

        def plot_trial(trial_idx):
            """Plot a single trial with detailed information."""
            # Extract trial-specific data
            trial_inputs = batch[trial_idx]["inputs"][0].numpy()  # [T, 5]
            trial_targets = batch[trial_idx]["targets"][0].numpy()  # [T, 2]
            trial_loss_mask = batch[trial_idx]["loss_mask"][0].numpy()  # [T, 2]
            trial_eval_mask = batch[trial_idx]["eval_mask"][0].numpy()  # [T, 2]

            # Use task's create_trial_figure with block overview (reuse existing figure)
            fig = task.create_trial_figure(
                inputs=trial_inputs,
                outputs=trial_targets,  # No model, so outputs = targets
                targets=trial_targets,
                eval_mask=trial_eval_mask,
                trial_idx=trial_idx,
                batch=batch,
                batch_idx=0,
                loss_mask=trial_loss_mask,
                block_overview=True,
                batch_metadata=batch_metadata,
                fig=state["fig"],  # Reuse existing figure
            )

            fig.suptitle("Use ← → arrow keys to navigate trials", fontsize=12, y=0.995)

            # Connect keyboard handler on first call
            if state["fig"] is None:
                fig.canvas.mpl_connect("key_press_event", on_key)

            state["fig"] = fig
            fig.canvas.draw()

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == "right":
                state["current_trial"] = min(state["current_trial"] + 1, num_trials - 1)
                plot_trial(state["current_trial"])
            elif event.key == "left":
                state["current_trial"] = max(state["current_trial"] - 1, 0)
                plot_trial(state["current_trial"])

        # Initial plot
        plot_trial(0)

    plt.show()


def demo_inferred_task(num_sequences: int, discrete: bool):
    """Demo the InferredTask with interactive navigation."""

    task = InferredTask(
        dt=20.0,
        pulse_width=50.0,
        decision_threshold=850.0,
        delta_t_min=530.0,
        delta_t_max=1170.0,
        fixation_delay_min=400.0,
        fixation_delay_max=900.0,
        rule_report_period=700.0,
        response_period=700.0,
        grace_period=400.0,
        input_noise_std=0.1,
        inter_trial_interval=1500.0,
        reward_duration=500.0,
        block_min=10,
        block_mean=6,
        trials_per_sequence=40,
        discrete_eval=discrete,
    )

    for _ in range(num_sequences):
        # New batch structure: list of trial dicts
        batch = task.generate_batch(batch_size=1)

        # batch is a list where each element is a trial dict
        num_trials = len(batch)

        # Extract metadata from trials
        batch_metadata = {
            "rule": np.array(
                [batch[i]["metadata"]["rule"][0] for i in range(num_trials)]
            ),
            "t_s": np.array(
                [batch[i]["metadata"]["t_s"][0] for i in range(num_trials)]
            ),
            "t_m": np.array(
                [batch[i]["metadata"]["t_m"][0] for i in range(num_trials)]
            ),
            "stim_direction": np.array(
                [batch[i]["metadata"]["stim_direction"][0] for i in range(num_trials)]
            ),
        }

        # State for interactive navigation
        state = {"current_trial": 0, "fig": None}

        def plot_trial(trial_idx):
            """Plot a single trial with detailed information."""
            # Extract trial-specific data
            trial_inputs = batch[trial_idx]["inputs"][0].numpy()  # [T, 5]
            trial_targets = batch[trial_idx]["targets"][0].numpy()  # [T, 2]
            trial_loss_mask = batch[trial_idx]["loss_mask"][0].numpy()  # [T, 2]
            trial_eval_mask = batch[trial_idx]["eval_mask"][0].numpy()  # [T, 2]

            # Use task's create_trial_figure with block overview (reuse existing figure)
            fig = task.create_trial_figure(
                inputs=trial_inputs,
                outputs=trial_targets,  # No model, so outputs = targets
                targets=trial_targets,
                eval_mask=trial_eval_mask,
                trial_idx=trial_idx,
                batch=batch,
                batch_idx=0,
                loss_mask=trial_loss_mask,
                block_overview=True,
                batch_metadata=batch_metadata,
                fig=state["fig"],  # Reuse existing figure
            )

            fig.suptitle("Use ← → arrow keys to navigate trials", fontsize=12, y=0.995)

            # Connect keyboard handler on first call
            if state["fig"] is None:
                fig.canvas.mpl_connect("key_press_event", on_key)

            state["fig"] = fig
            fig.canvas.draw()

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == "right":
                state["current_trial"] = min(state["current_trial"] + 1, num_trials - 1)
                plot_trial(state["current_trial"])
            elif event.key == "left":
                state["current_trial"] = max(state["current_trial"] - 1, 0)
                plot_trial(state["current_trial"])

        # Initial plot
        plot_trial(0)

    plt.show()


def demo_transition_task(num_sequences: int, discrete: bool, rule_cue_prob: float):
    """Demo the TransitionTask with interactive navigation.

    TransitionTask allows adjustable rule_cue_prob to interpolate between
    fully instructed (rule_cue_prob=1.0) and fully inferred (rule_cue_prob=0.0).
    """

    task = TransitionTask(
        dt=20.0,
        pulse_width=50.0,
        decision_threshold=850.0,
        delta_t_min=530.0,
        delta_t_max=1170.0,
        fixation_delay_min=400.0,
        fixation_delay_max=900.0,
        rule_report_period=700.0,
        response_period=700.0,
        grace_period=400.0,
        input_noise_std=0.1,
        inter_trial_interval=1500.0,
        reward_duration=500.0,
        block_min=10,
        block_mean=6,
        rule_cue_prob=rule_cue_prob,
        trials_per_sequence=40,
        discrete_eval=discrete,
    )

    for _ in range(num_sequences):
        # New batch structure: list of trial dicts
        batch = task.generate_batch(batch_size=1)

        # batch is a list where each element is a trial dict
        num_trials = len(batch)

        # Extract metadata from trials
        batch_metadata = {
            "rule": np.array(
                [batch[i]["metadata"]["rule"][0] for i in range(num_trials)]
            ),
            "has_instruction": np.array(
                [batch[i]["metadata"]["has_instruction"][0] for i in range(num_trials)]
            ),
            "t_s": np.array(
                [batch[i]["metadata"]["t_s"][0] for i in range(num_trials)]
            ),
            "t_m": np.array(
                [batch[i]["metadata"]["t_m"][0] for i in range(num_trials)]
            ),
            "stim_direction": np.array(
                [batch[i]["metadata"]["stim_direction"][0] for i in range(num_trials)]
            ),
        }

        # State for interactive navigation
        state = {"current_trial": 0, "fig": None}

        def plot_trial(trial_idx):
            """Plot a single trial with detailed information."""
            # Extract trial-specific data
            trial_inputs = batch[trial_idx]["inputs"][0].numpy()  # [T, 5]
            trial_targets = batch[trial_idx]["targets"][0].numpy()  # [T, 2]
            trial_loss_mask = batch[trial_idx]["loss_mask"][0].numpy()  # [T, 2]
            trial_eval_mask = batch[trial_idx]["eval_mask"][0].numpy()  # [T, 2]

            # Use task's create_trial_figure with block overview (reuse existing figure)
            fig = task.create_trial_figure(
                inputs=trial_inputs,
                outputs=trial_targets,  # No model, so outputs = targets
                targets=trial_targets,
                eval_mask=trial_eval_mask,
                trial_idx=trial_idx,
                batch=batch,
                batch_idx=0,
                loss_mask=trial_loss_mask,
                block_overview=True,
                batch_metadata=batch_metadata,
                fig=state["fig"],  # Reuse existing figure
            )

            fig.suptitle(
                f"TransitionTask (rule_cue_prob={rule_cue_prob:.2f}) - Use ← → arrow keys to navigate trials",
                fontsize=12,
                y=0.995,
            )

            # Connect keyboard handler on first call
            if state["fig"] is None:
                fig.canvas.mpl_connect("key_press_event", on_key)

            state["fig"] = fig
            fig.canvas.draw()

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == "right":
                state["current_trial"] = min(state["current_trial"] + 1, num_trials - 1)
                plot_trial(state["current_trial"])
            elif event.key == "left":
                state["current_trial"] = max(state["current_trial"] - 1, 0)
                plot_trial(state["current_trial"])

        # Initial plot
        plot_trial(0)

    plt.show()


def demo_rnn_processing():
    """Demonstrate timestep-by-timestep RNN processing with ITI and reward feedback.

    Uses the same forward pass logic as in training (via BaseTrainer._forward_pass)
    to ensure consistency.

    Shows:
    - Single timestep forward pass: [B, 5] -> [B, 2]
    - Hidden state evolution across multiple trials with ITI
    - Reward feedback processing during ITI (as in training)
    """
    print("\n" + "=" * 70)
    print("RNN TIMESTEP-BY-TIMESTEP PROCESSING DEMO")
    print("=" * 70 + "\n")

    # Create sequence task with 3 trials and RNN model
    task = InstructedTask(
        dt=20.0,
        input_noise_std=0.1,
        trials_per_sequence=3,
        inter_trial_interval=1000.0,
        reward_duration=300.0,
    )
    model = RNN(input_size=5, hidden_size=128, output_size=2, dt=20.0, tau=100.0)
    model.eval()

    # Generate a batch with 1 sequence
    batch = task.generate_batch(batch_size=1)
    num_trials = len(batch)

    # Convert batch to float32
    for trial_dict in batch:
        for key in ["inputs", "targets", "loss_mask", "eval_mask"]:
            if key in trial_dict:
                trial_dict[key] = trial_dict[key].float()

    H = model.hidden_size
    print(f"Sequence with {num_trials} trials, hidden_size={H}\n")

    # Create trainer instance to use its forward pass method
    # Provide minimal config with wandb disabled (no files saved)
    demo_config = {
        "wandb": {
            "project": "timing_context",
            "run_name": "demo_rnn_processing",
        }
    }

    # Temporarily set wandb to disabled mode
    os.environ["WANDB_MODE"] = "disabled"
    trainer = BaseTrainer(model=model, log_dir="logs/temp", config=demo_config)

    # Use trainer's forward pass with return_hidden=True to get everything in one pass
    with torch.no_grad():
        (
            all_outputs,
            all_inputs_list,
            all_targets_list,
            all_eval_masks,
            all_loss_masks,
            all_hidden,
            iti_regions,
            trial_boundaries,
            all_iti_inputs,
            all_iti_targets,
            all_iti_outputs,
        ) = trainer._forward_pass(batch, task, include_iti=True, return_hidden=True)

    # Interleave trial and ITI data for plotting
    # all_inputs_list contains only trial inputs, all_iti_inputs contains ITI inputs
    # Need to interleave them based on trial_boundaries
    combined_inputs = []
    combined_targets = []
    combined_loss_masks = []

    for trial_idx in range(num_trials):
        combined_inputs.append(all_inputs_list[trial_idx])
        combined_targets.append(all_targets_list[trial_idx])
        combined_loss_masks.append(batch[trial_idx]["loss_mask"])
        # Add ITI after each trial except the last
        if trial_idx < len(all_iti_inputs):
            combined_inputs.append(all_iti_inputs[trial_idx])
            combined_targets.append(all_iti_targets[trial_idx])
            # ITI has zero loss mask
            iti_len = all_iti_inputs[trial_idx].shape[1]
            iti_loss_mask = torch.zeros(1, iti_len, 2)
            combined_loss_masks.append(iti_loss_mask)

    # Concatenate and convert to numpy
    all_inputs_concat = torch.cat(combined_inputs, dim=1)[0].numpy()  # [T_total, 5]
    all_targets_concat = torch.cat(combined_targets, dim=1)[0].numpy()  # [T_total, 2]
    all_loss_masks_concat = torch.cat(combined_loss_masks, dim=1)[
        0
    ].numpy()  # [T_total, 2]
    all_hidden_states = torch.stack(all_hidden, dim=0)[:, 0, :].numpy()  # [T_total, H]

    T_total = all_inputs_concat.shape[0]
    time_ms = np.arange(T_total) * task.dt

    # Identify regions where loss_mask is 0 for both outputs (padded regions)
    zero_mask_regions = []
    in_zero_region = False
    region_start = None
    for t in range(T_total):
        is_zero = (all_loss_masks_concat[t, 0] == 0) and (
            all_loss_masks_concat[t, 1] == 0
        )
        if is_zero and not in_zero_region:
            region_start = t
            in_zero_region = True
        elif not is_zero and in_zero_region:
            zero_mask_regions.append((region_start, t))
            in_zero_region = False
    # Close final region if still open
    if in_zero_region:
        zero_mask_regions.append((region_start, T_total))

    # Select first 10 hidden dimensions to plot
    hidden_dims_to_plot = list(range(10))  # First 10 dimensions

    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    fig.suptitle(
        f"RNN Processing Across {num_trials} Trials with ITI",
        fontsize=14,
        fontweight="bold",
    )

    # Mark regions where loss_mask=0 (ITI + padding) on all plots
    for region_start, region_end in zero_mask_regions:
        for ax in axes:
            # Compute time coordinates directly to handle region_end = T_total
            start_time = region_start * task.dt
            end_time = region_end * task.dt
            ax.axvspan(start_time, end_time, color="lightgray", alpha=0.3, linewidth=0)

    # Mark trial boundaries
    for boundary in trial_boundaries[1:-1]:
        for ax in axes:
            ax.axvline(
                time_ms[boundary], color="black", linestyle=":", alpha=0.4, linewidth=1
            )

    # Plot 1: Inputs
    ax = axes[0]
    ax.plot(
        time_ms,
        all_inputs_concat[:, 0],
        label="Center Fixation",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.plot(
        time_ms,
        all_inputs_concat[:, 1],
        label="Horizontal Cue",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.plot(
        time_ms, all_inputs_concat[:, 2], label="Rule Cue", linewidth=1.5, alpha=0.6
    )
    ax.plot(
        time_ms, all_inputs_concat[:, 3], label="Vertical Cue", linewidth=1.5, alpha=0.6
    )
    ax.plot(
        time_ms, all_inputs_concat[:, 4], label="Reward Cue", linewidth=2, alpha=0.8
    )
    ax.set_ylabel("Input Value")
    ax.set_title("Inputs (5 channels, gray=loss_mask=0)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)

    # Plot 2: Outputs vs Targets
    # Recompute outputs from hidden states for plotting (already have them)
    # For simplicity, just show targets since we processed everything above
    ax = axes[1]
    ax.plot(
        time_ms,
        all_targets_concat[:, 0],
        "b-",
        linewidth=2,
        label="Horizontal Eye (target)",
        alpha=0.7,
    )
    ax.plot(
        time_ms,
        all_targets_concat[:, 1],
        "r-",
        linewidth=2,
        label="Vertical Eye (target)",
        alpha=0.7,
    )
    ax.set_ylabel("Target Value")
    ax.set_title("Target Outputs", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Hidden state dimensions (first 10)
    ax = axes[2]
    cmap = plt.cm.tab10
    for i, dim_idx in enumerate(hidden_dims_to_plot):
        ax.plot(
            time_ms,
            all_hidden_states[:, dim_idx],
            label=f"h{dim_idx}",
            linewidth=1.2,
            alpha=0.7,
            color=cmap(i),
        )
    ax.set_ylabel("Hidden Value")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Hidden State Evolution (first 10 dimensions)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Clean up temporary trainer
    trainer.close()

    plt.show()


def main(argv):
    """Plot stimulus and desired response for task trials."""
    if len(argv) < 2:
        print("Usage: python demo_task.py <task_type> [options]")
        print("Available tasks: single_trial, instructed, inferred, transition, rnn")
        return

    task_type = argv[1]

    if task_type == "single_trial":
        print(f"Demonstrating SingleTrialTask with {FLAGS.num_trials} trials...")
        demo_instructed_timing_task(FLAGS.num_trials, FLAGS.discrete)
    elif task_type == "instructed":
        print(f"Demonstrating InstructedTask with {FLAGS.num_trials} sequences...")
        demo_instructed_task(FLAGS.num_trials, FLAGS.discrete)
    elif task_type == "inferred":
        print(f"Demonstrating InferredTask with {FLAGS.num_trials} sequences...")
        demo_inferred_task(FLAGS.num_trials, FLAGS.discrete)
    elif task_type == "transition":
        print(
            f"Demonstrating TransitionTask with {FLAGS.num_trials} sequences (rule_cue_prob={FLAGS.rule_cue_prob})..."
        )
        demo_transition_task(FLAGS.num_trials, FLAGS.discrete, FLAGS.rule_cue_prob)
    elif task_type == "rnn":
        demo_rnn_processing()
    else:
        print(f"Unknown task type: {task_type}")
        print("Available tasks: single_trial, instructed, inferred, transition, rnn")


if __name__ == "__main__":
    app.run(main)
