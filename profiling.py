"""Vibe-coded profiling script for analyzing training performance.
Use --batch-sweep to compare different batch sizes.

Usage:
    python profiling.py configs/instructed.yaml
    python profiling.py configs/instructed.yaml --steps 3 --batch-sweep
"""

import argparse
import time
from collections import defaultdict

from omegaconf import OmegaConf

from main import create_model, create_trainer, set_random_seed


def print_batch_comparison(batch_sizes, all_results):
    """Print comparison table for batch sweep."""
    print("\n" + "=" * 115)
    print("Batch Size Comparison (Time in ms)")
    print("=" * 115)

    # Header
    header = f"{'Section':<35}"
    for bs in batch_sizes:
        header += f" | {f'B={bs}':>10}"
    print(header)
    print("-" * 115)

    # Eval row (at top, happens every ~20 steps)
    row = f"{'eval':<35}"
    for bs in batch_sizes:
        times = all_results[bs]
        time_val = times.get("eval", 0)
        row += f" | {time_val * 1000:>10.1f}"
    print(row)
    print("-" * 115)

    # Section names to display
    sections = [
        ("generate_batch", 0),
        ("train_step", 0),
        ("forward_pass", 1),
        ("trial_processing", 2),
        ("iti_generation", 2),
        ("iti_forward", 2),
        ("loss_computation", 1),
        ("backward", 1),
        ("gradient_clipping", 1),
        ("optimizer_step", 1),
        ("logging", 1),
    ]

    for section_name, indent in sections:
        # Calculate train_step total for each batch size
        if section_name == "train_step":
            display_name = "  " * indent + section_name
            row = f"{display_name:<35}"
            for bs in batch_sizes:
                times = all_results[bs]
                train_step_total = (
                    times["forward_pass"]
                    + times["loss_computation"]
                    + times["backward"]
                    + times["gradient_clipping"]
                    + times["optimizer_step"]
                    + times["logging"]
                )
                row += f" | {train_step_total * 1000:>10.1f}"
            print(row)
        else:
            # Skip if section doesn't exist or is 0 for all batch sizes
            if not any(all_results[bs].get(section_name, 0) > 0 for bs in batch_sizes):
                continue

            display_name = "  " * indent + section_name
            row = f"{display_name:<35}"
            for bs in batch_sizes:
                times = all_results[bs]
                time_val = times.get(section_name, 0)
                row += f" | {time_val * 1000:>10.1f}"
            print(row)

    # Total step row (excludes eval)
    print("-" * 115)
    row = f"{'Total step':<35}"
    totals = {}
    for bs in batch_sizes:
        times = all_results[bs]
        # Calculate total step time (excludes eval)
        total = (
            times["generate_batch"]
            + times["forward_pass"]
            + times["loss_computation"]
            + times["backward"]
            + times["gradient_clipping"]
            + times["optimizer_step"]
            + times["logging"]
        )
        totals[bs] = total
        row += f" | {total * 1000:>10.1f}"
    print(row)

    # Throughput row (samples/sec, excludes eval)
    print("=" * 115)
    row = f"{'Throughput (samples/sec, excl eval)':<35}"
    for bs in batch_sizes:
        throughput = bs / totals[bs]
        row += f" | {throughput:>10.1f}"
    print(row)


def print_single_batch_summary(times, num_steps):
    """Print summary table for single batch profiling."""
    # Calculate total correctly (only top-level components)
    total_time = (
        times["generate_batch"]
        + times["forward_pass"]
        + times["loss_computation"]
        + times["backward"]
        + times["gradient_clipping"]
        + times["optimizer_step"]
        + times["logging"]
        + times["eval"]
    )

    print(f"\nProfiling complete! Total time: {total_time * 1000:.1f}ms\n")

    # Calculate train_step total
    train_step_total = (
        times["forward_pass"]
        + times["loss_computation"]
        + times["backward"]
        + times["gradient_clipping"]
        + times["optimizer_step"]
        + times["logging"]
    )

    # Print results
    print("=" * 80)
    print("CPU Time Summary")
    print("=" * 80)
    print(f"{'Section':<40} {'Time (ms)':>12} {'%':>8} {'Avg/call (ms)':>15}")
    print("-" * 80)

    def print_section(name, indent=0):
        if times[name] > 0:
            pct = 100 * times[name] / total_time
            avg = times[name] / num_steps if name != "eval" else times[name]
            display_name = "  " * indent + name
            print(
                f"{display_name:<40} {times[name] * 1000:>12.1f} {pct:>7.1f}% {avg * 1000:>15.1f}"
            )

    # Top-level sections
    print_section("generate_batch")

    # Train step with breakdown
    if train_step_total > 0:
        pct = 100 * train_step_total / total_time
        avg = train_step_total / num_steps
        print(
            f"{'train_step':<40} {train_step_total * 1000:>12.1f} {pct:>7.1f}% {avg * 1000:>15.1f}"
        )
        print_section("forward_pass", indent=1)
        print_section("trial_processing", indent=2)
        print_section("iti_generation", indent=2)
        print_section("iti_forward", indent=2)
        print_section("loss_computation", indent=1)
        print_section("backward", indent=1)
        print_section("gradient_clipping", indent=1)
        print_section("optimizer_step", indent=1)
        print_section("logging", indent=1)

    print_section("eval")

    print("-" * 80)
    print(f"{'Total':<40} {total_time * 1000:>12.1f} {100.0:>7.1f}%")


def profile_batch_size(trainer, task, batch_size, num_steps):
    """Profile training for a specific batch size.

    Returns dict of timings for each section.
    """
    import torch

    times = defaultdict(float)

    for step in range(num_steps):
        # Time batch generation
        t0 = time.perf_counter()
        batch = task.generate_batch(batch_size)
        times["generate_batch"] += time.perf_counter() - t0

        # Break down train_step into components
        # Forward pass - manual breakdown to time trials vs ITI
        t0_forward = time.perf_counter()

        N = len(batch)
        B = batch[0]["inputs"].shape[0]
        h = torch.zeros(B, trainer.model.hidden_size)
        all_outputs = []

        for trial_idx, trial in enumerate(batch):
            # Time trial processing
            t0 = time.perf_counter()
            trial_inputs = trial["inputs"]
            trial_lengths = trial["trial_lengths"]
            max_trial_len = trial_inputs.shape[1]

            trial_outputs = []
            for t in range(max_trial_len):
                still_active = (t < trial_lengths).unsqueeze(-1).float()
                input_t = trial_inputs[:, t, :]
                output_t, h_new = trainer.model(input_t, h)
                h = still_active * h_new + (1 - still_active) * h
                trial_outputs.append(output_t)

            trial_outputs = torch.stack(trial_outputs, dim=1)
            all_outputs.append(trial_outputs)
            times["trial_processing"] += time.perf_counter() - t0

            # ITI processing
            if trial_idx < N - 1 and hasattr(task, "_generate_iti_inputs"):
                # Time ITI generation
                t0 = time.perf_counter()
                is_correct = task._evaluate_trial_correctness_batch(
                    trial_outputs, trial["targets"], trial["eval_mask"]
                )
                iti_inputs = task._generate_iti_inputs(
                    is_correct, trial["metadata"], task.iti_len, task.reward_len
                )
                times["iti_generation"] += time.perf_counter() - t0

                # Time ITI forward
                t0 = time.perf_counter()
                for t in range(iti_inputs.shape[1]):
                    output_t, h = trainer.model(iti_inputs[:, t, :], h)
                times["iti_forward"] += time.perf_counter() - t0

        times["forward_pass"] += time.perf_counter() - t0_forward

        # Loss computation
        t0 = time.perf_counter()
        all_targets = [trial["targets"] for trial in batch]
        all_masks = [trial["loss_mask"] for trial in batch]
        outputs = torch.cat(all_outputs, dim=1)
        targets = torch.cat(all_targets, dim=1)
        masks = torch.cat(all_masks, dim=1)
        mse = (outputs - targets) ** 2
        weighted_mse = mse * masks
        loss = weighted_mse.sum() / (masks.sum() + 1e-8)
        times["loss_computation"] += time.perf_counter() - t0

        # Backward
        t0 = time.perf_counter()
        trainer.optimizer.zero_grad()
        loss.backward()
        times["backward"] += time.perf_counter() - t0

        # Gradient clipping
        if trainer.clip_grad_norm is not None:
            t0 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), trainer.clip_grad_norm
            )
            times["gradient_clipping"] += time.perf_counter() - t0

        # Optimizer step
        t0 = time.perf_counter()
        trainer.optimizer.step()
        times["optimizer_step"] += time.perf_counter() - t0

        # Logging (gradient/weight norms + wandb)
        t0 = time.perf_counter()
        import wandb

        wandb_metrics = {}
        total_grad_norm = 0.0
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_norm**2
                wandb_metrics[f"train/grad_norm/{name}"] = param_norm
        total_grad_norm = total_grad_norm**0.5
        wandb_metrics["train/grad_norm/total"] = total_grad_norm
        wandb_metrics["loss"] = loss.item()
        for name, param in trainer.model.named_parameters():
            param_norm = param.data.norm(2).item()
            wandb_metrics[f"train/weight_norm/{name}"] = param_norm
        wandb.log(wandb_metrics, step=trainer.step)
        times["logging"] += time.perf_counter() - t0

        trainer.step += 1

    t0 = time.perf_counter()
    trainer.eval(task_idx=0, loss=loss)
    times["eval"] += time.perf_counter() - t0

    return times


def main():
    parser = argparse.ArgumentParser(description="Profile training performance")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of steps to profile (default: 1)"
    )
    parser.add_argument(
        "--batch-sweep", action="store_true", help="Sweep batch sizes from 32 to 256"
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    conf = OmegaConf.load(args.config)

    # Disable wandb for profiling (zero overhead)
    import os

    os.environ["WANDB_MODE"] = "disabled"

    # Ensure wandb config exists (required by trainer, but will be disabled)
    if "wandb" not in conf:
        conf["wandb"] = {"project": "profiling"}

    # Override total_steps to match profiling steps
    conf["training"]["total_steps"] = args.steps

    # Set random seed
    random_seed = conf.get("random_seed", 42)
    set_random_seed(random_seed)

    if args.batch_sweep:
        # Sweep batch sizes
        batch_sizes = [32, 64, 128, 256, 512]
        all_results = {}

        for batch_size in batch_sizes:
            print(f"\nProfiling batch size {batch_size} ({args.steps} steps)...")

            # Create fresh model and trainer for each batch size
            set_random_seed(random_seed)  # Reset seed for consistent initialization
            model = create_model(conf)
            trainer = create_trainer(conf, model, log_dir="logs/profiler")
            task = trainer.tasks[0]

            times = profile_batch_size(trainer, task, batch_size, args.steps)
            all_results[batch_size] = times

        # Print comparison table
        print_batch_comparison(batch_sizes, all_results)

    else:
        # Single batch size profiling
        # Create model and trainer
        print("Creating model and trainer...")
        model = create_model(conf)
        trainer = create_trainer(conf, model, log_dir="logs/profiler")
        task = trainer.tasks[0]

        print(f"\nProfiling {args.steps} training steps...")
        print(f"Tasks: {trainer.task_names}")
        print(f"Batch size: {trainer.batch_size}\n")

        times = profile_batch_size(trainer, task, trainer.batch_size, args.steps)

        # Print summary table
        print_single_batch_summary(times, args.steps)


if __name__ == "__main__":
    main()
