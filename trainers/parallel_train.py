"""Parallel trainer - interleaves batches from multiple tasks."""

import numpy as np
import wandb
from pathlib import Path
from typing import Optional

from models import Model
from tasks import BaseTask
from trainers.utils import BaseTrainer


class ParallelTrainer(BaseTrainer):
    """Trainer that interleaves batches from multiple tasks.

    Each training step randomly samples one task according to specified
    weights, generates a batch from that task, and trains the model on it.
    Can also be used for single-task training by providing one task with weight 1.0.
    """

    def __init__(
        self,
        model: Model,
        tasks: list[BaseTask],
        task_names: list[str],
        task_weights: list[float],
        task_eval_flags: Optional[list[bool]] = None,
        num_eval_samples: int = 100,
        total_steps: int = 10000,
        **kwargs,
    ):
        if len(tasks) != len(task_names) != len(task_weights):
            raise ValueError(
                "tasks, task_names, and task_weights must have same length"
            )

        if abs(sum(task_weights) - 1.0) > 1e-2:
            raise ValueError("task_weights must approx. sum to 1.0")

        # Default all tasks to eval=True if not specified
        if task_eval_flags is None:
            task_eval_flags = [True] * len(tasks)
        if len(task_eval_flags) != len(tasks):
            raise ValueError("task_eval_flags must have same length as tasks")

        # Pass common training parameters to base via kwargs
        super().__init__(model=model, **kwargs)

        self.tasks = tasks
        self.task_names = task_names
        self.task_weights = np.array(task_weights)
        self.task_eval_flags = task_eval_flags
        self.task_step_counts = [0] * len(tasks)
        self.total_steps = total_steps
        self.multi_task = len(self.tasks) > 1

        # Validate dt consistency between model and tasks
        assert all(task.dt == model.dt for task in tasks), (
            f"All tasks must have same dt as model (model.dt={model.dt})"
        )

        # Generate fixed eval batches for each task
        self.eval_batches = [task.generate_batch(num_eval_samples) for task in tasks]

        # Define summary metrics for accuracy (max)
        for task_name in task_names:
            wandb.define_metric(f"{task_name}/decision_accuracy", summary="max")
            wandb.define_metric(f"{task_name}/rule_accuracy", summary="max")

    def eval(self, task_idx: int, loss: Optional[float] = None) -> None:
        """Evaluate all tasks with eval=True, log scalars and figures.

        Args:
            task_idx: Index of task that was trained
            loss: Training loss to log
        """
        # Evaluate all tasks on fixed eval batches (only those with eval=True)
        all_metrics = {}

        # Add loss if provided
        if loss is not None:
            all_metrics["loss"] = loss

        for eval_task_idx, (task, task_name, eval_batch, eval_flag) in enumerate(
            zip(self.tasks, self.task_names, self.eval_batches, self.task_eval_flags)
        ):
            if not eval_flag:
                continue

            # Compute accuracies and log trial figures
            task_metrics = self.eval_task(
                task, eval_batch, log_figures=True, task_name=task_name, num_trials=2
            )

            # Always add task prefix for consistency
            for key, value in task_metrics.items():
                all_metrics[f"{task_name}/{key}"] = value

        # Add task sampling counts for multi-task
        if self.multi_task:
            for name, count in zip(self.task_names, self.task_step_counts):
                all_metrics[f"task_sampling/{name}_count"] = count

        print(
            f"Evaluation after step {self.step}: "
            + ", ".join([f"{k}={v:.4f}" for k, v in all_metrics.items()])
        )

        # Log metrics to wandb
        wandb.log(all_metrics, step=self.step)

        # Save latest checkpoint
        self.save_checkpoint(filename="latest.pt")

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> None:
        """Save model checkpoint with task sampling statistics."""
        super().save_checkpoint(
            filename=filename, extra_state={"task_step_counts": self.task_step_counts}
        )

    def load_checkpoint(
        self, path: str | Path = None, reset_counters: bool = False
    ) -> dict:
        """Load model checkpoint and optionally restore task sampling statistics."""
        extra_state = super().load_checkpoint(path, reset_counters)
        # Only restore task_step_counts if not resetting counters
        if not reset_counters and "task_step_counts" in extra_state:
            self.task_step_counts = extra_state["task_step_counts"]
        return extra_state

    def train(
        self,
    ) -> None:
        """Main training loop for parallel (or single-task) training."""
        # Determine if resuming
        starting_step = self.step
        final_step = starting_step + self.total_steps

        if starting_step > 0:
            print(
                f"Resuming training for {self.total_steps} steps (step {starting_step} → {final_step})..."
            )
        else:
            print(f"Starting training for {self.total_steps} steps...")
        if self.multi_task:
            print(f"Tasks: {self.task_names}")
            print(f"Task weights: {self.task_weights}")
        else:
            print(f"Task: {self.task_names[0]}")

        print(f"Log directory: {self.log_dir}")

        loss = None  # Initialize for first eval
        task_idx = 0  # Initialize for first eval
        for step in range(self.total_steps):
            # Evaluation and logging (before training step)
            if self.step % self.log_interval == 0:
                self.eval(task_idx, loss)
                if self.multi_task:
                    task_name = self.task_names[task_idx] if loss is not None else ""
                    task_str = f" [{task_name}]" if task_name else ""
                    print(
                        f"Step {self.step}/{self.total_steps}{task_str}"
                        + (f": loss={loss:.4f}" if loss is not None else "")
                    )
                else:
                    print(
                        f"Step {self.step}/{self.total_steps}"
                        + (f": loss={loss:.4f}" if loss is not None else "")
                    )

            # Training step
            self.model.train()

            # Sample task, generate batch, and train
            task_idx = np.random.choice(len(self.tasks), p=self.task_weights)
            task = self.tasks[task_idx]
            batch = task.generate_batch(self.batch_size)
            loss = self.train_step(batch, task)

            # Update counters
            self.step += 1
            self.task_step_counts[task_idx] += 1

            # Checkpointing
            if self.step % self.checkpoint_interval == 0:
                self.save_checkpoint(f"{self.step}.pt")

        print("Training complete!")

        if len(self.tasks) > 1:
            print(
                f"Final task step counts: {dict(zip(self.task_names, self.task_step_counts))}"
            )

        self.close()
