"""Sequential trainer - trains tasks one after another."""

import torch.optim as optim
import wandb
from pathlib import Path
from typing import Optional

from models import Model
from tasks import BaseTask
from trainers.utils import BaseTrainer


class SequentialTrainer(BaseTrainer):
    """Trainer that trains multiple tasks sequentially."""

    def __init__(
        self,
        model: Model,
        tasks: list[BaseTask],
        task_names: list[str],
        task_num_steps: list[int],
        task_eval_flags: Optional[list[bool]] = None,
        task_param_schedules: Optional[list[Optional[dict]]] = None,
        reset_optimizer_between_tasks: bool = False,
        num_eval_samples: int = 100,
        total_steps: Optional[int] = None,  # Ignored, here for config compatibility
        learning_rate: float = 1e-3,  # Kept explicit for reset_optimizer_between_tasks
        **kwargs,
    ):
        if len(tasks) != len(task_names) != len(task_num_steps):
            raise ValueError(
                "tasks, task_names, and task_num_steps must have same length"
            )

        # Default all tasks to eval=True if not specified
        if task_eval_flags is None:
            task_eval_flags = [True] * len(tasks)
        if len(task_eval_flags) != len(tasks):
            raise ValueError("task_eval_flags must have same length as tasks")

        # Parameter scheduling
        if task_param_schedules is None:
            task_param_schedules = [None] * len(tasks)
        if len(task_param_schedules) != len(tasks):
            raise ValueError("task_param_schedules must have same length as tasks")

        # Pass common training parameters to base via kwargs
        super().__init__(model=model, learning_rate=learning_rate, **kwargs)

        self.tasks = tasks
        self.task_names = task_names
        self.task_num_steps = task_num_steps
        self.task_eval_flags = task_eval_flags
        self.task_param_schedules = task_param_schedules
        self.reset_optimizer_between_tasks = reset_optimizer_between_tasks
        self.learning_rate = learning_rate

        # Sequential training state
        self.current_task_idx = 0
        self.current_task_step = 0
        self.started_from_checkpoint = False  # Track if we loaded from checkpoint

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

    def _update_task_parameter(self) -> None:
        """Update task parameter based on schedule for current task."""
        schedule = self.task_param_schedules[self.current_task_idx]
        if schedule is None:
            return

        # Extract schedule parameters
        param_name = schedule["param_name"]
        start_value = schedule["start_value"]
        end_value = schedule["end_value"]
        num_steps = self.task_num_steps[self.current_task_idx]

        # Compute linear interpolation
        progress = self.current_task_step / num_steps
        current_value = start_value + progress * (end_value - start_value)

        # Update task parameter
        task = self.tasks[self.current_task_idx]
        setattr(task, param_name, current_value)

    def eval(self, loss: Optional[float] = None) -> None:
        """Evaluate all tasks with eval=True, log scalars and figures."""
        all_metrics = {}
        for task_idx, (task, task_name, eval_batch, eval_flag) in enumerate(
            zip(self.tasks, self.task_names, self.eval_batches, self.task_eval_flags)
        ):
            if not eval_flag:
                continue

            # Compute accuracies and optionally log trial figures
            task_metrics = self.eval_task(
                task, eval_batch, log_figures=True, task_name=task_name, num_trials=1
            )

            # Add task prefix
            for key, value in task_metrics.items():
                all_metrics[f"{task_name}/{key}"] = value

        # Add loss if provided
        if loss is not None:
            all_metrics["loss"] = loss

        # Add current task index
        all_metrics["train/current_task_idx"] = self.current_task_idx

        # Log scheduled parameter value if exists
        schedule = self.task_param_schedules[self.current_task_idx]
        if schedule is not None:
            param_name = schedule["param_name"]
            task = self.tasks[self.current_task_idx]
            param_value = getattr(task, param_name)
            all_metrics[f"train/{param_name}"] = param_value

        # Print evaluation metrics
        print(
            f"Evaluation after step {self.step}: "
            + ", ".join([f"{k}={v:.4f}" for k, v in all_metrics.items()])
        )

        # Log metrics to wandb
        wandb.log(all_metrics, step=self.step)

        # Save latest checkpoint
        extra_state = {
            "current_task_idx": self.current_task_idx,
            "current_task_step": self.current_task_step,
        }
        self.save_checkpoint(filename="latest.pt", extra_state=extra_state)

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> None:
        """Save model checkpoint with curriculum state."""
        extra_state = {
            "current_task_idx": self.current_task_idx,
            "current_task_step": self.current_task_step,
        }

        super().save_checkpoint(filename=filename, extra_state=extra_state)

    def load_checkpoint(
        self, path: str | Path = None, reset_counters: bool = False
    ) -> dict:
        """Load model checkpoint and restore curriculum state."""
        extra_state = super().load_checkpoint(path, reset_counters)

        # Reset or restore curriculum counters
        if reset_counters:
            self.current_task_idx = 0
            self.current_task_step = 0
            self.started_from_checkpoint = True
        else:
            if "current_task_idx" in extra_state:
                self.current_task_idx = extra_state["current_task_idx"]
            if "current_task_step" in extra_state:
                self.current_task_step = extra_state["current_task_step"]

        return extra_state

    def train(
        self,
    ) -> None:
        """Main training loop for sequential multi-task training.

        Args:
            batch_size: Batch size for each task
            log_interval: Steps between logging
            checkpoint_interval: Steps between checkpoints
        """
        total_steps = sum(self.task_num_steps)

        # Calculate remaining steps if resuming
        if self.step > 0 and not self.started_from_checkpoint:
            # Resuming - calculate remaining steps
            steps_remaining = total_steps - self.step
            final_step = total_steps
            print(
                f"\nResuming sequential multi-task training for {steps_remaining} remaining steps (step {self.step} → {final_step})..."
            )
        elif self.started_from_checkpoint:
            print("\nStarting training from checkpoint initialization...")
            print(f"Model weights loaded, training {total_steps} steps from scratch")
        else:
            print(
                f"\nStarting sequential multi-task training for {total_steps} total steps..."
            )

        print(f"Tasks: {self.task_names}")
        print(f"Steps per task: {self.task_num_steps}")
        print(f"Reset optimizer between tasks: {self.reset_optimizer_between_tasks}")
        print(f"Log directory: {self.log_dir}")

        # Save initial task/step for resuming (if applicable)
        initial_task_idx = self.current_task_idx
        initial_task_step = self.current_task_step

        # Train each task sequentially, starting from current task if resuming
        for task_idx in range(initial_task_idx, len(self.tasks)):
            task_name = self.task_names[task_idx]
            num_steps = self.task_num_steps[task_idx]

            # Update current task index
            self.current_task_idx = task_idx

            # Determine starting step within this task
            if task_idx == initial_task_idx:
                # Resuming this task - start from where we left off
                starting_step = initial_task_step
                self.current_task_step = initial_task_step
            else:
                # New task - start from beginning
                starting_step = 0
                self.current_task_step = 0

                # Reset optimizer if configured to do so
                if self.reset_optimizer_between_tasks:
                    print(f"Resetting optimizer for task {task_name}")
                    self.optimizer = optim.Adam(
                        self.model.parameters(), lr=self.learning_rate
                    )

            print(f"\n{'=' * 60}")
            print(f"Starting task {task_idx + 1}/{len(self.tasks)}: {task_name}")
            if starting_step > 0:
                print(f"Resuming from step {starting_step}/{num_steps}...")
            else:
                print(f"Training for {num_steps} steps...")

            # Print schedule info if exists
            schedule = self.task_param_schedules[task_idx]
            if schedule is not None:
                print(
                    f"Parameter schedule: {schedule['param_name']} from {schedule['start_value']:.3f} to {schedule['end_value']:.3f}"
                )

            print(f"{'=' * 60}\n")

            loss = None  # Initialize for first eval
            for local_step in range(starting_step, num_steps):
                # Evaluation and logging (before training step)
                if self.current_task_step % self.log_interval == 0:
                    self.eval(loss)
                    print(
                        f"Task {task_idx + 1}/{len(self.tasks)} [{task_name}] "
                        f"Step {self.current_task_step}/{num_steps} "
                        f"(Global: {self.step}/{total_steps})"
                        + (f": loss={loss:.4f}" if loss is not None else "")
                    )

                # Training step
                self.model.train()
                self._update_task_parameter()
                task = self.tasks[self.current_task_idx]
                batch = task.generate_batch(self.batch_size)
                loss = self.train_step(batch, task)

                # Update counters
                self.step += 1
                self.current_task_step += 1

                # Checkpointing
                if self.step % self.checkpoint_interval == 0:
                    self.save_checkpoint(f"{self.step}.pt")

        print("\nTraining complete!")

        self.close()
