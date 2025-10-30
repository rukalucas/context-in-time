"""Parallel trainer - interleaves batches from multiple tasks."""

import numpy as np
from pathlib import Path

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
        learning_rate: float = 1e-3,
        log_dir: str = 'logs',
        num_eval_samples: int = 100,
        batch_size: int = 64,
        log_interval: int = 100,
        checkpoint_interval: int = 1000,
        total_steps: int = 10000,
        optimizer_type: str = 'adam',
        **kwargs
    ):
        if len(tasks) != len(task_names) != len(task_weights):
            raise ValueError("tasks, task_names, and task_weights must have same length")

        if abs(sum(task_weights) - 1.0) > 1e-2:
            raise ValueError("task_weights must approx. sum to 1.0")

        super().__init__(model, learning_rate, log_dir, batch_size, log_interval, checkpoint_interval, optimizer_type)

        self.tasks = tasks
        self.task_names = task_names
        self.task_weights = np.array(task_weights)
        self.task_step_counts = [0] * len(tasks)
        self.total_steps = total_steps
        self.multi_task = len(self.tasks) > 1

        # Generate fixed eval batches for each task
        self.eval_batches = [task.generate_batch(num_eval_samples) for task in tasks]

        # Initialize CSV with all expected fields
        csv_fields = []
        for task, task_name in zip(self.tasks, self.task_names):
            # Get metric names from task class attribute
            for metric_name in task.metric_names:
                if self.multi_task:
                    csv_fields.append(f'{task_name}/{metric_name}')
                else:
                    csv_fields.append(metric_name)
        # Add task sampling fields for multi-task
        if self.multi_task:
            for task_name in self.task_names:
                csv_fields.append(f'task_sampling/{task_name}_count')

        self._init_csv(csv_fields)

    def eval(self, task_idx: int) -> None:
        """Evaluate all tasks, log scalars and figures.

        Args:
            task_idx: Index of task that was trained
        """
        # Evaluate all tasks on fixed eval batches
        all_metrics = {}
        for eval_task_idx, (task, task_name, eval_batch) in enumerate(zip(self.tasks, self.task_names, self.eval_batches)):
            display_name = task_name if self.multi_task else ''

            # Compute accuracies and log trial figures
            task_metrics = self.eval_task(task, eval_batch, log_figures=True,
                                         task_name=display_name, num_trials=2)

            # Add task prefix for multi-task, keep raw for single-task
            if self.multi_task:
                for key, value in task_metrics.items():
                    all_metrics[f'{task_name}/{key}'] = value
            else:
                all_metrics.update(task_metrics)

        # Add task sampling counts for multi-task
        if self.multi_task:
            for name, count in zip(self.task_names, self.task_step_counts):
                all_metrics[f'task_sampling/{name}_count'] = count

        print(f"Evaluation after step {self.step}: " +
              ", ".join([f"{k}={v:.4f}" for k, v in all_metrics.items()]))

        # Log all metrics
        self.log_metrics(all_metrics, self.step)

    def save_checkpoint(self, filename: str = 'checkpoint.pt') -> None:
        """Save model checkpoint with task sampling statistics."""
        super().save_checkpoint(
            filename=filename,
            extra_state={'task_step_counts': self.task_step_counts}
        )

    def load_checkpoint(self, path: str | Path = None, reset_counters: bool = False) -> dict:
        """Load model checkpoint and restore task sampling statistics."""
        extra_state = super().load_checkpoint(path, reset_counters)
        if 'task_step_counts' in extra_state:
            self.task_step_counts = extra_state['task_step_counts']
        return extra_state

    def train(
        self,
    ) -> None:
        """Main training loop for parallel (or single-task) training."""
        if self.multi_task:
            print(f"Starting parallel multi-task training for {self.total_steps} steps...")
            print(f"Tasks: {self.task_names}")
            print(f"Task weights: {self.task_weights}")
        else:
            print(f"Starting training for {self.total_steps} steps...")
            print(f"Task: {self.task_names[0]}")

        print(f"Log directory: {self.log_dir}")

        for step in range(self.total_steps):
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

            # Evaluation and logging
            if (step + 1) % self.log_interval == 0:
                self.eval(task_idx)
                if self.multi_task:
                    task_name = self.task_names[task_idx]
                    print(f"Step {step+1}/{self.total_steps} [{task_name}]: train/loss={loss:.4f}")
                else:
                    print(f"Step {step+1}/{self.total_steps}: train/loss={loss:.4f}")

            # Checkpointing
            if (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{step+1}.pt')

        # Final checkpoint
        self.save_checkpoint('checkpoint_final.pt')
        print("Training complete!")

        if len(self.tasks) > 1:
            print(f"Final task step counts: {dict(zip(self.task_names, self.task_step_counts))}")

        self.close()
