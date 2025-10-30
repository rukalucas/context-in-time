"""Sequential trainer - trains tasks one after another."""

import torch.optim as optim
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
        task_param_schedules: Optional[list[Optional[dict]]] = None,
        learning_rate: float = 1e-3,
        reset_optimizer_between_tasks: bool = False,
        log_dir: str = 'logs',
        num_eval_samples: int = 100,
        batch_size: int = 64,
        log_interval: int = 100,
        checkpoint_interval: int = 1000,
        total_steps: Optional[int] = None,  # Ignored, here for config compatibility
        optimizer_type: str = 'adam',
        **kwargs
    ):
        if len(tasks) != len(task_names) != len(task_num_steps):
            raise ValueError("tasks, task_names, and task_num_steps must have same length")

        # Parameter scheduling
        if task_param_schedules is None:
            task_param_schedules = [None] * len(tasks)
        if len(task_param_schedules) != len(tasks):
            raise ValueError("task_param_schedules must have same length as tasks")

        super().__init__(model, learning_rate, log_dir, batch_size, log_interval, checkpoint_interval, optimizer_type)

        self.tasks = tasks
        self.task_names = task_names
        self.task_num_steps = task_num_steps
        self.task_param_schedules = task_param_schedules
        self.reset_optimizer_between_tasks = reset_optimizer_between_tasks
        self.learning_rate = learning_rate

        # Sequential training state
        self.current_task_idx = 0
        self.current_task_step = 0
        self.started_from_checkpoint = False  # Track if we loaded from checkpoint

        # Generate fixed eval batches for each task
        self.eval_batches = [task.generate_batch(num_eval_samples) for task in tasks]

        # Initialize CSV with all expected fields
        csv_fields = []
        for task, task_name in zip(self.tasks, self.task_names):
            # Get metric names from task class attribute
            for metric_name in task.metric_names:
                csv_fields.append(f'{task_name}/{metric_name}')

        # Add sequential training fields
        csv_fields.append('train/current_task_idx')

        # Add scheduled parameter fields if they exist
        for schedule in self.task_param_schedules:
            if schedule is not None:
                param_name = schedule['param_name']
                field_name = f'train/{param_name}'
                if field_name not in csv_fields:
                    csv_fields.append(field_name)

        self._init_csv(csv_fields)

    def _update_task_parameter(self) -> None:
        """Update task parameter based on schedule for current task."""
        schedule = self.task_param_schedules[self.current_task_idx]
        if schedule is None:
            return

        # Extract schedule parameters
        param_name = schedule['param_name']
        start_value = schedule['start_value']
        end_value = schedule['end_value']
        num_steps = self.task_num_steps[self.current_task_idx]

        # Compute linear interpolation
        progress = self.current_task_step / num_steps
        current_value = start_value + progress * (end_value - start_value)

        # Update task parameter
        task = self.tasks[self.current_task_idx]
        setattr(task, param_name, current_value)

    def eval(self) -> None:
        """Evaluate all tasks, log scalars and figures."""
        # Evaluate all tasks on fixed eval batches
        all_metrics = {}
        for task_idx, (task, task_name, eval_batch) in enumerate(zip(self.tasks, self.task_names, self.eval_batches)):
            # Compute accuracies and optionally log trial figures
            task_metrics = self.eval_task(task, eval_batch, log_figures=True,
                                         task_name=task_name, num_trials=1)

            # Add task prefix
            for key, value in task_metrics.items():
                all_metrics[f'{task_name}/{key}'] = value

        # Add current task index
        all_metrics['train/current_task_idx'] = self.current_task_idx

        # Log scheduled parameter value if exists
        schedule = self.task_param_schedules[self.current_task_idx]
        if schedule is not None:
            param_name = schedule['param_name']
            task = self.tasks[self.current_task_idx]
            param_value = getattr(task, param_name)
            all_metrics[f'train/{param_name}'] = param_value

        # Log all metrics
        self.log_metrics(all_metrics, self.step)

    def save_checkpoint(self, filename: str = 'checkpoint.pt') -> None:
        """Save model checkpoint with curriculum state."""
        extra_state = {
            'current_task_idx': self.current_task_idx,
            'current_task_step': self.current_task_step
        }

        super().save_checkpoint(filename=filename, extra_state=extra_state)

    def load_checkpoint(self, path: str | Path = None, reset_counters: bool = False) -> dict:
        """Load model checkpoint and restore curriculum state."""
        extra_state = super().load_checkpoint(path, reset_counters)

        # Reset or restore curriculum counters
        if reset_counters:
            self.current_task_idx = 0
            self.current_task_step = 0
            self.started_from_checkpoint = True
        else:
            if 'current_task_idx' in extra_state:
                self.current_task_idx = extra_state['current_task_idx']
            if 'current_task_step' in extra_state:
                self.current_task_step = extra_state['current_task_step']

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

        if self.started_from_checkpoint:
            print("\nStarting training from checkpoint initialization...")
            print(f"Model weights loaded, training {total_steps} steps from scratch")
        else:
            print(f"Starting sequential multi-task training for {total_steps} total steps...")

        print(f"Tasks: {self.task_names}")
        print(f"Steps per task: {self.task_num_steps}")
        print(f"Reset optimizer between tasks: {self.reset_optimizer_between_tasks}")
        print(f"Log directory: {self.log_dir}")

        # Train each task sequentially
        for task_idx, (task_name, num_steps) in enumerate(zip(self.task_names, self.task_num_steps)):
            print(f"\n{'='*60}")
            print(f"Starting task {task_idx+1}/{len(self.tasks)}: {task_name}")
            print(f"Training for {num_steps} steps...")

            # Print schedule info if exists
            schedule = self.task_param_schedules[task_idx]
            if schedule is not None:
                print(f"Parameter schedule: {schedule['param_name']} from {schedule['start_value']:.3f} to {schedule['end_value']:.3f}")

            print(f"{'='*60}\n")

            for local_step in range(num_steps):
                # Training step
                self.model.train()
                self._update_task_parameter()
                task = self.tasks[self.current_task_idx]
                batch = task.generate_batch(self.batch_size)
                loss = self.train_step(batch, task)

                # Update counters
                self.step += 1
                self.current_task_step += 1

                # Evaluation and logging
                if (local_step + 1) % self.log_interval == 0:
                    self.eval()
                    print(f"Task {task_idx+1}/{len(self.tasks)} [{task_name}] "
                          f"Step {local_step+1}/{num_steps} "
                          f"(Global: {self.step}/{total_steps}): "
                          f"train/loss={loss:.4f}")

                # Checkpointing
                if (self.step % self.checkpoint_interval == 0):
                    self.save_checkpoint(f'checkpoint_step_{self.step}.pt')

            # Save checkpoint after completing each task
            self.save_checkpoint(f'checkpoint_after_{task_name}.pt')

            # Advance to next task
            if task_idx < len(self.tasks) - 1:
                self.current_task_idx += 1
                self.current_task_step = 0
                if self.reset_optimizer_between_tasks and self.current_task_idx < len(self.tasks):
                    print(f"Resetting optimizer for task {self.task_names[self.current_task_idx]}")
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Final checkpoint
        self.save_checkpoint('checkpoint_final.pt')
        print("\nTraining complete!")

        self.close()
