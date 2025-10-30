"""Shared utilities for all trainers."""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import csv

from tasks import BaseTask
from models import Model


class BaseTrainer:
    """Base trainer class with all common functionality."""

    def __init__(
        self,
        model: Model,
        learning_rate: float = 1e-3,
        log_dir: str | Path = 'logs',
        batch_size: int = 64,
        log_interval: int = 100,
        checkpoint_interval: int = 1000,
        optimizer_type: str = 'adam',
        weight_decay: float = 0.01,
        clip_grad_norm: Optional[float] = None
    ):
        """Initialize base trainer.

        Args:
            model: Model to train
            learning_rate: Learning rate
            log_dir: Directory for logs and checkpoints
            batch_size: Default batch size for training
            log_interval: Steps between logging
            checkpoint_interval: Steps between checkpoints
            optimizer_type: Optimizer type ('adam', 'adamw', or 'sgd')
            weight_decay: Weight decay for AdamW optimizer (default 0.01)
            clip_grad_norm: Maximum gradient norm for clipping (None to disable)
        """
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.clip_grad_norm = clip_grad_norm

        # Optimizer and loss
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Use 'adam', 'adamw', or 'sgd'.")

        self.criterion = nn.MSELoss(reduction='none')

        # Logging
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.csv_path = self.log_dir / 'metrics.csv'
        self.csv_file = None
        self.csv_writer = None

        # Training state
        self.step = 0

    def _init_csv(self, fields: list[str]) -> None:
        """Initialize CSV file with specified fields.

        Args:
            fields: List of field names for CSV columns
        """
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=['step'] + fields,
            extrasaction='ignore',
            restval=''
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def train_step(self, batch: list, task: BaseTask) -> float:
        """Unified training step for both single-trial and sequence tasks.

        Batch structure (list of N trial dicts, where N=trials_per_sequence):
            Each trial dict contains:
                inputs: [B, max_trial_len, 5]
                targets: [B, max_trial_len, 2]
                loss_mask: [B, max_trial_len, 2]
                eval_mask: [B, max_trial_len, 2]
                trial_lengths: [B] - actual length for each batch element
                metadata: dict

        For single-trial tasks: N=1, no ITI processing
        For sequence tasks: N>1, includes ITI processing between trials

        Returns:
            Loss value (float)
        """
        # Forward pass with ITI processing
        all_outputs, _, _, _ = self._forward_pass(batch, task, include_iti=True)

        # Gather targets and loss masks
        all_targets = [trial['targets'] for trial in batch]
        all_masks = [trial['loss_mask'] for trial in batch]

        # Concatenate all trials
        outputs = torch.cat(all_outputs, dim=1)  # [B, N*max_trial_len, 2]
        targets = torch.cat(all_targets, dim=1)  # [B, N*max_trial_len, 2]
        masks = torch.cat(all_masks, dim=1)  # [B, N*max_trial_len, 2]

        # Compute masked MSE loss
        mse = (outputs - targets) ** 2
        weighted_mse = mse * masks
        loss = weighted_mse.sum() / (masks.sum() + 1e-8)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Compute and log gradient norms
        total_grad_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
                self.writer.add_scalar(f'train/grad_norm/{name}', param_norm, self.step)
        total_grad_norm = total_grad_norm ** 0.5
        self.writer.add_scalar('train/grad_norm/total', total_grad_norm, self.step)

        # Gradient clipping
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.clip_grad_norm
            )

        # Log loss
        loss_val = loss.item()
        self.writer.add_scalar('loss', loss_val, self.step)

        self.optimizer.step()

        # Log weight norms
        for name, param in self.model.named_parameters():
            param_norm = param.data.norm(2).item()
            self.writer.add_scalar(f'train/weight_norm/{name}', param_norm, self.step)

        return loss_val

    def _forward_pass(self, batch, task, include_iti: bool = False, return_hidden: bool = False) -> tuple:
        """Run forward pass on batch with optional ITI processing.

        Args:
            batch: List of trial dicts
            task: Task instance
            include_iti: Whether to process inter-trial intervals
            return_hidden: Whether to return hidden states at each timestep

        Returns:
            If return_hidden=False:
                (all_outputs, all_inputs, all_targets, all_masks) - lists of tensors per trial
            If return_hidden=True:
                (all_outputs, all_inputs, all_targets, all_masks, all_hidden, iti_regions,
                 trial_boundaries, all_iti_inputs, all_iti_targets)
                where:
                - all_hidden: list of hidden states [B, H] at each timestep
                - iti_regions: list of (start, end) tuples marking ITI periods
                - trial_boundaries: list of timestep indices where trials start/end
                - all_iti_inputs: list of ITI input tensors [B, iti_len, 5]
                - all_iti_targets: list of ITI target tensors [B, iti_len, 2] (zeros)
        """
        N = len(batch)
        B = batch[0]['inputs'].shape[0]
        h = torch.zeros(B, self.model.hidden_size)

        all_outputs, all_inputs, all_targets, all_masks = [], [], [], []
        all_hidden = [] if return_hidden else None
        iti_regions = [] if return_hidden else None
        trial_boundaries = [0] if return_hidden else None
        all_iti_inputs = [] if return_hidden else None
        all_iti_targets = [] if return_hidden else None

        for trial_idx, trial in enumerate(batch):
            trial_inputs = trial['inputs']  # [B, max_trial_len, 5]
            trial_lengths = trial['trial_lengths']  # [B]
            max_trial_len = trial_inputs.shape[1]

            trial_outputs = []
            for t in range(max_trial_len):
                still_active = (t < trial_lengths).unsqueeze(-1).float()
                input_t = trial_inputs[:, t, :]
                output_t, h_new = self.model(input_t, h)
                h = still_active * h_new + (1 - still_active) * h
                trial_outputs.append(output_t)
                if return_hidden:
                    all_hidden.append(h.clone())

            trial_outputs = torch.stack(trial_outputs, dim=1)  # [B, T, 2]
            all_outputs.append(trial_outputs)
            all_inputs.append(trial_inputs)
            all_targets.append(trial['targets'])
            all_masks.append(trial['eval_mask'])

            if return_hidden:
                trial_boundaries.append(len(all_hidden))

            # ITI processing (only if requested and not last trial)
            if include_iti and trial_idx < N - 1 and hasattr(task, '_generate_iti_inputs'):
                # Compute trial correctness
                is_correct = task._evaluate_trial_correctness_batch(
                    trial_outputs,
                    trial['targets'],
                    trial['eval_mask']
                )

                # Generate and process ITI
                iti_inputs = task._generate_iti_inputs(
                    is_correct,
                    trial['metadata'],
                    task.iti_len,
                    task.reward_len
                )

                if return_hidden:
                    iti_start = len(all_hidden)

                for t in range(iti_inputs.shape[1]):
                    _, h = self.model(iti_inputs[:, t, :], h)
                    if return_hidden:
                        all_hidden.append(h.clone())

                if return_hidden:
                    all_iti_inputs.append(iti_inputs)
                    all_iti_targets.append(torch.zeros(B, iti_inputs.shape[1], 2))
                    iti_regions.append((iti_start, len(all_hidden)))
                    trial_boundaries.append(len(all_hidden))

        if return_hidden:
            return (all_outputs, all_inputs, all_targets, all_masks, all_hidden,
                    iti_regions, trial_boundaries, all_iti_inputs, all_iti_targets)
        return all_outputs, all_inputs, all_targets, all_masks

    def eval_task(self, task: BaseTask, eval_batch, log_figures: bool = False,
                  task_name: str = '', num_trials: int = 1) -> dict:
        """Evaluate model and optionally log trial figures.

        Args:
            task: Task instance
            eval_batch: Pre-generated evaluation batch
            log_figures: Whether to log trial figures
            task_name: Task name for figure logging
            num_trials: Number of trials to log

        Returns:
            Dictionary of accuracy metrics
        """
        self.model.eval()
        with torch.no_grad():
            all_outputs, all_inputs, all_targets, all_masks = self._forward_pass(eval_batch, task)

            # Concatenate for accuracy computation
            outputs = torch.cat(all_outputs, dim=1)
            targets = torch.cat(all_targets, dim=1)
            eval_mask = torch.cat(all_masks, dim=1)

            # Compute accuracies
            metrics = task.compute_accuracy(outputs, targets, eval_mask, eval_batch)

            # Log trial figures if requested
            if log_figures:
                N = min(num_trials, len(all_outputs))
                for trial_idx in range(N):
                    # Randomly select a batch element to visualize
                    batch_size = all_inputs[trial_idx].shape[0]
                    batch_idx = torch.randint(0, batch_size, (1,)).item()

                    inputs_np = all_inputs[trial_idx][batch_idx].numpy()  # [T, 5]
                    outputs_np = all_outputs[trial_idx][batch_idx].numpy()  # [T, 2]
                    targets_np = all_targets[trial_idx][batch_idx].numpy()  # [T, 2]
                    mask_np = all_masks[trial_idx][batch_idx].numpy()  # [T, 2]

                    fig = task.create_trial_figure(
                        inputs=inputs_np,
                        outputs=outputs_np,
                        targets=targets_np,
                        eval_mask=mask_np,
                        trial_idx=trial_idx,
                        batch=eval_batch,
                        batch_idx=batch_idx
                    )
                    figure_name = f'{task_name}/trial_{trial_idx+1}' if task_name else f'trial_{trial_idx+1}'
                    self.writer.add_figure(figure_name, fig, self.step)
                    plt.close(fig)

        self.model.train()
        return metrics

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics to TensorBoard and CSV.

        Args:
            metrics: Dictionary of metric name -> value
            step: Current training step
        """
        # TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

        # CSV
        if self.csv_writer is not None:
            self.csv_writer.writerow({'step': step, **metrics})
            self.csv_file.flush()

    def save_checkpoint(self, filename: str = 'checkpoint.pt', extra_state: Optional[dict] = None) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
            extra_state: Optional extra state dict to save
        """
        checkpoint_path = self.log_dir / filename
        state = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if extra_state:
            state.update(extra_state)

        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path: str | Path = None, reset_counters: bool = False) -> dict:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file (if None, loads from log_dir/checkpoint.pt)
            reset_counters: If True, reset step counters to 0 (use checkpoint as initialization only)

        Returns:
            Dictionary of extra state (anything beyond step, model, optimizer)
        """
        if path is None:
            path = self.log_dir / 'checkpoint.pt'
        checkpoint_path = Path(path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if reset_counters:
            print(f"Loaded checkpoint from {checkpoint_path} as initialization (step counters reset)")
            self.step = 0
        else:
            self.step = checkpoint['step']
            print(f"Loaded checkpoint from {checkpoint_path} (resuming from step {self.step})")

        # Extract extra state
        extra_state = {k: v for k, v in checkpoint.items()
                      if k not in ['step', 'model_state_dict', 'optimizer_state_dict']}

        return extra_state

    def close(self) -> None:
        """Close logging resources."""
        self.writer.close()
        if self.csv_file is not None:
            self.csv_file.close()
