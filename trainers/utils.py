"""Shared utilities for all trainers."""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import os
import wandb

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
        clip_grad_norm: Optional[float] = None,
        config: Optional[dict] = None,
        from_checkpoint: Optional[str | Path] = None,
        resume: bool = False,
        resume_run_id: Optional[str] = None
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
            config: Full configuration dict (will extract wandb config and log full config to wandb)
            from_checkpoint: Path to checkpoint file to load (optional)
            resume: If True, resume the wandb run from checkpoint (requires wandb_run_id in checkpoint or resume_run_id)
            resume_run_id: Manually specify wandb run ID for resuming (for old checkpoints without wandb_run_id)
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

        # If resuming, load checkpoint early to get wandb_run_id
        wandb_run_id = None
        if resume:
            if from_checkpoint is None:
                raise ValueError("Cannot resume without specifying from_checkpoint")
            checkpoint_path = Path(from_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path)

            # Try to get run ID from checkpoint first, then from resume_run_id parameter
            if 'wandb_run_id' in checkpoint:
                wandb_run_id = checkpoint['wandb_run_id']
                print(f"Resuming wandb run {wandb_run_id} from checkpoint {checkpoint_path}")
            elif resume_run_id is not None:
                wandb_run_id = resume_run_id
                print(f"Resuming wandb run {wandb_run_id} (manually specified) from checkpoint {checkpoint_path}")
            else:
                raise ValueError(
                    f"Cannot resume from checkpoint {checkpoint_path}: "
                    "wandb_run_id not found in checkpoint and resume_run_id not provided. "
                    "For old checkpoints, specify resume_run_id parameter with the wandb run ID."
                )

        # Wandb logging (always enabled)
        if config and 'wandb' in config:
            wandb_config = config['wandb']

            # Prepare wandb.init kwargs
            init_kwargs = {
                'project': wandb_config['project'],
                'dir': str(self.log_dir),
                'config': config
            }

            # Optional parameters
            if 'entity' in wandb_config and wandb_config['entity']:
                init_kwargs['entity'] = wandb_config['entity']
            if 'run_name' in wandb_config and wandb_config['run_name']:
                init_kwargs['name'] = wandb_config['run_name']
            if wandb_config.get('offline', False):
                init_kwargs['mode'] = 'offline'

            # If resuming, add run ID and resume mode
            if resume:
                init_kwargs['id'] = wandb_run_id
                init_kwargs['resume'] = 'must'

            wandb.init(**init_kwargs)

            # Define summary metrics
            wandb.define_metric("loss", summary="min")

            # Log SLURM information if running on SLURM
            if 'SLURM_JOB_ID' in os.environ:
                slurm_info = {
                    'job_id': os.environ.get('SLURM_JOB_ID'),
                    'job_name': os.environ.get('SLURM_JOB_NAME'),
                    'partition': os.environ.get('SLURM_JOB_PARTITION'),
                    'nodelist': os.environ.get('SLURM_NODELIST'),
                    'cpus': os.environ.get('SLURM_CPUS_ON_NODE'),
                    'mem_per_node': os.environ.get('SLURM_MEM_PER_NODE'),
                }
                # Remove None values
                slurm_info = {k: v for k, v in slurm_info.items() if v is not None}

                # When resuming, append to list of SLURM jobs; otherwise create new list
                if resume and 'slurm_jobs' in wandb.config:
                    # Get existing list and append new job
                    existing_jobs = list(wandb.config.slurm_jobs)
                    existing_jobs.append(slurm_info)
                    wandb.config.update({'slurm_jobs': existing_jobs}, allow_val_change=True)
                else:
                    # Fresh run or first SLURM job - create list
                    wandb.config.update({'slurm_jobs': [slurm_info]}, allow_val_change=True)

                # Add job ID as a tag for easy filtering
                if wandb.run:
                    wandb.run.tags = wandb.run.tags + (f"slurm_{slurm_info['job_id']}",)
        else:
            raise ValueError("wandb config is required in config file")

        # Print full configuration
        print("\n" + "="*60)
        print("Configuration:")
        print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
        print(f"  PyTorch threads: {torch.get_num_threads()}")
        if config:
            import json
            print("\nFull Config:")
            print(json.dumps(config, indent=2, default=str))
        print("="*60 + "\n")

        # Training state
        self.step = 0

        # Load checkpoint if provided
        if from_checkpoint is not None:
            # resume=True: continue from checkpoint step (reset_counters=False)
            # resume=False: use as initialization, reset to step 0 (reset_counters=True)
            self.load_checkpoint(from_checkpoint, reset_counters=(not resume))

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
        all_outputs, _, _, _, _ = self._forward_pass(batch, task, include_iti=True)

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
        wandb_metrics = {}
        total_grad_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
                wandb_metrics[f'train/grad_norm/{name}'] = param_norm
        total_grad_norm = total_grad_norm ** 0.5
        wandb_metrics['train/grad_norm/total'] = total_grad_norm

        # Gradient clipping
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.clip_grad_norm
            )

        # Log loss
        loss_val = loss.item()
        wandb_metrics['loss'] = loss_val

        # Optimizer step
        self.optimizer.step()

        # Log weight norms
        for name, param in self.model.named_parameters():
            param_norm = param.data.norm(2).item()
            wandb_metrics[f'train/weight_norm/{name}'] = param_norm

        # Log to wandb
        wandb.log(wandb_metrics, step=self.step)

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
                (all_outputs, all_inputs, all_targets, all_eval_masks, all_loss_masks) - lists of tensors per trial
            If return_hidden=True:
                (all_outputs, all_inputs, all_targets, all_eval_masks, all_loss_masks, all_hidden, iti_regions,
                 trial_boundaries, all_iti_inputs, all_iti_targets, all_iti_outputs)
                where:
                - all_hidden: list of hidden states [B, H] at each timestep
                - iti_regions: list of (start, end) tuples marking ITI periods
                - trial_boundaries: list of timestep indices where trials start/end
                - all_iti_inputs: list of ITI input tensors [B, iti_len, 5]
                - all_iti_targets: list of ITI target tensors [B, iti_len, 2] (zeros)
                - all_iti_outputs: list of ITI output tensors [B, iti_len, 2]
        """
        N = len(batch)
        B = batch[0]['inputs'].shape[0]
        h = torch.zeros(B, self.model.hidden_size)

        all_outputs, all_inputs, all_targets, all_eval_masks, all_loss_masks = [], [], [], [], []
        all_hidden = [] if return_hidden else None
        iti_regions = [] if return_hidden else None
        trial_boundaries = [0] if return_hidden else None
        all_iti_inputs = [] if return_hidden else None
        all_iti_targets = [] if return_hidden else None
        all_iti_outputs = [] if return_hidden else None

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
            all_eval_masks.append(trial['eval_mask'])
            all_loss_masks.append(trial['loss_mask'])

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

                # Generate ITI inputs
                iti_inputs = task._generate_iti_inputs(
                    is_correct,
                    trial['metadata'],
                    task.iti_len,
                    task.reward_len
                )

                if return_hidden:
                    iti_start = len(all_hidden)

                iti_outputs = []
                for t in range(iti_inputs.shape[1]):
                    output_t, h = self.model(iti_inputs[:, t, :], h)
                    if return_hidden:
                        all_hidden.append(h.clone())
                        iti_outputs.append(output_t)

                if return_hidden:
                    all_iti_inputs.append(iti_inputs)
                    all_iti_targets.append(torch.zeros(B, iti_inputs.shape[1], 2))
                    all_iti_outputs.append(torch.stack(iti_outputs, dim=1))  # [B, iti_len, 2]
                    iti_regions.append((iti_start, len(all_hidden)))
                    trial_boundaries.append(len(all_hidden))

        if return_hidden:
            return (all_outputs, all_inputs, all_targets, all_eval_masks, all_loss_masks, all_hidden,
                    iti_regions, trial_boundaries, all_iti_inputs, all_iti_targets, all_iti_outputs)
        return all_outputs, all_inputs, all_targets, all_eval_masks, all_loss_masks

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
            # Get ITI data if logging figures
            if log_figures and hasattr(task, '_generate_iti_inputs'):
                (all_outputs, all_inputs, all_targets, all_eval_masks, all_loss_masks, all_hidden,
                 iti_regions, trial_boundaries, all_iti_inputs, all_iti_targets, all_iti_outputs) = \
                    self._forward_pass(eval_batch, task, include_iti=True, return_hidden=True)
            else:
                all_outputs, all_inputs, all_targets, all_eval_masks, all_loss_masks = self._forward_pass(eval_batch, task)

            # Concatenate for accuracy computation
            outputs = torch.cat(all_outputs, dim=1)
            targets = torch.cat(all_targets, dim=1)
            eval_mask = torch.cat(all_eval_masks, dim=1)

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
                    eval_mask_np = all_eval_masks[trial_idx][batch_idx].numpy()  # [T, 2]
                    loss_mask_np = all_loss_masks[trial_idx][batch_idx].numpy()  # [T, 2]

                    # Get ITI data if available (only if not last trial)
                    iti_inputs_np = None
                    iti_outputs_np = None
                    if hasattr(task, '_generate_iti_inputs') and trial_idx < len(all_iti_inputs):
                        iti_inputs_np = all_iti_inputs[trial_idx][batch_idx].numpy()  # [iti_len, 5]
                        iti_outputs_np = all_iti_outputs[trial_idx][batch_idx].numpy()  # [iti_len, 2]

                    fig = task.create_trial_figure(
                        inputs=inputs_np,
                        outputs=outputs_np,
                        targets=targets_np,
                        eval_mask=eval_mask_np,
                        trial_idx=trial_idx,
                        batch=eval_batch,
                        batch_idx=batch_idx,
                        iti_inputs=iti_inputs_np,
                        iti_outputs=iti_outputs_np,
                        loss_mask=loss_mask_np
                    )
                    figure_name = f'{task_name}/trial_{trial_idx+1}' if task_name else f'trial_{trial_idx+1}'
                    wandb.log({figure_name: wandb.Image(fig)}, step=self.step)
                    plt.close(fig)

        self.model.train()
        return metrics

    def save_checkpoint(self, filename: str = 'checkpoint.pt', extra_state: Optional[dict] = None) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
            extra_state: Optional extra state dict to save
        """
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir = self.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / filename
        state = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'wandb_run_id': wandb.run.id,
        }
        if extra_state:
            state.update(extra_state)

        torch.save(state, checkpoint_path)

        # Upload to wandb
        wandb.save(str(checkpoint_path))

        # Only print for non-latest checkpoints to reduce noise
        if filename != 'latest.pt':
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

        # Extract extra state (anything beyond base trainer state)
        extra_state = {k: v for k, v in checkpoint.items()
                      if k not in ['step', 'model_state_dict', 'optimizer_state_dict', 'wandb_run_id']}

        return extra_state

    def close(self) -> None:
        """Close logging resources."""
        wandb.finish()
