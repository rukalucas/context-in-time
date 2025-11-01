"""Orthogonal Sequential Trainer - sequential training with orthogonal continual learning from Duncker et al. 2020."""

import torch
from typing import Optional

from models import Model
from tasks import BaseTask
from trainers.sequential_train import SequentialTrainer


class OrthogonalSequentialTrainer(SequentialTrainer):
    """Sequential trainer with orthogonal projection for continual learning.

    Implements the orthogonal subspaces approach from:
    Duncker et al. (2020) "Organizing recurrent network dynamics by task-computation to enable continual learning"

    Extends SequentialTrainer by adding gradient projection to protect previous task subspaces.
    """

    def __init__(
        self,
        alpha_projection: float = 0.001,
        apply_proj_to: str = 'both',
        projection_collection_batch_size: int = 64,
        **kwargs
    ):
        # Pass all parameters to parent SequentialTrainer
        super().__init__(**kwargs)

        # Orthogonal projection parameters
        self.alpha_projection = alpha_projection
        self.apply_proj_to = apply_proj_to
        self.projection_collection_batch_size = projection_collection_batch_size

        # Initialize orthogonal projection state
        self._init_projection_state()
        self._register_gradient_hooks()

    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix: X @ X.T / (n-1).

        Args:
            x: (features, samples) tensor

        Returns:
            Covariance matrix (features, features)
        """
        n = x.shape[1]
        return (x @ x.T) / (n - 1)

    def _compute_projection_matrix(self, cov: torch.Tensor, alpha: float) -> torch.Tensor:
        """Compute projection matrix from covariance.

        High-variance directions (important for previous tasks) are projected out.
        Low-variance directions remain available for new learning.

        Args:
            cov: Covariance matrix (symmetric)
            alpha: Projection strength (smaller = stronger projection)

        Returns:
            Projection matrix P
        """
        # Eigendecomposition (eigh for symmetric matrices - more stable)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Rescale eigenvalues: alpha / (alpha + lambda)
        # High eigenvalues -> near 0 (project out)
        # Low eigenvalues -> near 1 (keep)
        scaled_evals = alpha / (alpha + eigenvalues.clamp(min=1e-8))

        # Reconstruct: V @ diag(scaled) @ V.T
        P = eigenvectors @ torch.diag(scaled_evals) @ eigenvectors.T
        return P

    def _init_projection_state(self) -> None:
        """Initialize projection matrices and covariance storage.

        Following Duncker et al. 2020 notation:
        - Swh (activity_cov): W @ input_cov @ W.T
        - Suh (input_cov): joint [input, hidden] covariance
        - Syy (output_cov): output covariance
        - Shh (recurrent_cov): hidden-hidden block from input_cov
        """
        # Running average of covariance matrices (will be initialized on first task)
        self.cov_input_running = None      # Suh: joint [input, hidden] covariance
        self.cov_activity_running = None   # Swh: W @ input_cov @ W.T
        self.cov_output_running = None     # Syy: output covariance
        self.cov_recurrent_running = None  # Shh: hidden-hidden block from cov_input

        # Projection matrices (initialized to identity, updated after each task)
        # Following paper notation from Algorithm 1
        self.P_wz = None  # From Swh (activity covariance)
        self.P_z = None   # From Suh (input covariance)
        self.P_y = None   # From Syy (output covariance)
        self.P_h = None   # From Shh (recurrent covariance)

        # Gradient hook handles
        self.hook_handles = []

    def _register_gradient_hooks(self) -> None:
        """Register backward hooks to project gradients during training.

        Algorithm 1 from paper:
        - W <- W - eta P_wz^{1:k-1} grad_W L P_z^{1:k-1}
        - W^out <- W^out - eta P_y^{1:k-1} grad_{W^out} L P_h^{1:k-1}

        PyTorch (weights separated):
        - w_in: P_wz @ grad @ P_z[input_block]
        - w_rec: P_wz @ grad @ P_h
        - w_out: P_y @ grad @ P_h
        """
        def create_projection_hook(param_name: str):
            """Create a hook function for a specific parameter."""
            def hook(grad):
                # Only project if we have computed projections (after first task)
                if self.current_task_idx == 0 or self.P_wz is None:
                    return grad

                # Project gradient based on parameter name
                if 'w_in.weight' in param_name:
                    if self.apply_proj_to in ['both', 'recurrent']:
                        return self._project_gradient_w_in(grad)

                elif 'w_rec.weight' in param_name:
                    if self.apply_proj_to in ['both', 'recurrent']:
                        return self._project_gradient_w_rec(grad)

                elif 'w_out.weight' in param_name:
                    if self.apply_proj_to in ['both', 'readout']:
                        return self._project_gradient_w_out(grad)

                # Bias terms are not projected
                return grad
            return hook

        # Register hooks on weight parameters (including w_in now!)
        for name, param in self.model.named_parameters():
            if 'weight' in name and ('w_in' in name or 'w_rec' in name or 'w_out' in name):
                handle = param.register_hook(create_projection_hook(name))
                self.hook_handles.append(handle)

    def _project_gradient_w_in(self, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient for w_in: P_wz @ grad @ P_z[input_block].

        Args:
            grad: (n_rnn, n_input) gradient tensor

        Returns:
            Projected gradient
        """
        # Extract input block from P_z
        n_input = grad.shape[1]
        P_z_input_block = self.P_z[:n_input, :n_input]
        grad_proj = self.P_wz @ grad @ P_z_input_block
        return grad_proj

    def _project_gradient_w_rec(self, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient for w_rec: P_wz @ grad @ P_h.

        Args:
            grad: (n_rnn, n_rnn) gradient tensor

        Returns:
            Projected gradient
        """
        # P_h is bottom-right block of P_z (recurrent-recurrent)
        grad_proj = self.P_wz @ grad @ self.P_h
        return grad_proj

    def _project_gradient_w_out(self, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient for w_out: P_y @ grad @ P_h.

        Args:
            grad: (n_output, n_rnn) gradient tensor

        Returns:
            Projected gradient
        """
        grad_proj = self.P_y @ grad @ self.P_h
        return grad_proj

    def _collect_task_activity_and_update_projections(self) -> None:
        """Collect activity from completed task and update projection matrices."""
        task = self.tasks[self.current_task_idx]
        task_name = self.task_names[self.current_task_idx]

        print(f"Collecting activity for task '{task_name}' to update projections...")

        # Generate batch for activity collection
        batch = task.generate_batch(self.projection_collection_batch_size)

        # Run forward pass and collect activities
        self.model.eval()
        with torch.no_grad():
            if self._is_sequence_task(batch):
                # Sequence task: process trials with timestep loop
                N = len(batch)
                B = batch[0]['inputs'].shape[0]
                h = torch.zeros(B, self.model.hidden_size)

                all_inputs = []
                all_hidden = []
                all_outputs = []

                for trial in batch:
                    trial_inputs = trial['inputs']  # [B, max_trial_len, 5]
                    trial_lengths = trial['trial_lengths']  # [B]
                    max_trial_len = trial_inputs.shape[1]

                    # Process trial timestep by timestep
                    for t in range(max_trial_len):
                        still_active = (t < trial_lengths).unsqueeze(-1).float()
                        input_t = trial_inputs[:, t, :]  # [B, 5]
                        output_t, h_new = self.model(input_t, h)
                        h = still_active * h_new + (1 - still_active) * h

                        # Collect activities for active timesteps
                        all_inputs.append(input_t)  # [B, 5]
                        all_hidden.append(h)  # [B, hidden_size]
                        all_outputs.append(output_t)  # [B, output_size]

                # Stack and flatten: List of [B, C] -> [B*N*T, C]
                inputs_flat = torch.cat(all_inputs, dim=0)  # [B*N*T, 5]
                hidden_flat = torch.cat(all_hidden, dim=0)  # [B*N*T, hidden_size]
                outputs_flat = torch.cat(all_outputs, dim=0)  # [B*N*T, output_size]
            else:
                # Single trial task: process with timestep loop
                inputs = batch['inputs']  # [B, T, 5]
                B, T, _ = inputs.shape
                h = torch.zeros(B, self.model.hidden_size)

                all_inputs = []
                all_hidden = []
                all_outputs = []

                for t in range(T):
                    input_t = inputs[:, t, :]  # [B, 5]
                    output_t, h = self.model(input_t, h)

                    all_inputs.append(input_t)  # [B, 5]
                    all_hidden.append(h)  # [B, hidden_size]
                    all_outputs.append(output_t)  # [B, output_size]

                # Stack and flatten: List of [B, C] -> [B*T, C]
                inputs_flat = torch.cat(all_inputs, dim=0)  # [B*T, 5]
                hidden_flat = torch.cat(all_hidden, dim=0)  # [B*T, hidden_size]
                outputs_flat = torch.cat(all_outputs, dim=0)  # [B*T, output_size]

        # Compute covariances for this task (transpose for features×samples format)
        joint_state = torch.cat([inputs_flat, hidden_flat], dim=1).T  # (n_input+n_hidden, samples)
        cov_input_task = self._compute_covariance(joint_state)
        cov_output_task = self._compute_covariance(outputs_flat.T)

        # Get weight matrices for computing activity covariance
        W_in = self.model.w_in.weight.data  # (hidden_size, input_size)
        W_rec = self.model.w_rec.weight.data  # (hidden_size, hidden_size)
        W_full = torch.cat([W_in, W_rec], dim=1)  # (hidden_size, input_size + hidden_size)

        # Compute activity covariance: W @ cov_input @ W.T
        cov_activity_task = W_full @ cov_input_task @ W_full.T

        # Extract recurrent block (bottom-right of cov_input)
        n_hidden = self.model.hidden_size
        cov_recurrent_task = cov_input_task[-n_hidden:, -n_hidden:]

        # Delete large tensors immediately to free memory
        del inputs_flat, hidden_flat, outputs_flat, joint_state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Update running averages
        k = self.current_task_idx
        if k == 0:
            # First task - initialize
            self.cov_input_running = cov_input_task
            self.cov_activity_running = cov_activity_task
            self.cov_output_running = cov_output_task
            self.cov_recurrent_running = cov_recurrent_task
        else:
            # Subsequent tasks - weighted running average
            weight_old = k / (k + 1)
            weight_new = 1 / (k + 1)

            self.cov_input_running = weight_old * self.cov_input_running + weight_new * cov_input_task
            self.cov_activity_running = weight_old * self.cov_activity_running + weight_new * cov_activity_task
            self.cov_output_running = weight_old * self.cov_output_running + weight_new * cov_output_task
            self.cov_recurrent_running = weight_old * self.cov_recurrent_running + weight_new * cov_recurrent_task

        # Recompute projection matrices (following Algorithm 1 notation)
        # P_wz from Swh, P_z from Suh, P_y from Syy, P_h from Shh
        self.P_wz = self._compute_projection_matrix(self.cov_activity_running, self.alpha_projection)
        self.P_z = self._compute_projection_matrix(self.cov_input_running, self.alpha_projection)
        self.P_y = self._compute_projection_matrix(self.cov_output_running, self.alpha_projection)
        self.P_h = self._compute_projection_matrix(self.cov_recurrent_running, self.alpha_projection)

        # Log projection statistics
        eigenvalues_input, _ = torch.linalg.eigh(self.cov_input_running)
        eigenvalues_output, _ = torch.linalg.eigh(self.cov_output_running)

        print(f"  Input covariance eigenvalues (max/min): {eigenvalues_input.max():.4f} / {eigenvalues_input.min():.4f}")
        print(f"  Output covariance eigenvalues (max/min): {eigenvalues_output.max():.4f} / {eigenvalues_output.min():.4f}")
        print(f"  Projection matrices updated for {k+1} task(s)")

        self.model.train()

    def _recalculate_all_task_covariances(self) -> None:
        """Recalculate covariances for all tasks when starting from non-orthogonal checkpoint."""
        for task_idx, (task, task_name) in enumerate(zip(self.tasks, self.task_names)):
            print(f"  Computing covariances for task {task_idx+1}/{len(self.tasks)}: {task_name}")

            # Generate batch for activity collection
            batch = task.generate_batch(self.projection_collection_batch_size)

            # Run forward pass and collect activities
            self.model.eval()
            with torch.no_grad():
                if self._is_sequence_task(batch):
                    # Sequence task: process trials with timestep loop
                    N = len(batch)
                    B = batch[0]['inputs'].shape[0]
                    h = torch.zeros(B, self.model.hidden_size)

                    all_inputs = []
                    all_hidden = []
                    all_outputs = []

                    for trial in batch:
                        trial_inputs = trial['inputs']  # [B, max_trial_len, 5]
                        trial_lengths = trial['trial_lengths']  # [B]
                        max_trial_len = trial_inputs.shape[1]

                        # Process trial timestep by timestep
                        for t in range(max_trial_len):
                            still_active = (t < trial_lengths).unsqueeze(-1).float()
                            input_t = trial_inputs[:, t, :]  # [B, 5]
                            output_t, h_new = self.model(input_t, h)
                            h = still_active * h_new + (1 - still_active) * h

                            # Collect activities for active timesteps
                            all_inputs.append(input_t)  # [B, 5]
                            all_hidden.append(h)  # [B, hidden_size]
                            all_outputs.append(output_t)  # [B, output_size]

                    # Stack and flatten: List of [B, C] -> [B*N*T, C]
                    inputs_flat = torch.cat(all_inputs, dim=0)  # [B*N*T, 5]
                    hidden_flat = torch.cat(all_hidden, dim=0)  # [B*N*T, hidden_size]
                    outputs_flat = torch.cat(all_outputs, dim=0)  # [B*N*T, output_size]
                else:
                    # Single trial task: process with timestep loop
                    inputs = batch['inputs']  # [B, T, 5]
                    B, T, _ = inputs.shape
                    h = torch.zeros(B, self.model.hidden_size)

                    all_inputs = []
                    all_hidden = []
                    all_outputs = []

                    for t in range(T):
                        input_t = inputs[:, t, :]  # [B, 5]
                        output_t, h = self.model(input_t, h)

                        all_inputs.append(input_t)  # [B, 5]
                        all_hidden.append(h)  # [B, hidden_size]
                        all_outputs.append(output_t)  # [B, output_size]

                    # Stack and flatten: List of [B, C] -> [B*T, C]
                    inputs_flat = torch.cat(all_inputs, dim=0)  # [B*T, 5]
                    hidden_flat = torch.cat(all_hidden, dim=0)  # [B*T, hidden_size]
                    outputs_flat = torch.cat(all_outputs, dim=0)  # [B*T, output_size]

            # Compute covariances for this task
            joint_state = torch.cat([inputs_flat, hidden_flat], dim=1).T
            cov_input_task = self._compute_covariance(joint_state)
            cov_output_task = self._compute_covariance(outputs_flat.T)

            # Get weight matrices for computing activity covariance
            W_in = self.model.w_in.weight.data
            W_rec = self.model.w_rec.weight.data
            W_full = torch.cat([W_in, W_rec], dim=1)

            # Compute activity covariance
            cov_activity_task = W_full @ cov_input_task @ W_full.T

            # Extract recurrent block
            n_hidden = self.model.hidden_size
            cov_recurrent_task = cov_input_task[-n_hidden:, -n_hidden:]

            # Delete large tensors
            del inputs_flat, hidden_flat, outputs_flat, joint_state
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Update running averages
            if task_idx == 0:
                # First task - initialize
                self.cov_input_running = cov_input_task
                self.cov_activity_running = cov_activity_task
                self.cov_output_running = cov_output_task
                self.cov_recurrent_running = cov_recurrent_task
            else:
                # Subsequent tasks - weighted running average
                weight_old = task_idx / (task_idx + 1)
                weight_new = 1 / (task_idx + 1)

                self.cov_input_running = weight_old * self.cov_input_running + weight_new * cov_input_task
                self.cov_activity_running = weight_old * self.cov_activity_running + weight_new * cov_activity_task
                self.cov_output_running = weight_old * self.cov_output_running + weight_new * cov_output_task
                self.cov_recurrent_running = weight_old * self.cov_recurrent_running + weight_new * cov_recurrent_task

        # Recompute projection matrices
        self.P_wz = self._compute_projection_matrix(self.cov_activity_running, self.alpha_projection)
        self.P_z = self._compute_projection_matrix(self.cov_input_running, self.alpha_projection)
        self.P_y = self._compute_projection_matrix(self.cov_output_running, self.alpha_projection)
        self.P_h = self._compute_projection_matrix(self.cov_recurrent_running, self.alpha_projection)

        print(f"  Covariances computed for all {len(self.tasks)} tasks")
        self.model.train()

    def _advance_to_next_task(self) -> None:
        """Override to collect activity and update projections before advancing."""
        self._collect_task_activity_and_update_projections()
        super()._advance_to_next_task()

    def save_checkpoint(self, filename: str = 'checkpoint.pt') -> None:
        """Override to save projection matrices in addition to base state."""
        extra_state = {
            'use_orthogonal_projection': True,
            'alpha_projection': self.alpha_projection,
            'apply_proj_to': self.apply_proj_to
        }

        # Save covariance matrices if they exist
        if self.cov_input_running is not None:
            extra_state['cov_input_running'] = self.cov_input_running
            extra_state['cov_activity_running'] = self.cov_activity_running
            extra_state['cov_output_running'] = self.cov_output_running
            extra_state['cov_recurrent_running'] = self.cov_recurrent_running

        # Save projection matrices if they exist
        if self.P_wz is not None:
            extra_state['P_wz'] = self.P_wz
            extra_state['P_z'] = self.P_z
            extra_state['P_y'] = self.P_y
            extra_state['P_h'] = self.P_h

        super().save_checkpoint(filename=filename, extra_state=extra_state)

    def _restore_extra_state(self, extra_state: dict) -> None:
        """Override to restore projection state in addition to base state."""
        # Restore base sequential trainer state
        super()._restore_extra_state(extra_state)

        # Restore projection state if it was saved
        if extra_state.get('use_orthogonal_projection', False):
            # Restore covariance matrices
            if 'cov_input_running' in extra_state:
                self.cov_input_running = extra_state['cov_input_running']
                self.cov_activity_running = extra_state['cov_activity_running']
                self.cov_output_running = extra_state['cov_output_running']
                self.cov_recurrent_running = extra_state['cov_recurrent_running']

            # Restore projection matrices
            if 'P_wz' in extra_state:
                self.P_wz = extra_state['P_wz']
                self.P_z = extra_state['P_z']
                self.P_y = extra_state['P_y']
                self.P_h = extra_state['P_h']

                print("Restored projection state from checkpoint")
            else:
                # No projection state in checkpoint - recalculate from current model state
                print("No projection state in checkpoint - recalculating covariances for all tasks...")
                self._recalculate_all_task_covariances()

    def train(
        self,
        batch_size: int = None,
        log_interval: int = None,
        checkpoint_interval: int = None
    ) -> None:
        """Override to add orthogonal projection info to startup message."""
        print(f"Orthogonal projection: ENABLED (alpha={self.alpha_projection}, apply_to={self.apply_proj_to})")
        super().train(batch_size=batch_size, log_interval=log_interval, checkpoint_interval=checkpoint_interval)
