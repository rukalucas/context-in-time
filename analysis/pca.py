"""PCA analysis tools for neural trajectory visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from typing import NamedTuple
from pathlib import Path

# Import dependencies from utils
from analysis.utils import compute_trial_events, get_metadata, _extract_period_boundaries


class ProjectionVector(NamedTuple):
    """Represents a projection direction with human-readable label."""
    vector: np.ndarray  # [H] normalized direction vector
    name: str  # Human-readable label (e.g., "Output dim 0")


def parse_projection(projection_spec, model):
    """
    Parse projection specification into ProjectionVector.

    Args:
        projection_spec: str like 'output_dim_0', 'input_dim_2', or np.ndarray
        model: RNN model with w_out and w_in attributes

    Returns:
        ProjectionVector with normalized vector and label
    """
    if isinstance(projection_spec, np.ndarray):
        # Custom vector
        vector = projection_spec / np.linalg.norm(projection_spec)
        return ProjectionVector(vector=vector, name='Custom')

    if isinstance(projection_spec, str):
        parts = projection_spec.split('_')
        if len(parts) != 3:
            raise ValueError(f"Invalid projection spec: {projection_spec}. Expected format: 'output_dim_0' or 'input_dim_2'")

        direction_type = parts[0]  # 'output' or 'input'
        dim_idx = int(parts[2])  # dimension index

        if direction_type == 'output':
            # w_out shape: [output_size, hidden_size]
            vector = model.w_out.weight.detach().cpu().numpy()[dim_idx, :]
            name = f"Output dim {dim_idx}"
        elif direction_type == 'input':
            # w_in shape: [hidden_size, input_size]
            vector = model.w_in.weight.detach().cpu().numpy()[:, dim_idx]
            name = f"Input dim {dim_idx}"
        else:
            raise ValueError(f"Unknown direction type: {direction_type}. Must be 'output' or 'input'")

        # Normalize
        vector = vector / np.linalg.norm(vector)
        return ProjectionVector(vector=vector, name=name)

    raise TypeError(f"projection_spec must be str or np.ndarray, got {type(projection_spec)}")


def _get_trial_lengths(batch):
    """Extract trial lengths from batch (handles both sequence and non-sequence formats)."""
    if len(batch) > 1:
        # Sequence: N trial dicts with B=1
        return [batch[i]['trial_lengths'][0].item() for i in range(len(batch))]
    else:
        # Non-sequence: 1 trial dict with B=N
        return batch[0]['trial_lengths'].cpu().numpy().tolist()


def _flatten_trials(data, lengths):
    """Flatten [N, T, H] data using only non-padded timesteps."""
    return np.vstack([data[i, :lengths[i], :] for i in range(len(lengths))])


def _compute_pca(data_flat, n_components):
    """
    Compute PCA on flattened data.

    Returns:
        pcs: [H, n_components] principal components
        eigenvalues: [H] all eigenvalues (sorted descending)
        mean: [H] data mean
    """
    mean = data_flat.mean(axis=0)
    centered = data_flat - mean # [N_total_steps, H]
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pcs = eigenvectors[:, :n_components]
    return pcs, eigenvalues, mean


def _extract_periods(hidden_np, period_boundaries_list, period_list):
    """
    Extract and concatenate multiple periods from hidden states.

    Args:
        hidden_np: [N, H, T] hidden states
        period_boundaries_list: Dict mapping period names to lists of boundary dicts
        period_list: List of period names to extract

    Returns:
        extracted_data: [N, T_extracted, H]
        time_indices: [N, T_extracted] mapping to original timesteps
        lengths: [N] actual (non-padded) lengths for each trial
    """
    num_trials = hidden_np.shape[0]

    # Compute lengths for each trial
    lengths = []
    for i in range(num_trials):
        total_len = 0
        for pname in period_list:
            info = period_boundaries_list[pname][i]
            period_len = info['end'] - info['start']
            total_len += period_len
        lengths.append(total_len)

    max_len = max(lengths) if lengths else 0
    extracted_data = np.zeros((num_trials, max_len, hidden_np.shape[1]))
    time_indices = np.zeros((num_trials, max_len), dtype=int)

    # Extract data for each trial
    for i in range(num_trials):
        idx = 0
        for period_name in period_list:
            info = period_boundaries_list[period_name][i]
            start, end = info['start'], info['end']
            period_len = end - start

            # Skip empty periods
            if period_len == 0:
                continue

            # Check if we need to extract from a different trial (e.g., pre_iti)
            trial_idx = info.get('trial_idx', i)

            # Extract [H, T_period] and transpose to [T_period, H]
            extracted_data[i, idx:idx+period_len, :] = hidden_np[trial_idx, :, start:end].T
            time_indices[i, idx:idx+period_len] = np.arange(start, end)
            idx += period_len

    return extracted_data, time_indices, lengths


def do_pca(
    data_dict,
    task,
    model=None,
    periods='all',
    projection=None,
    n_components=3,
):
    """
    Unified PCA function for neural trajectory analysis.

    Args:
        data_dict: dict from generate_data() containing 'hidden_states' and 'batch'
        task: Task instance
        model: RNN model (required if projection specified)
        periods: 'all', single period name, or list of period names to concatenate
                Available periods: 'rule_report', 'timing', 'decision', 'pre_iti', 'post_iti'
                (pre_iti and post_iti only available for sequence tasks)
        projection: None, projection spec string, np.ndarray, or list of specs
                   Examples: 'output_dim_0', 'input_dim_2', ['output_dim_0', 'output_dim_1']
        n_components: Number of orthogonal PCs (if projection) or total PCs (if no projection)

    Returns:
        dict containing:
            - 'pca_data': [N, T, D] where D = n_projections (0 if not projection) + n_components
            - 'explained_variance': [n_components] variance explained by orthogonal PCs
            - 'projection_variance': [n_projections] variance explained by projection dims (if projection)
            - 'projection_vectors': [ProjectionVector(...)] or None
            - 'pcs': [H, n_components], PC vectors (orthogonal PCs if projection)
            - 'mean': [H]
            - 'axis_labels': ['Output dim 0 (X% var)', 'PC1 (orth, Y% var)', ...]
            - 'lengths': [N] trial lengths
            - 'events': dict of event timestamps
            - 'metadata': trial metadata
            - 'period_info': period boundaries
    """
    # Extract from data_dict (generate_data() guarantees [N, H, T] format)
    hidden_states = data_dict['hidden_states']  # [N, H, T]
    batch = data_dict['batch']

    hidden_np = hidden_states.cpu().numpy()  # [N, H, T]
    num_trials = hidden_np.shape[0]

    # Extract periods
    if periods == 'all':
        # Use all hidden states: [N, H, T] -> [N, T, H]
        extracted_data = np.transpose(hidden_np, (0, 2, 1))
        lengths = _get_trial_lengths(batch)

        # Map each trial to its original timesteps [0, 1, 2, ..., T-1]
        max_len = extracted_data.shape[1]
        time_indices = np.zeros((num_trials, max_len), dtype=int)
        for i in range(num_trials):
            time_indices[i, :lengths[i]] = np.arange(lengths[i])

        period_info = {'period_names': ['all'], 'boundaries': None}
    else:
        # Extract specific period(s)
        period_list = periods if isinstance(periods, list) else [periods]
        period_boundaries = _extract_period_boundaries(batch, task, num_trials)

        extracted_data, time_indices, lengths = _extract_periods(
            hidden_np, period_boundaries, period_list
        )

        period_info = {'period_names': period_list, 'boundaries': period_boundaries}

    # Flatten data and compute total variance
    flat_data = _flatten_trials(extracted_data, lengths) # [N_total, H]
    mean = flat_data.mean(axis=0)
    total_variance = np.sum(np.var(flat_data - mean, axis=0))

    # Handle projection
    if projection is not None:
        if model is None:
            raise ValueError("model must be provided when using projection")

        # Parse projection spec(s)
        projection_list = projection if isinstance(projection, list) else [projection]
        projection_vectors = [parse_projection(spec, model) for spec in projection_list]

        # Extract outputs and inputs for actual value projections (includes bias terms)
        outputs_np = data_dict['outputs'].cpu().numpy()  # [N, T_full, output_dim]
        inputs_np = data_dict['inputs'].cpu().numpy()  # [N, T_full, input_dim]

        # Project data ONTO projection vectors and compute variance
        projection_data = []
        projection_variance = []
        for proj_vec in projection_vectors:
            # Check if this is an output or input dimension projection
            if proj_vec.name.startswith('Output dim'):
                # Extract output dimension index from name like "Output dim 0"
                dim_idx = int(proj_vec.name.split()[-1])
                # Use actual output values (includes bias)
                proj_full = outputs_np[:, :, dim_idx]  # [N, T_full]

                # Extract same periods/timesteps as extracted_data
                proj_onto = np.zeros((num_trials, extracted_data.shape[1]))
                for i in range(num_trials):
                    for t in range(lengths[i]):
                        original_t = time_indices[i, t]
                        proj_onto[i, t] = proj_full[i, original_t]

            elif proj_vec.name.startswith('Input dim'):
                # Extract input dimension index
                dim_idx = int(proj_vec.name.split()[-1])
                # Use actual input values
                proj_full = inputs_np[:, :, dim_idx]  # [N, T_full]

                # Extract same periods/timesteps as extracted_data
                proj_onto = np.zeros((num_trials, extracted_data.shape[1]))
                for i in range(num_trials):
                    for t in range(lengths[i]):
                        original_t = time_indices[i, t]
                        proj_onto[i, t] = proj_full[i, original_t]
            else:
                # Custom projection: project hidden states
                proj_onto = np.einsum('nth,h->nt', extracted_data, proj_vec.vector)

            projection_data.append(proj_onto[:, :, np.newaxis])  # [N, T, 1]

            # Compute variance explained
            flat_proj = np.concatenate([proj_onto[i, :lengths[i]] for i in range(num_trials)])
            projection_variance.append(np.var(flat_proj) / total_variance)

        projection_data = np.concatenate(projection_data, axis=2)  # [N, T, n_proj]
        projection_variance = np.array(projection_variance)

        # Project OUT projection vectors from data (orthogonalize)
        orthogonal_data = extracted_data.copy()
        for proj_vec in projection_vectors:
            proj_coef = np.einsum('nth,h->nt', orthogonal_data, proj_vec.vector)
            orthogonal_data -= np.einsum('nt,h->nth', proj_coef, proj_vec.vector)

        # PCA on orthogonal data
        flat_orthogonal = _flatten_trials(orthogonal_data, lengths) # [N_total, H]
        pcs, eigenvalues, _ = _compute_pca(flat_orthogonal, n_components)

        # Variance explained (relative to total variance)
        explained_variance = eigenvalues[:n_components] / total_variance

        # Project orthogonal data onto PCs
        orthogonal_pca = np.einsum('nth,hc->ntc', orthogonal_data - mean, pcs)
        pca_data = np.concatenate([projection_data, orthogonal_pca], axis=2)

        # Generate axis labels
        axis_labels = [
            f"{pv.name} ({projection_variance[i]:.1%} var)"
            for i, pv in enumerate(projection_vectors)
        ] + [
            f"PC{i+1} (orth, {explained_variance[i]:.1%} var)"
            for i in range(n_components)
        ]

    else:
        # Standard PCA (no projection)
        pcs, eigenvalues, mean = _compute_pca(flat_data, n_components)
        explained_variance = eigenvalues[:n_components] / eigenvalues.sum()
        projection_variance = None
        projection_vectors = None

        # Project all data onto PCs
        pca_data = np.einsum('nth,hc->ntc', extracted_data - mean, pcs)

        axis_labels = [f"PC{i+1} ({explained_variance[i]:.1%} var)" for i in range(n_components)]

    # Compute events (in original full-trial coordinates)
    events = compute_trial_events(batch, task)

    # Extract metadata
    metadata = get_metadata(batch)

    # Return result dict
    return {
        'pca_data': pca_data,
        'explained_variance': explained_variance,
        'projection_variance': projection_variance,
        'projection_vectors': projection_vectors,
        'pcs': pcs if projection is None else None,
        'mean': mean,
        'axis_labels': axis_labels,
        'lengths': lengths,
        'time_indices': time_indices,  # Maps pca_data indices to original trial timesteps
        'events': events,  # Event timesteps in original trial coordinates
        'metadata': metadata,
        'period_info': period_info,
    }


def plot_cross_period_variance(data_dict, task, period_names=None, n_components=3,
                               figsize=(8, 7), save_path=None):
    """
    Plot heatmap showing how much variance in one period is explained by another period's PCs.

    This performs cross-validation style analysis: fits PCs on one period and tests
    how much variance those PCs explain in other periods.

    Args:
        data_dict: dict from generate_data() containing 'hidden_states' and 'batch'
        task: Task instance
        period_names: list of period names to compare. Use 'iti' for ITI period (internally uses post_iti).
                     Default None returns all available periods: ['rule_report', 'timing', 'decision', 'iti']
                     Available periods: 'rule_report', 'timing', 'decision', 'iti'
        n_components: Number of PCs to use for each period
        figsize: tuple, figure size
        save_path: str or Path, optional path to save figure

    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    hidden_states = data_dict['hidden_states']
    batch = data_dict['batch']

    # Set default period names
    if period_names is None:
        period_names = ['rule_report', 'timing', 'decision']
        # Check if this is a sequence task with ITI
        if len(batch) > 1 or (len(batch) == 1 and 'iti_start' in batch[0]['metadata']):
            period_names.append('iti')

    # Map 'iti' to 'post_iti' for internal use
    internal_period_names = [p if p != 'iti' else 'post_iti' for p in period_names]

    # Create display labels (capitalize, convert 'iti' to 'ITI')
    display_labels = []
    for p in period_names:
        if p == 'iti':
            display_labels.append('ITI')
        else:
            display_labels.append(p.replace('_', ' ').title())

    n_periods = len(period_names)

    # Run do_pca for each period to get PCs and flattened data
    period_results = {}
    for internal_name in internal_period_names:
        result = do_pca(data_dict, task, periods=internal_name, n_components=n_components)

        pcs = result['pcs']  # [H, n_comp]
        mean = result['mean']  # [H]

        # We need the raw hidden data for this period, not the PCA'd data
        # Extract it again using _extract_period_boundaries
        num_trials = hidden_states.shape[0]
        period_info_dict = _extract_period_boundaries(batch, task, num_trials)
        period_boundaries = period_info_dict[internal_name]

        hidden_np = hidden_states.cpu().numpy()  # [N, H, T]

        # Flatten period data
        flat_data_list = []
        for i, info in enumerate(period_boundaries):
            start, end = info['start'], info['end']
            period_len = end - start

            # Skip empty periods
            if period_len == 0:
                continue

            # Check if we need to extract from a different trial (e.g., pre_iti)
            trial_idx = info.get('trial_idx', i)

            # Extract and flatten: [H, T_period] -> [T_period, H]
            flat_data_list.append(hidden_np[trial_idx, :, start:end].T)

        hidden_flat = np.vstack(flat_data_list)  # [sum(lengths), H]

        period_results[internal_name] = {
            'pcs': pcs,
            'hidden_flat': hidden_flat,
            'mean': mean
        }

    # Create variance explained matrix
    variance_matrix = np.zeros((n_periods, n_periods))

    for i, fit_period in enumerate(internal_period_names):
        fit_pcs = period_results[fit_period]['pcs']  # (H, n_components)

        for j, test_period in enumerate(internal_period_names):
            test_data = period_results[test_period]['hidden_flat']  # (N, H)

            # Center test data
            test_centered = test_data - test_data.mean(axis=0)

            # Project onto fit period's PCs
            test_projected = test_centered @ fit_pcs  # (N, n_components)

            # Compute variance explained
            projected_variance = np.sum(np.var(test_projected, axis=0))
            total_variance = np.sum(np.var(test_centered, axis=0))
            variance_explained = projected_variance / total_variance

            variance_matrix[i, j] = variance_explained * 100  # Convert to percentage

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(variance_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=100)

    # Set ticks and labels (use display labels)
    ax.set_xticks(np.arange(n_periods))
    ax.set_yticks(np.arange(n_periods))
    ax.set_xticklabels(display_labels)
    ax.set_yticklabels(display_labels)

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add value annotations
    for i in range(n_periods):
        for j in range(n_periods):
            ax.text(j, i, f'{variance_matrix[i, j]:.1f}',
                    ha='center', va='center', color='white' if variance_matrix[i, j] < 50 else 'black',
                    fontsize=11)

    # Labels and title
    ax.set_xlabel('Test', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fit', fontsize=12, fontweight='bold')
    ax.set_title(f'Cross-Period Variance Explained by Top {n_components} PCs (%)', fontsize=13, pad=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Variance Explained (%)', fontsize=11)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cross-period variance plot to {save_path}")

    return fig, ax


def visualize_pca(result, segments=None, plot_3d=False, num_trials=None, color_by='rule',
                  figsize=(10, 8), save_path=None):
    """
    Create static PCA visualization (2D or 3D) with optional segmentation.

    Args:
        result: dict from do_pca() containing pca_data, axis_labels, metadata, events, lengths
        segments: Optional list of segment specs:
                 [{'start': event_name, 'end': event_name, 'alpha': float, 'label': str}, ...]
                 Events can be: 'trial_start', 'cue_onset', 'first_pulse', 'second_pulse', etc.
        plot_3d: bool, whether to plot in 3D
        num_trials: Number of trials to plot (default None = plot all trials)
        color_by: str - what to color by
                       Discrete: 'rule', 'decision', 'stim_direction', 'instructed', 'switch', 'reward'
                       Continuous (with colorbar): 't_m', 't_s'
        figsize: tuple of figure size
        save_path: Optional path to save figure

    Returns:
        fig, ax
    """
    # Extract from result dict
    pca_data = result['pca_data']
    axis_labels = result['axis_labels']
    trial_lengths = result['lengths']
    metadata = result['metadata']
    events = result.get('events', None)
    time_indices = result.get('time_indices', None)
    if plot_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Set up coloring
    color_map, labels = _get_color_scheme(color_by, metadata)

    # Limit number of trials (None = all trials)
    if num_trials is None:
        num_trials = len(trial_lengths)
    else:
        num_trials = min(num_trials, len(trial_lengths))

    # Check if using continuous colormap
    use_continuous = color_map == 'continuous'

    if use_continuous and metadata is not None and color_by in metadata:
        # Set up continuous colormap
        import matplotlib.cm as cm
        cmap = cm.viridis
        color_values = metadata[color_by][:num_trials]
        vmin, vmax = color_values.min(), color_values.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Helper function to get event index for a trial
    def get_event_idx(event_name, trial_idx):
        """
        Get timestep index for an event in the extracted PCA data.

        Uses time_indices to map from original trial coordinates to extracted data indices.
        Returns None if event is not found within the extracted period.
        """
        # Get event timestep in original trial coordinates
        if event_name == 'trial_end' or event_name == 'period_end':
            # Special case: use the actual length of extracted data
            return trial_lengths[trial_idx]
        elif events and event_name in events:
            event_timestep = events[event_name][trial_idx]
        else:
            # Try parsing as integer
            try:
                event_timestep = int(event_name)
            except ValueError:
                raise ValueError(f"Unknown event: {event_name}")

        # Find where this timestep appears in the extracted data
        if time_indices is not None:
            # Search for the event timestep in this trial's time_indices
            trial_time_indices = time_indices[trial_idx, :trial_lengths[trial_idx]]
            matches = np.where(trial_time_indices == event_timestep)[0]
            if len(matches) > 0:
                return matches[0]
            else:
                # Event not found in extracted period
                return None
        else:
            # No time_indices (shouldn't happen, but fallback to direct index)
            return event_timestep

    # Collect line segments and colors for batch rendering
    if plot_3d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        segments_3d = []
        colors_3d = []
        alphas_3d = []
    else:
        from matplotlib.collections import LineCollection
        segments_2d = []
        colors_2d = []
        alphas_2d = []

    # Collect start/end markers
    start_points = []
    end_points = []
    marker_colors = []
    legend_handles = {}  # Track unique labels for legend

    # Collect all line segments and markers
    for i in range(num_trials):
        T_trial = min(trial_lengths[i], pca_data.shape[1])

        # Get color for this trial
        if use_continuous and metadata is not None and color_by in metadata:
            color_val = metadata[color_by][i]
            color = scalar_map.to_rgba(color_val)
            label = None
        elif metadata is not None and color_by in metadata:
            color_val = metadata[color_by][i]
            color = color_map.get(color_val, 'gray')
            label = labels.get(color_val, None)
        else:
            color = f'C{i % 10}'
            label = f'Trial {i+1}' if i == 0 else None

        # Handle segmentation
        if segments is not None and events is not None:
            # Render trajectory with variable transparency per segment
            for seg_spec in segments:
                start_idx = get_event_idx(seg_spec['start'], i)
                end_idx = get_event_idx(seg_spec['end'], i)

                # Check if events are within extracted period
                if start_idx is None or end_idx is None:
                    # Warn once per segment type (not per trial)
                    seg_label = seg_spec.get('label', f"{seg_spec['start']}→{seg_spec['end']}")
                    if i == 0:  # Only warn on first trial
                        print(f"Warning: Segment '{seg_label}' (events {seg_spec['start']}→{seg_spec['end']}) "
                              f"is outside the extracted period and will be skipped.")
                    continue

                start_idx = max(0, min(start_idx, T_trial))
                end_idx = max(0, min(end_idx, T_trial))

                if end_idx > start_idx:
                    if plot_3d:
                        segment = pca_data[i, start_idx:end_idx, :3]
                        segments_3d.append(segment)
                        colors_3d.append(color)
                        alphas_3d.append(seg_spec.get('alpha', 1.0))
                    else:
                        segment = pca_data[i, start_idx:end_idx, :2]
                        segments_2d.append(segment)
                        colors_2d.append(color)
                        alphas_2d.append(seg_spec.get('alpha', 1.0))
        else:
            # Uniform rendering (no segments)
            if plot_3d:
                segment = pca_data[i, :T_trial, :3]
                segments_3d.append(segment)
                colors_3d.append(color)
                alphas_3d.append(0.6)
            else:
                segment = pca_data[i, :T_trial, :2]
                segments_2d.append(segment)
                colors_2d.append(color)
                alphas_2d.append(0.6)

        # Add start/end markers
        if plot_3d:
            start_points.append([pca_data[i, 0, 0], pca_data[i, 0, 1], pca_data[i, 0, 2]])
            end_points.append([pca_data[i, T_trial-1, 0], pca_data[i, T_trial-1, 1], pca_data[i, T_trial-1, 2]])
        else:
            start_points.append([pca_data[i, 0, 0], pca_data[i, 0, 1]])
            end_points.append([pca_data[i, T_trial-1, 0], pca_data[i, T_trial-1, 1]])

        marker_colors.append(color)

        # Track label for legend (only add once per unique color_val)
        if label is not None and label not in legend_handles:
            legend_handles[label] = color

    # Render all lines with variable alpha
    if plot_3d:
        for segment, color, alpha in zip(segments_3d, colors_3d, alphas_3d):
            line_collection = Line3DCollection([segment], colors=[color],
                                              linewidths=1.5, alpha=alpha)
            ax.add_collection(line_collection)
    else:
        for segment, color, alpha in zip(segments_2d, colors_2d, alphas_2d):
            line_collection = LineCollection([segment], colors=[color],
                                            linewidths=1.5, alpha=alpha)
            ax.add_collection(line_collection)

    # Add markers
    start_points = np.array(start_points)
    end_points = np.array(end_points)

    if plot_3d:
        ax.scatter(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                  c=marker_colors, s=50, marker='*', alpha=0.8, zorder=5)
        ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2],
                  c=marker_colors, s=50, marker='x', alpha=0.8, zorder=5)
    else:
        ax.scatter(start_points[:, 0], start_points[:, 1],
                  c=marker_colors, s=50, marker='*', alpha=0.8, zorder=5)
        ax.scatter(end_points[:, 0], end_points[:, 1],
                  c=marker_colors, s=50, marker='x', alpha=0.8, zorder=5)

    # Autoscale to fit all data (collections + scatter points)
    if not plot_3d:
        ax.autoscale_view()
    else:
        # For 3D plots, manually set limits to include all trajectory data
        # Collect all data from trajectories to find global bounds
        all_data = []
        for i in range(num_trials):
            T_trial = min(trial_lengths[i], pca_data.shape[1])
            all_data.append(pca_data[i, :T_trial, :3])
        all_data = np.vstack(all_data)

        padding = 0.5
        ax.set_xlim(all_data[:, 0].min() - padding, all_data[:, 0].max() + padding)
        ax.set_ylim(all_data[:, 1].min() - padding, all_data[:, 1].max() + padding)
        ax.set_zlim(all_data[:, 2].min() - padding, all_data[:, 2].max() + padding)

    # Labels (use axis_labels from result dict)
    if plot_3d:
        ax.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Dim 0')
        ax.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Dim 1')
        ax.set_zlabel(axis_labels[2] if len(axis_labels) > 2 else 'Dim 2')
    else:
        ax.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Dim 0')
        ax.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Dim 1')

    # Title
    title = f'Hidden State Trajectories in PC Space ({"3D" if plot_3d else "2D"})'
    ax.set_title(title)

    # Add legend or colorbar
    if use_continuous and metadata is not None and color_by in metadata:
        # Add colorbar
        cbar = plt.colorbar(scalar_map, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(labels, rotation=270, labelpad=20)
    elif legend_handles:
        # Create manual legend patches for discrete colors
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color, linewidth=2, label=label)
                          for label, color in legend_handles.items()]
        ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig, ax


def animate_pca(result, period_name='', plot_simultaneously=True,
                plot_3d=True, color_by='rule', num_trials=None, interval=50,
                figsize=(14, 10), save_path=None):
    """
    Create animated PCA visualization (2D or 3D).

    Args:
        result: dict from do_pca() containing pca_data, axis_labels, metadata, events, lengths
        period_name: str, name of period being animated (for title)
        plot_simultaneously: bool, whether all trials evolve together
        plot_3d: bool, whether to animate in 3D (default True)
        color_by: str - what to color by
                       Discrete: 'rule', 'decision', 'stim_direction', 'instructed', 'switch', 'reward'
                       Continuous (with colorbar): 't_m', 't_s'
        num_trials: Number of trials to animate (default None = animate all trials)
        interval: Animation interval in ms
        figsize: tuple of figure size
        save_path: Optional path to save animation (as .mp4 or .gif)

    Returns:
        FuncAnimation object
    """
    # Extract from result dict
    pca_data = result['pca_data']
    axis_labels = result['axis_labels']
    period_lengths = result['lengths']
    metadata = result['metadata']
    import matplotlib
    matplotlib.rcParams['animation.embed_limit'] = 2**128

    # Set up color scheme
    color_map, labels = _get_color_scheme(color_by, metadata)

    # Limit number of trials (None = all trials)
    if num_trials is None:
        num_trials = len(period_lengths)
    else:
        num_trials = min(num_trials, len(period_lengths))

    # Check if using continuous colormap
    use_continuous = color_map == 'continuous'

    if use_continuous and metadata is not None and color_by in metadata:
        # Set up continuous colormap
        import matplotlib.cm as cm
        cmap = cm.viridis
        color_values = metadata[color_by][:num_trials]
        vmin, vmax = color_values.min(), color_values.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 20, height_ratios=[0.1, 1])

    # Time display panel
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.set_xlim(0, 1)
    ax_time.set_ylim(0, 1)
    ax_time.axis('off')

    # Main plot (2D or 3D)
    if plot_3d:
        ax_main = fig.add_subplot(gs[1, :], projection='3d')
        ax_main.set_xlim(pca_data[:, :, 0].min() - 0.5, pca_data[:, :, 0].max() + 0.5)
        ax_main.set_ylim(pca_data[:, :, 1].min() - 0.5, pca_data[:, :, 1].max() + 0.5)
        ax_main.set_zlim(pca_data[:, :, 2].min() - 0.5, pca_data[:, :, 2].max() + 0.5)
        ax_main.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Dim 0')
        ax_main.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Dim 1')
        ax_main.set_zlabel(axis_labels[2] if len(axis_labels) > 2 else 'Dim 2')
    else:
        ax_main = fig.add_subplot(gs[1, :])
        ax_main.set_xlim(pca_data[:, :, 0].min() - 0.5, pca_data[:, :, 0].max() + 0.5)
        ax_main.set_ylim(pca_data[:, :, 1].min() - 0.5, pca_data[:, :, 1].max() + 0.5)
        ax_main.set_xlabel(axis_labels[0] if len(axis_labels) > 0 else 'Dim 0')
        ax_main.set_ylabel(axis_labels[1] if len(axis_labels) > 1 else 'Dim 1')

    title = f'RNN Trajectories ({"3D" if plot_3d else "2D"}) - All Trials Evolving Together' if plot_simultaneously else f'RNN Trajectories ({"3D" if plot_3d else "2D"}) - Sequential'
    if period_name:
        title = f'{period_name.upper()} Period - ' + title
    ax_main.set_title(title)

    # Time text
    time_text = ax_time.text(0.5, 0.5, '', ha='center', va='center',
                            fontsize=16, fontweight='bold')

    # Add legend or colorbar
    if use_continuous and metadata is not None and color_by in metadata:
        # Add colorbar
        cbar = plt.colorbar(scalar_map, ax=ax_main, pad=0.1, shrink=0.6)
        cbar.set_label(labels, rotation=270, labelpad=20)
    else:
        # Add discrete legend
        legend_patches = []
        for val, label in labels.items():
            if val in color_map:
                legend_patches.append(mpatches.Patch(color=color_map[val], label=label, alpha=0.7))
        if legend_patches:
            ax_main.legend(handles=legend_patches, loc='upper right')

    # Initial view angle
    initial_elev = 20
    initial_azim = 45

    # Create line objects for each trial
    lines = []
    current_points = []
    end_markers = []

    for i in range(num_trials):
        # Get color for this trial
        if use_continuous and metadata is not None and color_by in metadata:
            color_val = metadata[color_by][i]
            color = scalar_map.to_rgba(color_val)
        elif metadata is not None and color_by in metadata:
            color_val = metadata[color_by][i]
            color = color_map.get(color_val, 'gray')
        else:
            color = f'C{i % 10}'

        # Get trial end position
        period_len = min(period_lengths[i], pca_data.shape[1])

        if plot_3d:
            line, = ax_main.plot([], [], [], color=color, alpha=0.7, linewidth=2)
            lines.append(line)
            point = ax_main.scatter([], [], [], color=color, s=100, marker='o',
                                   edgecolors='black', linewidths=2, zorder=10)
            current_points.append(point)
            # Add static end marker (x)
            end_marker = ax_main.scatter([pca_data[i, period_len-1, 0]],
                                        [pca_data[i, period_len-1, 1]],
                                        [pca_data[i, period_len-1, 2]],
                                        color=color, s=60, marker='x',
                                        linewidths=2, alpha=0.8, zorder=5)
            end_markers.append(end_marker)
        else:
            line, = ax_main.plot([], [], color=color, alpha=0.7, linewidth=2)
            lines.append(line)
            point, = ax_main.plot([], [], 'o', color=color, markersize=10,
                                 markeredgecolor='black', markeredgewidth=2, zorder=10)
            current_points.append(point)
            # Add static end marker (x)
            end_marker = ax_main.scatter([pca_data[i, period_len-1, 0]],
                                        [pca_data[i, period_len-1, 1]],
                                        color=color, s=60, marker='x',
                                        linewidths=2, alpha=0.8, zorder=5)
            end_markers.append(end_marker)

    # Find maximum length
    max_length = max(period_lengths[:num_trials])

    def init():
        for line in lines:
            line.set_data([], [])
            if hasattr(line, 'set_3d_properties'):
                line.set_3d_properties([])
        for point in current_points:
            if hasattr(point, '_offsets3d'):
                point._offsets3d = ([], [], [])
            else:
                point.set_data([], [])
        time_text.set_text('')
        if hasattr(ax_main, 'view_init'):
            ax_main.view_init(elev=initial_elev, azim=initial_azim)
        return lines + current_points + [time_text]

    def animate(frame):
        # Update each trial's trajectory up to current time
        for i in range(num_trials):
            period_len = min(period_lengths[i], pca_data.shape[1])

            if frame < period_len:
                # Trial is still ongoing
                lines[i].set_data(pca_data[i, :frame+1, 0], pca_data[i, :frame+1, 1])

                if hasattr(lines[i], 'set_3d_properties'):
                    # 3D line
                    lines[i].set_3d_properties(pca_data[i, :frame+1, 2])
                    current_points[i]._offsets3d = ([pca_data[i, frame, 0]],
                                                   [pca_data[i, frame, 1]],
                                                   [pca_data[i, frame, 2]])
                else:
                    # 2D line
                    current_points[i].set_data([pca_data[i, frame, 0]], [pca_data[i, frame, 1]])
            elif frame >= period_len:
                # Trial has ended
                lines[i].set_data(pca_data[i, :period_len, 0], pca_data[i, :period_len, 1])

                if hasattr(lines[i], 'set_3d_properties'):
                    # 3D line
                    lines[i].set_3d_properties(pca_data[i, :period_len, 2])
                    current_points[i]._offsets3d = ([], [], [])
                else:
                    # 2D line
                    current_points[i].set_data([], [])

        # Update time display (assuming 10ms per step)
        current_time_ms = frame * 10
        if period_name:
            time_text.set_text(f'{period_name.upper()}: {current_time_ms} ms from period start')
        else:
            time_text.set_text(f'Time: {current_time_ms} ms')

        # Rotate view slightly (only for 3D)
        if hasattr(ax_main, 'view_init'):
            ax_main.view_init(elev=initial_elev, azim=initial_azim + frame * 0.2)

        return lines + current_points + [time_text]

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=max_length,
                        interval=interval, blit=False, repeat=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if str(save_path).endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=20)
        elif str(save_path).endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=20)

        print(f"Saved animation to {save_path}")

    return anim


def _get_color_scheme(color_by, metadata):
    """
    Get color mapping and labels for visualization.

    Returns:
        color_map: dict mapping values to colors (or 'continuous' for continuous colormaps)
        labels: dict mapping values to label strings (or colorbar label for continuous)
    """
    if color_by == 'rule':
        color_map = {1: 'C0', -1: 'C1'}
        labels = {1: 'Rule 1', -1: 'Rule 2'}
    elif color_by == 'decision':
        color_map = {1: 'purple', -1: 'orange'}
        labels = {1: 'Right', -1: 'Left'}
    elif color_by == 'stim_direction':
        color_map = {1: 'purple', -1: 'orange'}
        labels = {1: 'Right', -1: 'Left'}
    elif color_by == 't_m':
        color_map = 'continuous'
        labels = 'Measured interval t_m (ms)'
    elif color_by == 't_s':
        color_map = 'continuous'
        labels = 'True interval t_s (ms)'
    elif color_by == 'instructed':
        color_map = {True: 'blue', False: 'gray'}
        labels = {True: 'Instructed', False: 'Uninstructed'}
    elif color_by == 'switch':
        color_map = {True: 'red', False: 'blue'}
        labels = {True: 'Switch', False: 'No Switch'}
    elif color_by == 'reward':
        color_map = {1: 'gold', 0: 'gray'}
        labels = {1: 'Rewarded', 0: 'Not Rewarded'}
    else:
        # Default coloring
        color_map = {}
        labels = {}

    return color_map, labels
