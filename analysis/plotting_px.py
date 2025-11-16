"""Plotly Express plotting utilities for neural trajectory visualization in marimo notebooks."""

import numpy as np
import pandas as pd
import plotly.express as px

from analysis.pca import do_pca
from analysis.utils import _extract_period_boundaries


def _get_color_scheme(color_by, metadata):
    """
    Get color mapping and labels for visualization.

    Returns:
        color_scheme: dict with 'type' ('discrete' or 'continuous'),
                     'map' (color mapping), 'labels' (label mapping),
                     'colorbar_title' (for continuous)
    """
    if color_by == "rule":
        return {
            "type": "discrete",
            "map": {1: "Rule 1", -1: "Rule 2"},
            "labels": {1: "Rule 1", -1: "Rule 2"},
        }
    elif color_by == "decision":
        return {
            "type": "discrete",
            "map": {1: "Right", -1: "Left"},
            "labels": {1: "Right", -1: "Left"},
        }
    elif color_by == "stim_direction":
        return {
            "type": "discrete",
            "map": {1: "Right", -1: "Left"},
            "labels": {1: "Right", -1: "Left"},
        }
    elif color_by == "t_m":
        return {
            "type": "continuous",
            "colorbar_title": "Measured interval t_m (ms)",
        }
    elif color_by == "t_s":
        return {
            "type": "continuous",
            "colorbar_title": "True interval t_s (ms)",
        }
    elif color_by == "instructed":
        return {
            "type": "discrete",
            "map": {True: "Instructed", False: "Uninstructed"},
            "labels": {True: "Instructed", False: "Uninstructed"},
        }
    elif color_by == "switch":
        return {
            "type": "discrete",
            "map": {True: "Switch", False: "No Switch"},
            "labels": {True: "Switch", False: "No Switch"},
        }
    elif color_by == "reward":
        return {
            "type": "discrete",
            "map": {1: "Rewarded", 0: "Not Rewarded"},
            "labels": {1: "Rewarded", 0: "Not Rewarded"},
        }
    else:
        # Default: trial index
        return {"type": "trial", "map": {}, "labels": {}}


def visualize_pca(
    result,
    segments=None,
    plot_3d=False,
    num_trials=None,
    color_by="rule",
    width=900,
    height=700,
    title=None,
):
    """
    Create static PCA visualization (2D or 3D) with optional segmentation.

    Args:
        result: dict from do_pca() containing pca_data, axis_labels, metadata, events, lengths
        segments: Optional list of segment specs (note: variable alpha not supported in pure px)
        plot_3d: bool, whether to plot in 3D
        num_trials: Number of trials to plot (default None = plot all trials)
        color_by: str - what to color by
                 Discrete: 'rule', 'decision', 'stim_direction', 'instructed', 'switch', 'reward'
                 Continuous: 't_m', 't_s'
        width: Figure width in pixels
        height: Figure height in pixels
        title: Optional custom title

    Returns:
        plotly.graph_objects.Figure
    """
    pca_data = result["pca_data"]
    axis_labels = result["axis_labels"]
    lengths = result["lengths"]
    metadata = result["metadata"]
    color_scheme = _get_color_scheme(color_by, metadata)

    # Limit trials
    if num_trials is None:
        num_trials = len(lengths)
    else:
        num_trials = min(num_trials, len(lengths))

    # Build dataframe
    rows = []

    for trial_idx in range(num_trials):
        trial_len = int(lengths[trial_idx])

        # Determine color value for this trial
        if color_scheme["type"] == "continuous":
            color_val = float(metadata[color_by][trial_idx])
        elif color_scheme["type"] == "discrete":
            raw_val = metadata[color_by][trial_idx]
            color_val = color_scheme["map"].get(raw_val, f"Unknown ({raw_val})")
        else:
            # Trial-based coloring
            color_val = f"Trial {trial_idx + 1}"

        # Add trajectory points
        for t in range(trial_len):
            row = {
                "trial": trial_idx,
                "trial_group": f"trial_{trial_idx}",  # For line grouping
                "timestep": t,
                axis_labels[0]: pca_data[trial_idx, t, 0],
                axis_labels[1]: pca_data[trial_idx, t, 1],
                "color": color_val,
                "marker_type": "trajectory",
            }

            if pca_data.shape[2] >= 3:
                row[axis_labels[2]] = pca_data[trial_idx, t, 2]

            rows.append(row)

        # Add start marker
        rows.append(
            {
                "trial": trial_idx,
                "trial_group": f"trial_{trial_idx}",
                "timestep": -1,  # Before trajectory
                axis_labels[0]: pca_data[trial_idx, 0, 0],
                axis_labels[1]: pca_data[trial_idx, 0, 1],
                axis_labels[2]: pca_data[trial_idx, 0, 2]
                if pca_data.shape[2] >= 3
                else None,
                "color": color_val,
                "marker_type": "start",
            }
        )

        # Add end marker
        rows.append(
            {
                "trial": trial_idx,
                "trial_group": f"trial_{trial_idx}",
                "timestep": trial_len,  # After trajectory
                axis_labels[0]: pca_data[trial_idx, trial_len - 1, 0],
                axis_labels[1]: pca_data[trial_idx, trial_len - 1, 1],
                axis_labels[2]: pca_data[trial_idx, trial_len - 1, 2]
                if pca_data.shape[2] >= 3
                else None,
                "color": color_val,
                "marker_type": "end",
            }
        )

    df = pd.DataFrame(rows)

    # Create figure
    if plot_3d and len(axis_labels) >= 3:
        fig = px.line_3d(
            df[df["marker_type"] == "trajectory"],
            x=axis_labels[0],
            y=axis_labels[1],
            z=axis_labels[2],
            color="color",
            line_group="trial_group",
            hover_data=["trial", "timestep"],
            title=title or "Hidden State Trajectories in PC Space (3D)",
            width=width,
            height=height,
        )
        fig.update_traces(opacity=0.7, line=dict(width=2))

        # Add start markers
        fig_start = px.scatter_3d(
            df[df["marker_type"] == "start"],
            x=axis_labels[0],
            y=axis_labels[1],
            z=axis_labels[2],
            color="color",
            hover_data=["trial"],
        )
        fig_start.update_traces(
            marker=dict(size=6, symbol="diamond", line=dict(width=1, color="black")),
            showlegend=False,
        )
        for trace in fig_start.data:
            fig.add_trace(trace)

        # Add end markers
        fig_end = px.scatter_3d(
            df[df["marker_type"] == "end"],
            x=axis_labels[0],
            y=axis_labels[1],
            z=axis_labels[2],
            color="color",
            hover_data=["trial"],
        )
        fig_end.update_traces(
            marker=dict(size=6, symbol="x", line=dict(width=2)), showlegend=False
        )
        for trace in fig_end.data:
            fig.add_trace(trace)

    else:
        # 2D plot
        fig = px.line(
            df[df["marker_type"] == "trajectory"],
            x=axis_labels[0],
            y=axis_labels[1],
            color="color",
            line_group="trial_group",
            hover_data=["trial", "timestep"],
            title=title or "Hidden State Trajectories in PC Space (2D)",
            width=width,
            height=height,
        )
        fig.update_traces(opacity=0.7, line=dict(width=2))

        # Add start markers
        fig_start = px.scatter(
            df[df["marker_type"] == "start"],
            x=axis_labels[0],
            y=axis_labels[1],
            color="color",
            hover_data=["trial"],
        )
        fig_start.update_traces(
            marker=dict(size=8, symbol="diamond", line=dict(width=1, color="black")),
            showlegend=False,
        )
        for trace in fig_start.data:
            fig.add_trace(trace)

        # Add end markers
        fig_end = px.scatter(
            df[df["marker_type"] == "end"],
            x=axis_labels[0],
            y=axis_labels[1],
            color="color",
            hover_data=["trial"],
        )
        fig_end.update_traces(
            marker=dict(size=8, symbol="x", line=dict(width=2)), showlegend=False
        )
        for trace in fig_end.data:
            fig.add_trace(trace)

    return fig


def animate_pca(
    result,
    plot_3d=True,
    num_trials=None,
    color_by="rule",
    width=900,
    height=700,
    title=None,
    show_trajectories=True,
):
    """
    Create animated PCA visualization showing trajectories evolving over time.

    Note: With plotly express, animation shows cumulative trajectory up to each frame.
    The show_trajectories parameter toggles between showing full trajectory or just current point.

    Args:
        result: dict from do_pca() containing pca_data, axis_labels, metadata, lengths
        plot_3d: bool, whether to animate in 3D (default True)
        num_trials: Number of trials to animate (default None = animate all trials)
        color_by: str - what to color by
        width: Figure width in pixels
        height: Figure height in pixels
        title: Optional custom title
        show_trajectories: If True, show cumulative trajectory; if False, show only current point

    Returns:
        plotly.graph_objects.Figure with animation
    """
    pca_data = result["pca_data"]
    axis_labels = result["axis_labels"]
    lengths = result["lengths"]
    metadata = result["metadata"]
    color_scheme = _get_color_scheme(color_by, metadata)

    # Limit trials
    if num_trials is None:
        num_trials = len(lengths)
    else:
        num_trials = min(num_trials, len(lengths))

    max_length = max(int(lengths[i]) for i in range(num_trials))

    # Build dataframe for animation
    rows = []

    for trial_idx in range(num_trials):
        trial_len = int(lengths[trial_idx])

        # Determine color value for this trial
        if color_scheme["type"] == "continuous":
            color_val = float(metadata[color_by][trial_idx])
        elif color_scheme["type"] == "discrete":
            raw_val = metadata[color_by][trial_idx]
            color_val = color_scheme["map"].get(raw_val, f"Unknown ({raw_val})")
        else:
            color_val = f"Trial {trial_idx + 1}"

        # For each frame, add data up to that point
        for frame in range(max_length):
            if show_trajectories:
                # Show trajectory up to current frame
                for t in range(min(frame + 1, trial_len)):
                    row = {
                        "trial": trial_idx,
                        "trial_group": f"trial_{trial_idx}",
                        "frame": frame,
                        "timestep": t,
                        "time_ms": frame * 10,  # Assuming 10ms timesteps
                        axis_labels[0]: pca_data[trial_idx, t, 0],
                        axis_labels[1]: pca_data[trial_idx, t, 1],
                        "color": color_val,
                    }
                    if pca_data.shape[2] >= 3:
                        row[axis_labels[2]] = pca_data[trial_idx, t, 2]
                    rows.append(row)
            else:
                # Show only current point
                if frame < trial_len:
                    row = {
                        "trial": trial_idx,
                        "trial_group": f"trial_{trial_idx}",
                        "frame": frame,
                        "timestep": frame,
                        "time_ms": frame * 10,
                        axis_labels[0]: pca_data[trial_idx, frame, 0],
                        axis_labels[1]: pca_data[trial_idx, frame, 1],
                        "color": color_val,
                    }
                    if pca_data.shape[2] >= 3:
                        row[axis_labels[2]] = pca_data[trial_idx, frame, 2]
                    rows.append(row)

    df = pd.DataFrame(rows)

    # Create animated figure
    if plot_3d and pca_data.shape[2] >= 3:
        if show_trajectories:
            fig = px.line_3d(
                df,
                x=axis_labels[0],
                y=axis_labels[1],
                z=axis_labels[2],
                color="color",
                line_group="trial_group",
                animation_frame="frame",
                hover_data=["trial", "timestep"],
                title=title or "RNN Trajectories - Evolving Over Time (3D)",
                width=width,
                height=height,
                range_x=[pca_data[:, :, 0].min(), pca_data[:, :, 0].max()],
                range_y=[pca_data[:, :, 1].min(), pca_data[:, :, 1].max()],
                range_z=[pca_data[:, :, 2].min(), pca_data[:, :, 2].max()],
            )
            fig.update_traces(opacity=0.7, line=dict(width=3))
        else:
            fig = px.scatter_3d(
                df,
                x=axis_labels[0],
                y=axis_labels[1],
                z=axis_labels[2],
                color="color",
                animation_frame="frame",
                hover_data=["trial", "timestep"],
                title=title or "RNN Trajectories - Evolving Over Time (3D)",
                width=width,
                height=height,
                range_x=[pca_data[:, :, 0].min(), pca_data[:, :, 0].max()],
                range_y=[pca_data[:, :, 1].min(), pca_data[:, :, 1].max()],
                range_z=[pca_data[:, :, 2].min(), pca_data[:, :, 2].max()],
            )
            fig.update_traces(marker=dict(size=6, line=dict(width=1, color="white")))
    else:
        if show_trajectories:
            fig = px.line(
                df,
                x=axis_labels[0],
                y=axis_labels[1],
                color="color",
                line_group="trial_group",
                animation_frame="frame",
                hover_data=["trial", "timestep"],
                title=title or "RNN Trajectories - Evolving Over Time (2D)",
                width=width,
                height=height,
                range_x=[pca_data[:, :, 0].min(), pca_data[:, :, 0].max()],
                range_y=[pca_data[:, :, 1].min(), pca_data[:, :, 1].max()],
            )
            fig.update_traces(opacity=0.7, line=dict(width=3))
        else:
            fig = px.scatter(
                df,
                x=axis_labels[0],
                y=axis_labels[1],
                color="color",
                animation_frame="frame",
                hover_data=["trial", "timestep"],
                title=title or "RNN Trajectories - Evolving Over Time (2D)",
                width=width,
                height=height,
                range_x=[pca_data[:, :, 0].min(), pca_data[:, :, 0].max()],
                range_y=[pca_data[:, :, 1].min(), pca_data[:, :, 1].max()],
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))

    # Update animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0

    return fig


def plot_cross_period_variance(
    data_dict,
    task,
    period_names=None,
    n_components=3,
    width=700,
    height=700,
    title=None,
):
    """
    Plot heatmap showing how much variance in one period is explained by another period's PCs.

    Args:
        data_dict: dict from generate_data() containing 'hidden_states' and 'batch'
        task: Task instance
        period_names: list of period names to compare. Use 'iti' for ITI period.
                     Default None uses all available periods
        n_components: Number of PCs to use for each period
        width: Figure width in pixels
        height: Figure height in pixels
        title: Optional custom title

    Returns:
        plotly.graph_objects.Figure
    """
    hidden_states = data_dict["hidden_states"]
    batch = data_dict["batch"]

    # Set default period names
    if period_names is None:
        period_names = ["rule_report", "timing", "decision"]
        # Check if this is a sequence task with ITI
        if len(batch) > 1 or (len(batch) == 1 and "iti_start" in batch[0]["metadata"]):
            period_names.append("iti")

    # Map 'iti' to 'post_iti' for internal use
    internal_period_names = [p if p != "iti" else "post_iti" for p in period_names]

    # Create display labels
    display_labels = []
    for p in period_names:
        if p == "iti":
            display_labels.append("ITI")
        else:
            display_labels.append(p.replace("_", " ").title())

    n_periods = len(period_names)

    # Run do_pca for each period to get PCs and flattened data
    period_results = {}
    for internal_name in internal_period_names:
        result = do_pca(
            data_dict, task, periods=internal_name, n_components=n_components
        )

        pcs = result["pcs"]  # [H, n_comp]

        # Extract raw hidden data for this period
        num_trials = hidden_states.shape[0]
        period_info_dict = _extract_period_boundaries(batch, task, num_trials)
        period_boundaries = period_info_dict[internal_name]

        hidden_np = hidden_states.cpu().numpy()  # [N, H, T]

        # Flatten period data
        flat_data_list = []
        for i, info in enumerate(period_boundaries):
            start, end = info["start"], info["end"]
            period_len = end - start

            if period_len == 0:
                continue

            trial_idx = info.get("trial_idx", i)
            flat_data_list.append(hidden_np[trial_idx, :, start:end].T)

        hidden_flat = np.vstack(flat_data_list)  # [sum(lengths), H]

        period_results[internal_name] = {"pcs": pcs, "hidden_flat": hidden_flat}

    # Create variance explained matrix
    variance_matrix = np.zeros((n_periods, n_periods))

    for i, fit_period in enumerate(internal_period_names):
        fit_pcs = period_results[fit_period]["pcs"]  # (H, n_components)

        for j, test_period in enumerate(internal_period_names):
            test_data = period_results[test_period]["hidden_flat"]  # (N, H)

            # Center test data
            test_centered = test_data - test_data.mean(axis=0)

            # Project onto fit period's PCs
            test_projected = test_centered @ fit_pcs  # (N, n_components)

            # Compute variance explained
            projected_variance = np.sum(np.var(test_projected, axis=0))
            total_variance = np.sum(np.var(test_centered, axis=0))
            variance_explained = projected_variance / total_variance

            variance_matrix[i, j] = variance_explained * 100  # Convert to percentage

    # Create plotly heatmap
    fig = px.imshow(
        variance_matrix,
        labels=dict(x="Test Period", y="Fit Period", color="Variance (%)"),
        x=display_labels,
        y=display_labels,
        color_continuous_scale="viridis",
        zmin=0,
        zmax=100,
        text_auto=".1f",
        width=width,
        height=height,
        title=title or f"Cross-Period Variance Explained by Top {n_components} PCs (%)",
    )

    fig.update_xaxes(side="bottom")

    return fig
