"""Plotly Express plotting utilities for neural trajectory visualization in marimo notebooks."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.utils import _extract_period_boundaries
from analysis.pca import do_pca


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


def _prepare_trajectory_data(
    result,
    num_trials=None,
    color_by="rule",
    segments=None,
):
    """
    Prepare trajectory data for plotly visualization.

    Args:
        result: dict from do_pca()
        num_trials: Number of trials to include (None = all)
        color_by: What to color by
        segments: Optional list of segment specs for variable transparency

    Returns:
        pd.DataFrame with columns: trial, timestep, PC1, PC2, [PC3], color, [segment, alpha]
    """
    pca_data = result["pca_data"]
    axis_labels = result["axis_labels"]
    lengths = result["lengths"]
    metadata = result["metadata"]
    time_indices = result.get("time_indices", None)
    events = result.get("events", None)

    # Limit trials
    if num_trials is None:
        num_trials = len(lengths)
    else:
        num_trials = min(num_trials, len(lengths))

    # Get color scheme
    color_scheme = _get_color_scheme(color_by, metadata)

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

        # Add each timestep
        for t in range(trial_len):
            row = {
                "trial": trial_idx,
                "timestep": t,
                axis_labels[0]: pca_data[trial_idx, t, 0],
                axis_labels[1]: pca_data[trial_idx, t, 1],
                "color": color_val,
            }

            if pca_data.shape[2] >= 3:
                row[axis_labels[2]] = pca_data[trial_idx, t, 2]

            # Handle segments
            if segments is not None and events is not None and time_indices is not None:
                # Find which segment this timestep belongs to
                original_t = time_indices[trial_idx, t]
                segment_found = False

                for seg_idx, seg_spec in enumerate(segments):
                    # Get event indices in original trial coordinates
                    if (
                        seg_spec["start"] == "trial_end"
                        or seg_spec["start"] == "period_end"
                    ):
                        start_t = lengths[trial_idx]
                    elif seg_spec["start"] in events:
                        start_t = events[seg_spec["start"]][trial_idx]
                    else:
                        try:
                            start_t = int(seg_spec["start"])
                        except ValueError:
                            continue

                    if (
                        seg_spec["end"] == "trial_end"
                        or seg_spec["end"] == "period_end"
                    ):
                        end_t = lengths[trial_idx]
                    elif seg_spec["end"] in events:
                        end_t = events[seg_spec["end"]][trial_idx]
                    else:
                        try:
                            end_t = int(seg_spec["end"])
                        except ValueError:
                            continue

                    if start_t <= original_t < end_t:
                        row["segment"] = seg_spec.get(
                            "label", f"{seg_spec['start']}→{seg_spec['end']}"
                        )
                        row["alpha"] = seg_spec.get("alpha", 1.0)
                        segment_found = True
                        break

                if not segment_found:
                    row["segment"] = "Other"
                    row["alpha"] = 0.3

            rows.append(row)

    return pd.DataFrame(rows)


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
        segments: Optional list of segment specs:
                 [{'start': event_name, 'end': event_name, 'alpha': float, 'label': str}, ...]
                 Events can be: 'trial_start', 'cue_onset', 'first_pulse', 'decision_start', etc.
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
    axis_labels = result["axis_labels"]
    metadata = result["metadata"]
    color_scheme = _get_color_scheme(color_by, metadata)

    # Prepare data
    df = _prepare_trajectory_data(
        result, num_trials=num_trials, color_by=color_by, segments=segments
    )

    # Determine if we have segments with variable alpha
    has_segments = "segment" in df.columns

    # Create figure
    if plot_3d and len(axis_labels) >= 3:
        if has_segments:
            # Need to create separate traces for each segment to handle alpha
            fig = go.Figure()

            # Group by trial and segment
            for trial_idx in df["trial"].unique():
                trial_data = df[df["trial"] == trial_idx]
                color_val = trial_data["color"].iloc[0]

                # Get color (for continuous, use viridis scale)
                if color_scheme["type"] == "continuous":
                    # Map to viridis colorscale
                    norm_val = (color_val - df["color"].min()) / (
                        df["color"].max() - df["color"].min() + 1e-10
                    )
                    color = px.colors.sample_colorscale("viridis", [norm_val])[0]
                else:
                    # Use plotly default colors for discrete
                    color_idx = list(df["color"].unique()).index(color_val)
                    color = px.colors.qualitative.Plotly[
                        color_idx % len(px.colors.qualitative.Plotly)
                    ]

                for segment in trial_data["segment"].unique():
                    seg_data = trial_data[trial_data["segment"] == segment].sort_values(
                        "timestep"
                    )
                    alpha = seg_data["alpha"].iloc[0]

                    fig.add_trace(
                        go.Scatter3d(
                            x=seg_data[axis_labels[0]],
                            y=seg_data[axis_labels[1]],
                            z=seg_data[axis_labels[2]],
                            mode="lines",
                            line=dict(color=color, width=2),
                            opacity=alpha,
                            name=f"Trial {trial_idx}",
                            legendgroup=str(color_val),
                            showlegend=(segment == trial_data["segment"].iloc[0]),
                            hovertemplate=f"Trial: {trial_idx}<br>"
                            + f"{axis_labels[0]}: %{{x:.2f}}<br>"
                            + f"{axis_labels[1]}: %{{y:.2f}}<br>"
                            + f"{axis_labels[2]}: %{{z:.2f}}<br>"
                            + f"Segment: {segment}<extra></extra>",
                        )
                    )

            fig.update_layout(
                scene=dict(
                    xaxis_title=axis_labels[0],
                    yaxis_title=axis_labels[1],
                    zaxis_title=axis_labels[2],
                ),
                width=width,
                height=height,
                title=title or "Hidden State Trajectories in PC Space (3D)",
            )
        else:
            # Simple 3D line plot without segments
            fig = px.line_3d(
                df,
                x=axis_labels[0],
                y=axis_labels[1],
                z=axis_labels[2],
                color="color",
                line_group="trial",
                hover_data=["trial", "timestep"],
                title=title or "Hidden State Trajectories in PC Space (3D)",
                width=width,
                height=height,
            )
            fig.update_traces(opacity=0.7, line=dict(width=2))

    else:
        # 2D plot
        if has_segments:
            # Need to create separate traces for each segment to handle alpha
            fig = go.Figure()

            # Group by trial and segment
            for trial_idx in df["trial"].unique():
                trial_data = df[df["trial"] == trial_idx]
                color_val = trial_data["color"].iloc[0]

                # Get color
                if color_scheme["type"] == "continuous":
                    norm_val = (color_val - df["color"].min()) / (
                        df["color"].max() - df["color"].min() + 1e-10
                    )
                    color = px.colors.sample_colorscale("viridis", [norm_val])[0]
                else:
                    color_idx = list(df["color"].unique()).index(color_val)
                    color = px.colors.qualitative.Plotly[
                        color_idx % len(px.colors.qualitative.Plotly)
                    ]

                for segment in trial_data["segment"].unique():
                    seg_data = trial_data[trial_data["segment"] == segment].sort_values(
                        "timestep"
                    )
                    alpha = seg_data["alpha"].iloc[0]

                    fig.add_trace(
                        go.Scatter(
                            x=seg_data[axis_labels[0]],
                            y=seg_data[axis_labels[1]],
                            mode="lines",
                            line=dict(color=color, width=2),
                            opacity=alpha,
                            name=f"Trial {trial_idx}",
                            legendgroup=str(color_val),
                            showlegend=(segment == trial_data["segment"].iloc[0]),
                            hovertemplate=f"Trial: {trial_idx}<br>"
                            + f"{axis_labels[0]}: %{{x:.2f}}<br>"
                            + f"{axis_labels[1]}: %{{y:.2f}}<br>"
                            + f"Segment: {segment}<extra></extra>",
                        )
                    )

            fig.update_layout(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                width=width,
                height=height,
                title=title or "Hidden State Trajectories in PC Space (2D)",
            )
        else:
            # Simple 2D line plot without segments
            fig = px.line(
                df,
                x=axis_labels[0],
                y=axis_labels[1],
                color="color",
                line_group="trial",
                hover_data=["trial", "timestep"],
                title=title or "Hidden State Trajectories in PC Space (2D)",
                width=width,
                height=height,
            )
            fig.update_traces(opacity=0.7, line=dict(width=2))

    # Add start/end markers
    pca_data = result["pca_data"]
    lengths = result["lengths"]
    num_show = min(num_trials or len(lengths), len(lengths))

    start_points = []
    end_points = []
    colors = []

    for i in range(num_show):
        T_trial = int(lengths[i])
        color_val = df[df["trial"] == i]["color"].iloc[0]

        if plot_3d and len(axis_labels) >= 3:
            start_points.append(
                {
                    axis_labels[0]: pca_data[i, 0, 0],
                    axis_labels[1]: pca_data[i, 0, 1],
                    axis_labels[2]: pca_data[i, 0, 2],
                }
            )
            end_points.append(
                {
                    axis_labels[0]: pca_data[i, T_trial - 1, 0],
                    axis_labels[1]: pca_data[i, T_trial - 1, 1],
                    axis_labels[2]: pca_data[i, T_trial - 1, 2],
                }
            )
        else:
            start_points.append(
                {axis_labels[0]: pca_data[i, 0, 0], axis_labels[1]: pca_data[i, 0, 1]}
            )
            end_points.append(
                {
                    axis_labels[0]: pca_data[i, T_trial - 1, 0],
                    axis_labels[1]: pca_data[i, T_trial - 1, 1],
                }
            )

        colors.append(color_val)

    df_starts = pd.DataFrame(start_points)
    df_starts["color"] = colors
    df_ends = pd.DataFrame(end_points)
    df_ends["color"] = colors

    if plot_3d and len(axis_labels) >= 3:
        fig.add_trace(
            go.Scatter3d(
                x=df_starts[axis_labels[0]],
                y=df_starts[axis_labels[1]],
                z=df_starts[axis_labels[2]],
                mode="markers",
                marker=dict(
                    size=6,
                    symbol="diamond",
                    color="white",
                    line=dict(width=1, color="black"),
                ),
                name="Start",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=df_ends[axis_labels[0]],
                y=df_ends[axis_labels[1]],
                z=df_ends[axis_labels[2]],
                mode="markers",
                marker=dict(size=6, symbol="x", color="black"),
                name="End",
                showlegend=True,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df_starts[axis_labels[0]],
                y=df_starts[axis_labels[1]],
                mode="markers",
                marker=dict(
                    size=8,
                    symbol="diamond",
                    color="white",
                    line=dict(width=1, color="black"),
                ),
                name="Start",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_ends[axis_labels[0]],
                y=df_ends[axis_labels[1]],
                mode="markers",
                marker=dict(size=8, symbol="x", color="black"),
                name="End",
                showlegend=True,
            )
        )

    return fig


def animate_pca(
    result,
    plot_3d=True,
    num_trials=None,
    color_by="rule",
    interval=50,
    width=900,
    height=700,
    title=None,
    show_trajectories=True,
):
    """
    Create animated PCA visualization with trajectories evolving over time.

    Memory-efficient implementation: trajectories are drawn frame-by-frame
    showing the line being created in time.

    Args:
        result: dict from do_pca() containing pca_data, axis_labels, metadata, lengths
        plot_3d: bool, whether to animate in 3D (default True)
        num_trials: Number of trials to animate (default None = animate all trials)
        color_by: str - what to color by
                 Discrete: 'rule', 'decision', 'stim_direction', 'instructed', 'switch', 'reward'
                 Continuous: 't_m', 't_s'
        interval: Animation interval in ms (frame duration)
        width: Figure width in pixels
        height: Figure height in pixels
        title: Optional custom title
        show_trajectories: If True, show lines; if False, show only current point

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

    # Prepare color mapping
    color_values = []
    for i in range(num_trials):
        if color_scheme["type"] == "continuous":
            color_values.append(float(metadata[color_by][i]))
        elif color_scheme["type"] == "discrete":
            raw_val = metadata[color_by][i]
            color_values.append(
                color_scheme["map"].get(raw_val, f"Unknown ({raw_val})")
            )
        else:
            color_values.append(f"Trial {i + 1}")

    # Create frames
    frames = []

    for frame_idx in range(max_length):
        frame_data = []

        for trial_idx in range(num_trials):
            trial_len = int(lengths[trial_idx])

            if frame_idx < trial_len:
                # Show trajectory up to current frame
                end_idx = frame_idx + 1

                # Get color
                color_val = color_values[trial_idx]
                if color_scheme["type"] == "continuous":
                    norm_val = (color_val - min(color_values)) / (
                        max(color_values) - min(color_values) + 1e-10
                    )
                    color = px.colors.sample_colorscale("viridis", [norm_val])[0]
                else:
                    color_idx = list(set(color_values)).index(color_val)
                    color = px.colors.qualitative.Plotly[
                        color_idx % len(px.colors.qualitative.Plotly)
                    ]

                if plot_3d and pca_data.shape[2] >= 3:
                    if show_trajectories:
                        # Line trace showing trajectory
                        frame_data.append(
                            go.Scatter3d(
                                x=pca_data[trial_idx, :end_idx, 0],
                                y=pca_data[trial_idx, :end_idx, 1],
                                z=pca_data[trial_idx, :end_idx, 2],
                                mode="lines",
                                line=dict(color=color, width=3),
                                opacity=0.7,
                                name=str(color_val),
                                legendgroup=str(color_val),
                                showlegend=(
                                    trial_idx == 0
                                    or color_val
                                    not in [color_values[j] for j in range(trial_idx)]
                                ),
                                hoverinfo="skip",
                            )
                        )

                    # Current point marker
                    frame_data.append(
                        go.Scatter3d(
                            x=[pca_data[trial_idx, frame_idx, 0]],
                            y=[pca_data[trial_idx, frame_idx, 1]],
                            z=[pca_data[trial_idx, frame_idx, 2]],
                            mode="markers",
                            marker=dict(
                                size=6, color=color, line=dict(width=1, color="white")
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                else:
                    if show_trajectories:
                        # Line trace showing trajectory
                        frame_data.append(
                            go.Scatter(
                                x=pca_data[trial_idx, :end_idx, 0],
                                y=pca_data[trial_idx, :end_idx, 1],
                                mode="lines",
                                line=dict(color=color, width=3),
                                opacity=0.7,
                                name=str(color_val),
                                legendgroup=str(color_val),
                                showlegend=(
                                    trial_idx == 0
                                    or color_val
                                    not in [color_values[j] for j in range(trial_idx)]
                                ),
                                hoverinfo="skip",
                            )
                        )

                    # Current point marker
                    frame_data.append(
                        go.Scatter(
                            x=[pca_data[trial_idx, frame_idx, 0]],
                            y=[pca_data[trial_idx, frame_idx, 1]],
                            mode="markers",
                            marker=dict(
                                size=8, color=color, line=dict(width=1, color="white")
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    # Create initial figure (first frame)
    fig = go.Figure(data=frames[0].data, frames=frames)

    # Update layout
    if plot_3d and pca_data.shape[2] >= 3:
        fig.update_layout(
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
                xaxis=dict(
                    range=[pca_data[:, :, 0].min() - 0.5, pca_data[:, :, 0].max() + 0.5]
                ),
                yaxis=dict(
                    range=[pca_data[:, :, 1].min() - 0.5, pca_data[:, :, 1].max() + 0.5]
                ),
                zaxis=dict(
                    range=[pca_data[:, :, 2].min() - 0.5, pca_data[:, :, 2].max() + 0.5]
                ),
            ),
            width=width,
            height=height,
            title=title or "RNN Trajectories - Evolving Over Time (3D)",
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "steps": [
                        {
                            "args": [
                                [f"{i}"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": f"{i * 10}ms",
                            "method": "animate",
                        }
                        for i in range(max_length)
                    ],
                    "x": 0.1,
                    "len": 0.85,
                    "y": 0,
                }
            ],
        )
    else:
        fig.update_layout(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            xaxis=dict(
                range=[pca_data[:, :, 0].min() - 0.5, pca_data[:, :, 0].max() + 0.5]
            ),
            yaxis=dict(
                range=[pca_data[:, :, 1].min() - 0.5, pca_data[:, :, 1].max() + 0.5]
            ),
            width=width,
            height=height,
            title=title or "RNN Trajectories - Evolving Over Time (2D)",
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "steps": [
                        {
                            "args": [
                                [f"{i}"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": f"{i * 10}ms",
                            "method": "animate",
                        }
                        for i in range(max_length)
                    ],
                    "x": 0.1,
                    "len": 0.85,
                    "y": 0,
                }
            ],
        )

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
