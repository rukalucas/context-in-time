import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Neural Trajectory Analysis - Interactive Notebook

    Explore RNN hidden state dynamics with PCA visualizations and analysis tools.
    """
    )
    return


@app.cell
def _():
    import torch
    import numpy as np

    from tasks import SingleTrialTask
    from models.rnn import RNN
    from analysis import generate_data, do_pca
    from analysis.plotting_px import (
        visualize_pca,
        animate_pca,
        plot_cross_period_variance,
    )

    return (
        RNN,
        SingleTrialTask,
        animate_pca,
        do_pca,
        generate_data,
        np,
        plot_cross_period_variance,
        torch,
        visualize_pca,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Setup: Load Model and Generate Data")
    return


@app.cell
def _(mo):
    # Configuration controls
    checkpoint_path = mo.ui.text(
        value="logs/single_trial/checkpoints/2000.pt", label="Checkpoint path"
    )

    num_trials = mo.ui.slider(
        start=20, stop=200, step=10, value=80, label="Number of trials"
    )

    mo.vstack([checkpoint_path, num_trials])
    return checkpoint_path, num_trials


@app.cell
def _(RNN, SingleTrialTask, checkpoint_path, generate_data, mo, num_trials, torch):
    try:
        checkpoint = torch.load(checkpoint_path.value, map_location="cpu")

        task = SingleTrialTask(w_m=0.05)
        model = RNN(hidden_size=128, noise_std=0.0)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Generate data
        data_dict = generate_data(task, model, num_trials=num_trials.value)

        setup_status = mo.md(
            f"""
        ✓ **Model loaded successfully!**
        - Checkpoint step: {checkpoint["step"]}
        - Hidden units: {model.hidden_size}
        - Generated trials: {num_trials.value}
        """
        )

    except Exception as e:
        setup_status = mo.md(f"**Error loading checkpoint:** {str(e)}")
        data_dict = None
        task = None
        model = None

    setup_status
    return checkpoint, data_dict, model, setup_status, task


@app.cell(hide_code=True)
def _(mo):
    mo.md("## PCA Visualization")
    return


@app.cell
def _(mo):
    # PCA controls
    pca_controls = {
        "n_components": mo.ui.slider(
            start=2,
            stop=10,
            step=1,
            value=3,
            label="Number of components",
            show_value=True,
        ),
        "plot_3d": mo.ui.checkbox(value=True, label="3D plot"),
        "animate": mo.ui.checkbox(value=False, label="Animate over time"),
        "show_trajectories": mo.ui.checkbox(
            value=True, label="Show trajectory lines (animation)"
        ),
        "num_trials_viz": mo.ui.slider(
            start=10,
            stop=80,
            step=10,
            value=30,
            label="Trials to visualize",
            show_value=True,
        ),
        "color_by": mo.ui.dropdown(
            options=["rule", "decision", "stim_direction", "t_m", "t_s"],
            value="rule",
            label="Color by",
        ),
        "period": mo.ui.dropdown(
            options=["all", "rule_report", "timing", "decision"],
            value="all",
            label="Period to analyze",
        ),
        "projection": mo.ui.dropdown(
            options=["none", "output_dim_0", "output_dim_1"],
            value="none",
            label="Projection axis",
        ),
    }

    mo.hstack(
        [
            mo.vstack(
                [
                    pca_controls["n_components"],
                    pca_controls["num_trials_viz"],
                    pca_controls["period"],
                ]
            ),
            mo.vstack(
                [
                    pca_controls["plot_3d"],
                    pca_controls["animate"],
                    pca_controls["show_trajectories"],
                ]
            ),
            mo.vstack([pca_controls["color_by"], pca_controls["projection"]]),
        ]
    )
    return (pca_controls,)


@app.cell
def _(
    animate_pca,
    data_dict,
    do_pca,
    mo,
    model,
    pca_controls,
    task,
    visualize_pca,
):
    if data_dict is not None and task is not None:
        # Run PCA
        projection_val = (
            None
            if pca_controls["projection"].value == "none"
            else pca_controls["projection"].value
        )

        pca_result = do_pca(
            data_dict,
            task,
            model=model,
            periods=pca_controls["period"].value,
            projection=projection_val,
            n_components=pca_controls["n_components"].value,
        )

        # Create visualization
        if pca_controls["animate"].value:
            pca_fig = animate_pca(
                pca_result,
                plot_3d=pca_controls["plot_3d"].value,
                num_trials=pca_controls["num_trials_viz"].value,
                color_by=pca_controls["color_by"].value,
                interval=50,
                show_trajectories=pca_controls["show_trajectories"].value,
            )
        else:
            pca_fig = visualize_pca(
                pca_result,
                plot_3d=pca_controls["plot_3d"].value,
                num_trials=pca_controls["num_trials_viz"].value,
                color_by=pca_controls["color_by"].value,
            )

        pca_info = mo.md(
            f"""
        **PCA Info:**
        - Shape: {pca_result["pca_data"].shape}
        - Axes: {pca_result["axis_labels"]}
        - Variance explained: {[f"{v:.1%}" for v in pca_result["explained_variance"]]}
        """
        )

        mo.vstack([pca_info, mo.ui.plotly(pca_fig)])
    else:
        mo.md("*Load data first*")
    return pca_fig, pca_info, pca_result, projection_val


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Segment-Based Visualization")
    return


@app.cell
def _(mo):
    # Segment controls
    use_segments = mo.ui.checkbox(value=False, label="Enable segment visualization")

    segment_period = mo.ui.dropdown(
        options=["timing", "all"], value="timing", label="Period for segments"
    )

    mo.hstack([use_segments, segment_period])
    return segment_period, use_segments


@app.cell
def _(
    data_dict,
    do_pca,
    mo,
    model,
    pca_controls,
    segment_period,
    task,
    use_segments,
    visualize_pca,
):
    if use_segments.value and data_dict is not None and task is not None:
        # Run PCA on selected period
        seg_result = do_pca(
            data_dict,
            task,
            model=model,
            periods=segment_period.value,
            n_components=3,
        )

        # Define segments based on events
        segments = [
            {
                "start": "timing_start",
                "end": "first_pulse",
                "alpha": 0.4,
                "label": "Pre-pulse",
            },
            {
                "start": "first_pulse",
                "end": "decision_start",
                "alpha": 1.0,
                "label": "Measurement",
            },
            {
                "start": "decision_start",
                "end": "trial_end",
                "alpha": 0.6,
                "label": "Decision",
            },
        ]

        seg_fig = visualize_pca(
            seg_result,
            segments=segments,
            plot_3d=pca_controls["plot_3d"].value,
            num_trials=20,
            color_by="rule",
            title=f"{segment_period.value.title()} Period - Segmented by Trial Events",
        )

        mo.ui.plotly(seg_fig)
    elif use_segments.value:
        mo.md("*Load data first*")
    else:
        mo.md("*Enable segment visualization to see segmented trajectories*")
    return seg_fig, seg_result, segments


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Cross-Period Variance Analysis

    How well do PCs from one period explain variance in other periods?
    """
    )
    return


@app.cell
def _(mo):
    n_comp_cross = mo.ui.slider(
        start=2,
        stop=10,
        step=1,
        value=3,
        label="Components for cross-period",
        show_value=True,
    )
    n_comp_cross
    return (n_comp_cross,)


@app.cell
def _(data_dict, mo, n_comp_cross, plot_cross_period_variance, task):
    if data_dict is not None and task is not None:
        cross_fig = plot_cross_period_variance(
            data_dict, task, n_components=n_comp_cross.value
        )
        mo.ui.plotly(cross_fig)
    else:
        mo.md("*Load data first*")
    return (cross_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Variance Explained by Components

    How many principal components are needed to capture neural variance?
    """
    )
    return


@app.cell
def _(mo):
    n_comp_var = mo.ui.slider(
        start=5,
        stop=50,
        step=5,
        value=20,
        label="Components to analyze",
        show_value=True,
    )
    n_comp_var
    return (n_comp_var,)


@app.cell
def _(data_dict, do_pca, mo, n_comp_var, np, task):
    import plotly.express as px
    import pandas as pd

    if data_dict is not None and task is not None:
        # Perform PCA with many components
        var_result = do_pca(
            data_dict, task, periods="all", n_components=n_comp_var.value
        )

        # Extract variance explained
        var_explained = var_result["explained_variance"]
        var_explained_pct = 100 * var_explained / var_explained.sum()
        cumulative_var = np.cumsum(var_explained_pct)

        # Summary statistics
        n_90 = (
            np.argmax(cumulative_var >= 90) + 1
            if np.any(cumulative_var >= 90)
            else len(cumulative_var)
        )
        n_95 = (
            np.argmax(cumulative_var >= 95) + 1
            if np.any(cumulative_var >= 95)
            else len(cumulative_var)
        )

        var_summary = mo.md(
            f"""
        **Dimensionality Summary:**
        - Components for 90% variance: {n_90}
        - Components for 95% variance: {n_95}
        - Top 5 PCs: {cumulative_var[4]:.1f}% variance
        - Top 10 PCs: {cumulative_var[9]:.1f}% variance
        """
        )

        # Create plots
        x_vals = np.arange(1, len(var_explained_pct) + 1)

        fig_bar = px.bar(
            pd.DataFrame({"PC": x_vals, "Variance (%)": var_explained_pct}),
            x="PC",
            y="Variance (%)",
            title="Variance Explained by Each PC",
        )
        fig_bar.update_layout(height=400)

        fig_cum = px.line(
            pd.DataFrame(
                {"Components": x_vals, "Cumulative Variance (%)": cumulative_var}
            ),
            x="Components",
            y="Cumulative Variance (%)",
            title="Cumulative Variance Explained",
            markers=True,
        )
        fig_cum.add_hline(
            y=90, line_dash="dash", line_color="red", opacity=0.5, annotation_text="90%"
        )
        fig_cum.add_hline(
            y=95,
            line_dash="dash",
            line_color="orange",
            opacity=0.5,
            annotation_text="95%",
        )
        fig_cum.update_layout(height=400, yaxis_range=[0, 105])

        mo.vstack(
            [var_summary, mo.hstack([mo.ui.plotly(fig_bar), mo.ui.plotly(fig_cum)])]
        )
    else:
        mo.md("*Load data first*")
    return (
        cumulative_var,
        fig_bar,
        fig_cum,
        n_90,
        n_95,
        pd,
        px,
        var_explained,
        var_explained_pct,
        var_result,
        var_summary,
        x_vals,
    )


if __name__ == "__main__":
    app.run()
