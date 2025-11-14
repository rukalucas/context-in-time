import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("# Analysis Demo - PCA Notebook (Marimo)")
    return


@app.cell
def __():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    from tasks import InstructedTimingTask, SequenceInstructedTask
    from models.rnn import RNN
    from analysis import (
        generate_data,
        do_pca,
        visualize_pca,
        plot_cross_period_variance,
        animate_pca,
        compute_psychometric_curves,
    )

    plt.rcParams['figure.dpi'] = 100
    return (
        InstructedTimingTask,
        RNN,
        SequenceInstructedTask,
        animate_pca,
        compute_psychometric_curves,
        do_pca,
        generate_data,
        np,
        plot_cross_period_variance,
        plt,
        torch,
        visualize_pca,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Setup: Load Model and Generate Data")
    return


@app.cell
def __(mo):
    # Configuration controls
    checkpoint_selector = mo.ui.text(
        value='logs/transition_to_inferred/checkpoint_step_3200.pt',
        label='Checkpoint path:'
    )

    num_trials_slider = mo.ui.slider(
        start=20,
        stop=200,
        step=10,
        value=80,
        label='Number of trials to generate:'
    )

    trials_per_seq_slider = mo.ui.slider(
        start=10,
        stop=100,
        step=10,
        value=40,
        label='Trials per sequence:'
    )

    mo.vstack([checkpoint_selector, num_trials_slider, trials_per_seq_slider])
    return (
        checkpoint_selector,
        num_trials_slider,
        trials_per_seq_slider,
    )


@app.cell
def __(
    RNN,
    SequenceInstructedTask,
    checkpoint_selector,
    generate_data,
    mo,
    np,
    torch,
    trials_per_seq_slider,
    num_trials_slider,
):
    checkpoint_path = checkpoint_selector.value
    num_trials = num_trials_slider.value
    trials_per_sequence = trials_per_seq_slider.value

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        task = SequenceInstructedTask(
            w_m=0.05,
            input_noise_std=0.05,
            trials_per_sequence=trials_per_sequence
        )

        model = RNN(
            hidden_size=256,
            noise_std=0.0  # No noise during evaluation
        )

        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Generate data
        data_dict = generate_data(task, model, num_trials=num_trials)
        rules = np.array([
            data_dict['batch'][i]['metadata']['rule'][0]
            for i in range(len(data_dict['batch']))
        ])

        setup_status = mo.md(f"""
        **Model loaded successfully!**
        - Checkpoint step: {checkpoint['step']}
        - Hidden units: {model.hidden_size}
        - Trials per sequence: {task.trials_per_sequence}
        - Generated trials: {num_trials}
        - Rule 1 count: {(rules == 1).sum()}, Rule 2 count: {(rules == -1).sum()}
        """)

    except Exception as e:
        setup_status = mo.md(f"**Error loading checkpoint:** {str(e)}")
        data_dict = None
        task = None
        model = None
        rules = None

    setup_status
    return (
        checkpoint,
        checkpoint_path,
        data_dict,
        model,
        num_trials,
        rules,
        setup_status,
        task,
        trials_per_sequence,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Random Trial Visualization")
    return


@app.cell
def __(data_dict, mo, np):
    # Trial selector
    if data_dict is not None:
        trial_selector = mo.ui.slider(
            start=0,
            stop=len(data_dict['batch']) - 1,
            step=1,
            value=np.random.randint(0, len(data_dict['batch'])),
            label='Trial index:',
            show_value=True
        )

        randomize_button = mo.ui.button(label='Random Trial')
    else:
        trial_selector = None
        randomize_button = None

    mo.hstack([trial_selector, randomize_button]) if trial_selector else mo.md("*Load data first*")
    return randomize_button, trial_selector


@app.cell
def __(data_dict, mo, np, randomize_button, trial_selector):
    # Handle randomization
    if randomize_button is not None and randomize_button.value:
        trial_idx = np.random.randint(0, len(data_dict['batch']))
    elif trial_selector is not None:
        trial_idx = trial_selector.value
    else:
        trial_idx = 0

    mo.md(f"**Showing trial:** {trial_idx}")
    return (trial_idx,)


@app.cell
def __(data_dict, mo, plt, task, trial_idx):
    if data_dict is not None and task is not None:
        batch = data_dict['batch']
        outputs = data_dict['outputs']

        trial_outputs = outputs[trial_idx].numpy()
        trial_inputs = batch[trial_idx]['inputs'][0].numpy()
        trial_targets = batch[trial_idx]['targets'][0].numpy()
        trial_eval_mask = batch[trial_idx]['eval_mask'][0].numpy()
        trial_loss_mask = batch[trial_idx]['loss_mask'][0].numpy()

        # Create figure using task method
        trial_fig = task.create_trial_figure(
            inputs=trial_inputs,
            outputs=trial_outputs,
            targets=trial_targets,
            eval_mask=trial_eval_mask,
            trial_idx=trial_idx,
            batch=batch,
            batch_idx=0,
            loss_mask=trial_loss_mask,
        )

        trial_plot = mo.ui.pyplot(trial_fig)
        plt.close(trial_fig)
    else:
        trial_plot = mo.md("*Load data first*")

    trial_plot
    return (
        batch,
        outputs,
        trial_eval_mask,
        trial_fig,
        trial_inputs,
        trial_loss_mask,
        trial_outputs,
        trial_plot,
        trial_targets,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## PCA Visualization")
    return


@app.cell
def __(mo):
    # PCA controls
    n_components_slider = mo.ui.slider(
        start=2,
        stop=10,
        step=1,
        value=3,
        label='Number of PCA components:',
        show_value=True
    )

    plot_3d_checkbox = mo.ui.checkbox(
        value=True,
        label='3D plot'
    )

    num_trials_pca_slider = mo.ui.slider(
        start=10,
        stop=80,
        step=10,
        value=40,
        label='Number of trials to visualize:',
        show_value=True
    )

    color_by_dropdown = mo.ui.dropdown(
        options=['rule', 'trial'],
        value='rule',
        label='Color by:'
    )

    mo.hstack([
        mo.vstack([n_components_slider, num_trials_pca_slider]),
        mo.vstack([plot_3d_checkbox, color_by_dropdown])
    ])
    return (
        color_by_dropdown,
        n_components_slider,
        num_trials_pca_slider,
        plot_3d_checkbox,
    )


@app.cell
def __(
    color_by_dropdown,
    data_dict,
    do_pca,
    model,
    mo,
    n_components_slider,
    num_trials_pca_slider,
    plot_3d_checkbox,
    plt,
    task,
    visualize_pca,
):
    if data_dict is not None and task is not None:
        # Run PCA
        result_full = do_pca(
            data_dict,
            task,
            model=model,
            n_components=n_components_slider.value
        )

        # Visualize
        pca_plot_fig = visualize_pca(
            result_full,
            plot_3d=plot_3d_checkbox.value,
            num_trials=num_trials_pca_slider.value,
            color_by=color_by_dropdown.value,
        )

        pca_info = mo.md(f"""
        **PCA Results:**
        - Data shape: {result_full['pca_data'].shape}
        - Axis labels: {result_full['axis_labels']}
        """)

        pca_plot = mo.ui.pyplot(pca_plot_fig)
        plt.close(pca_plot_fig)
    else:
        result_full = None
        pca_plot = mo.md("*Load data first*")
        pca_info = mo.md("")

    mo.vstack([pca_info, pca_plot])
    return pca_info, pca_plot, pca_plot_fig, result_full


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## Cross-Period Variance

    How well do PCs from one period explain variance in other periods?
    """)
    return


@app.cell
def __(mo):
    # Cross-period variance controls
    n_components_cross_slider = mo.ui.slider(
        start=2,
        stop=10,
        step=1,
        value=3,
        label='Number of components for cross-period analysis:',
        show_value=True
    )

    n_components_cross_slider
    return (n_components_cross_slider,)


@app.cell
def __(
    data_dict,
    mo,
    n_components_cross_slider,
    plot_cross_period_variance,
    plt,
    task,
):
    if data_dict is not None and task is not None:
        cross_period_fig, cross_period_ax = plot_cross_period_variance(
            data_dict,
            task=task,
            n_components=n_components_cross_slider.value,
        )

        cross_period_plot = mo.ui.pyplot(cross_period_fig)
        plt.close(cross_period_fig)
    else:
        cross_period_plot = mo.md("*Load data first*")

    cross_period_plot
    return cross_period_ax, cross_period_fig, cross_period_plot


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## Psychometric Function

    P(Anti saccade) vs interval duration for each rule.
    """)
    return


@app.cell
def __(mo):
    # Psychometric function controls
    num_trials_per_interval_slider = mo.ui.slider(
        start=50,
        stop=500,
        step=50,
        value=100,
        label='Trials per interval:',
        show_value=True
    )

    num_trials_per_interval_slider
    return (num_trials_per_interval_slider,)


@app.cell
def __(
    compute_psychometric_curves,
    model,
    mo,
    num_trials_per_interval_slider,
    plt,
    task,
):
    if model is not None and task is not None:
        # Compute psychometric curves
        psycho_results = compute_psychometric_curves(
            task,
            model,
            num_trials_per_interval=num_trials_per_interval_slider.value,
            rules=[1, -1]
        )

        # Plot
        psycho_fig, psycho_ax = plt.subplots(1, 1, figsize=(6, 4))
        for rule in [1, -1]:
            rule_label = 'C1' if rule == 1 else 'C2'
            anti_probs = psycho_results['anti_probs'][rule]
            psycho_ax.plot(
                psycho_results['intervals'],
                anti_probs,
                marker='o',
                label=f'{rule_label} rule',
                linewidth=2
            )

        psycho_ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        psycho_ax.axvline(
            task.decision_threshold,
            color='gray',
            linestyle='--',
            alpha=0.5,
            label=f'Threshold ({task.decision_threshold}ms)'
        )
        psycho_ax.set_xlabel('Interval (ms)')
        psycho_ax.set_ylabel('P(Anti saccade)')
        psycho_ax.set_title('Psychometric Curves')
        psycho_ax.legend()
        psycho_ax.grid(True, alpha=0.3)
        psycho_fig.tight_layout()

        psycho_plot = mo.ui.pyplot(psycho_fig)
        plt.close(psycho_fig)
    else:
        psycho_results = None
        psycho_plot = mo.md("*Load data first*")

    psycho_plot
    return psycho_ax, psycho_fig, psycho_plot, psycho_results


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## Variance Explained

    How many principal components are needed to capture neural variance?
    """)
    return


@app.cell
def __(mo):
    # Variance explained controls
    n_components_var_slider = mo.ui.slider(
        start=5,
        stop=50,
        step=5,
        value=20,
        label='Number of components to analyze:',
        show_value=True
    )

    n_components_var_slider
    return (n_components_var_slider,)


@app.cell
def __(
    data_dict,
    do_pca,
    mo,
    n_components_var_slider,
    np,
    plt,
    task,
):
    if data_dict is not None and task is not None:
        # Perform PCA with many components
        result_dims = do_pca(
            data_dict,
            task,
            periods='all',
            n_components=n_components_var_slider.value
        )

        # Extract variance explained
        var_explained = result_dims['explained_variance']
        var_explained_pct = 100 * var_explained / var_explained.sum()
        cumulative_var = np.cumsum(var_explained_pct)

        # Plot
        var_fig, var_axes = plt.subplots(1, 2, figsize=(12, 4))

        # Bar plot of variance explained
        var_ax = var_axes[0]
        x = np.arange(1, len(var_explained_pct) + 1)
        var_ax.bar(x, var_explained_pct, alpha=0.7)
        var_ax.set_xlabel('Principal Component')
        var_ax.set_ylabel('Variance Explained (%)')
        var_ax.set_title('Variance Explained by Each PC')
        var_ax.set_xticks(x[::2])  # Show every other tick
        var_ax.grid(True, alpha=0.3, axis='y')

        # Cumulative variance plot
        var_ax = var_axes[1]
        var_ax.plot(x, cumulative_var, marker='o', linewidth=2, markersize=4)
        var_ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% variance')
        var_ax.axhline(95, color='orange', linestyle='--', alpha=0.5, label='95% variance')
        var_ax.set_xlabel('Number of Components')
        var_ax.set_ylabel('Cumulative Variance Explained (%)')
        var_ax.set_title('Cumulative Variance Explained')
        var_ax.legend()
        var_ax.grid(True, alpha=0.3)
        var_ax.set_ylim([0, 105])

        var_fig.tight_layout()

        # Summary statistics
        n_90 = np.argmax(cumulative_var >= 90) + 1 if np.any(cumulative_var >= 90) else len(cumulative_var)
        n_95 = np.argmax(cumulative_var >= 95) + 1 if np.any(cumulative_var >= 95) else len(cumulative_var)

        var_summary = mo.md(f"""
        **Dimensionality Summary:**
        - Components for 90% variance: {n_90}
        - Components for 95% variance: {n_95}
        - Top 5 PCs explain: {cumulative_var[4]:.1f}% of variance
        - Top 10 PCs explain: {cumulative_var[9]:.1f}% of variance
        """)

        var_plot = mo.ui.pyplot(var_fig)
        plt.close(var_fig)
    else:
        result_dims = None
        var_summary = mo.md("*Load data first*")
        var_plot = mo.md("")

    mo.vstack([var_summary, var_plot])
    return (
        cumulative_var,
        n_90,
        n_95,
        result_dims,
        var_ax,
        var_axes,
        var_explained,
        var_explained_pct,
        var_fig,
        var_plot,
        var_summary,
        x,
    )


if __name__ == "__main__":
    app.run()
