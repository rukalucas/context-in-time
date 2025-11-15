"""Analysis package for context-in-time RNN experiments.

Provides tools for data generation, PCA analysis, and visualization of neural trajectories.
"""

# PCA analysis and visualization
from analysis.pca import (
    do_pca,
    visualize_pca,
    animate_pca,
    plot_cross_period_variance,
    parse_projection,
    ProjectionVector,
)

# Data generation and utilities
from analysis.utils import (
    generate_data,
    get_metadata,
    compute_psychometric_curves,
    compute_trial_events,
)

__all__ = [
    # PCA
    "do_pca",
    "visualize_pca",
    "animate_pca",
    "plot_cross_period_variance",
    "parse_projection",
    "ProjectionVector",
    # Utils
    "generate_data",
    "get_metadata",
    "compute_psychometric_curves",
    "compute_trial_events",
]
