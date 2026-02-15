"""
castle.visualization
Visualization functions separated from data classes (B-01).
"""

from castle.visualization.embedding_plots import (  # noqa: F401
    plot_embedding,
    plot_named_embedding,
    plot_syllables,
    plot_focus_embedding,
    plot_syllables_bar,
)

__all__ = [
    "plot_embedding",
    "plot_named_embedding",
    "plot_syllables",
    "plot_focus_embedding",
    "plot_syllables_bar",
]
