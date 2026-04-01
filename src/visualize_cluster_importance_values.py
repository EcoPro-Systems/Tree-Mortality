#!/usr/bin/env python3
"""
Plot feature importance comparison across three model subsets (C1, C2, C1+2).

Usage:
    python plot_feature_importance.py <c1_json> <c2_json> <c1c2_json> <output_pdf>
"""

import json
import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend


@click.command()
@click.argument("c1_json", type=click.Path(exists=True))
@click.argument("c2_json", type=click.Path(exists=True))
@click.argument("c1c2_json", type=click.Path(exists=True))
@click.argument("output_pdf", type=click.Path())
def plot_feature_importance(c1_json, c2_json, c1c2_json, output_pdf):
    """Compare feature importances across C1, C2, and C1+2 model subsets.

    Arguments:\n
        C1_JSON    Path to JSON file for model trained on subset C1.\n
        C2_JSON    Path to JSON file for model trained on subset C2.\n
        C1C2_JSON  Path to JSON file for model trained on combined subset C1+2.\n
        OUTPUT_PDF Path to output PDF file for the plot.
    """
    # Load data
    def load_importances(path):
        with open(path) as f:
            data = json.load(f)
        return data["overall_importances"]

    c1_imp = load_importances(c1_json)
    c2_imp = load_importances(c2_json)
    c1c2_imp = load_importances(c1c2_json)

    # Use the union of all feature names, sorted by C1+2 importance (descending)
    all_features = sorted(
        set(c1_imp) | set(c2_imp) | set(c1c2_imp),
        key=lambda f: c1c2_imp.get(f, 0),
        reverse=True,
    )

    n = len(all_features)
    x = np.arange(n)

    # Bar layout: each group has 3 bars; order is C1, C1+2, C2 (combined in middle)
    bar_width = 0.25
    offsets = {
        "C1":   -bar_width,
        "C1+2":  0.0,
        "C2":    bar_width,
    }

    colors = {
        "C1":   "#F5C518",   # golden yellow
        "C1+2": "#4CAF50",   # green (blend of yellow + blue)
        "C2":   "#2196F3",   # blue
    }

    datasets = {
        "C1":   c1_imp,
        "C2":   c2_imp,
        "C1+2": c1c2_imp,
    }

    fig, ax = plt.subplots(figsize=(max(12, n * 0.9), 6))

    for label, offset in offsets.items():
        vals = [datasets[label].get(f, 0.0) for f in all_features]
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_width,
            label=label,
            color=colors[label],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_features, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Feature Importance", fontsize=11)
    ax.set_title("Feature Importance Comparison Across Clusters", fontsize=13, fontweight="bold")
    ax.legend(title="Dataset", fontsize=10, title_fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    with pdf_backend.PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    click.echo(f"Plot saved to {output_pdf}")


if __name__ == "__main__":
    plot_feature_importance()
