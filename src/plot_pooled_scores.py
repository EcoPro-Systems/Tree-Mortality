#!/usr/bin/env python3
"""
Plot pooled score comparison across three model subsets (C1, C2, C1+2).

Usage:
    python plot_pooled_scores.py <c1_json> <c2_json> <c1c2_json> <output_pdf>
"""

import json
import click
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend


@click.command()
@click.argument("c1_json", type=click.Path(exists=True))
@click.argument("c2_json", type=click.Path(exists=True))
@click.argument("c1c2_json", type=click.Path(exists=True))
@click.argument("output_pdf", type=click.Path())
def plot_pooled_scores(c1_json, c2_json, c1c2_json, output_pdf):
    """Compare pooled scores across C1, C2, and C1+2 model subsets.

    Arguments:\n
        C1_JSON    Path to JSON file for model trained on subset C1.\n
        C2_JSON    Path to JSON file for model trained on subset C2.\n
        C1C2_JSON  Path to JSON file for model trained on combined subset C1+2.\n
        OUTPUT_PDF Path to output PDF file for the plot.
    """
    def load_score(path):
        with open(path) as f:
            data = json.load(f)
        return data["pooled_score"]

    labels = ["C1", "C1+2", "C2"]
    scores = [
        load_score(c1_json),
        load_score(c1c2_json),
        load_score(c2_json),
    ]
    colors = ["#F5C518", "#4CAF50", "#2196F3"]

    fig, ax = plt.subplots(figsize=(5, 5))

    bars = ax.bar(labels, scores, color=colors, edgecolor="white", linewidth=0.8, width=0.5, zorder=3)

    # Annotate each bar with its value
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Pooled Score", fontsize=11)
    ax.set_title("Pooled Score Comparison Across Model Subsets", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Set y-axis to start just below the minimum score for better visual differentiation
    min_score = min(scores)
    max_score = max(scores)
    padding = (max_score - min_score) * 0.4 or 0.05
    ax.set_ylim(max(0, min_score - padding), max_score + padding * 2)

    plt.tight_layout()

    with pdf_backend.PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    click.echo(f"Plot saved to {output_pdf}")


if __name__ == "__main__":
    plot_pooled_scores()
