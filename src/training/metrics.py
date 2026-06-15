"""Evaluation metrics and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    y_true, y_pred, class_names: list[str]
) -> dict[str, Any]:
    """Compute headline metrics plus a per-class report."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "report_text": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    normalize: bool = True,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 12),
):
    """Plot (and optionally save) a confusion matrix. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=False,
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
