"""
Visualization Utilities for Healthcare Intelligence System
=============================================================
Provides plotting functions for confusion matrices, ROC/PR curves,
feature importances, risk distributions, modality comparisons,
calibration curves, and Grad-CAM overlays.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for pipeline use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# --------------------------------------------------------------------------- #
#  1. Confusion Matrix
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix",
                          normalize=False, cmap="Blues", figsize=(8, 6)):
    """Plot a confusion matrix as a matplotlib heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    classes : list[str]
        Class names for axis labels.
    title : str
        Plot title.
    normalize : bool
        If True, show row-normalized percentages.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  2. ROC Curves
# --------------------------------------------------------------------------- #

def plot_roc_curves(y_true, y_prob, classes, figsize=(10, 8)):
    """Plot per-class, micro-averaged, and macro-averaged ROC curves.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Integer-encoded true labels.
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities.
    classes : list[str]
        Class names.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}

    # Per-class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr["micro"], tpr["micro"],
            label=f"Micro-avg ROC (AUC = {roc_auc['micro']:.3f})",
            linestyle=":", linewidth=3)
    ax.plot(fpr["macro"], tpr["macro"],
            label=f"Macro-avg ROC (AUC = {roc_auc['macro']:.3f})",
            linestyle=":", linewidth=3)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color,
                label=f"{classes[i]} (AUC = {roc_auc[i]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  3. Precision-Recall Curves
# --------------------------------------------------------------------------- #

def plot_precision_recall_curves(y_true, y_prob, classes, figsize=(10, 8)):
    """Plot per-class precision-recall curves.

    Parameters
    ----------
    y_true : array-like  (n_samples,)
    y_prob : array-like  (n_samples, n_classes)
    classes : list[str]

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, color=color,
                label=f"{classes[i]} (AP = {ap:.3f})")

    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
           xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  4. Feature Importance
# --------------------------------------------------------------------------- #

def plot_feature_importance(feature_names, importances, top_n=20, figsize=(10, 8)):
    """Horizontal bar chart of feature importances.

    Parameters
    ----------
    feature_names : list[str]
    importances : array-like
    top_n : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    importances = np.asarray(importances)
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  5. Risk Distribution
# --------------------------------------------------------------------------- #

def plot_risk_distribution(risk_scores, risk_levels, figsize=(10, 6)):
    """Distribution of predicted risk scores coloured by risk level.

    Parameters
    ----------
    risk_scores : array-like  (continuous risk value)
    risk_levels : array-like  (categorical: Low / Medium / High / Critical)

    Returns
    -------
    matplotlib.figure.Figure  or  plotly.graph_objects.Figure (if plotly available)
    """
    risk_scores = np.asarray(risk_scores)
    risk_levels = np.asarray(risk_levels)

    if HAS_PLOTLY:
        import pandas as pd
        df = pd.DataFrame({"Risk Score": risk_scores, "Risk Level": risk_levels})
        fig = px.histogram(
            df, x="Risk Score", color="Risk Level",
            nbins=50, barmode="overlay", opacity=0.7,
            title="Risk Score Distribution by Level",
            color_discrete_map={
                "Low": "green", "Medium": "gold",
                "High": "orange", "Critical": "red",
            },
        )
        return fig

    # Matplotlib fallback
    level_colors = {"Low": "green", "Medium": "gold", "High": "orange", "Critical": "red"}
    fig, ax = plt.subplots(figsize=figsize)
    for level in np.unique(risk_levels):
        mask = risk_levels == level
        ax.hist(risk_scores[mask], bins=30, alpha=0.6,
                label=level, color=level_colors.get(level, "grey"))
    ax.set(xlabel="Risk Score", ylabel="Count",
           title="Risk Score Distribution by Level")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  6. Modality Comparison
# --------------------------------------------------------------------------- #

def plot_modality_comparison(comparison_df, figsize=(12, 6)):
    """Grouped bar chart comparing metrics across modalities.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Rows = modalities, columns = metric names.

    Returns
    -------
    matplotlib.figure.Figure  or  plotly.graph_objects.Figure
    """
    if HAS_PLOTLY:
        fig = go.Figure()
        for col in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=comparison_df.index.tolist(),
                y=comparison_df[col].tolist(),
            ))
        fig.update_layout(
            barmode="group",
            title="Modality Performance Comparison",
            xaxis_title="Modality",
            yaxis_title="Score",
        )
        return fig

    # Matplotlib fallback
    x = np.arange(len(comparison_df.index))
    width = 0.8 / len(comparison_df.columns)
    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(comparison_df.columns):
        ax.bar(x + i * width, comparison_df[col], width, label=col)
    ax.set_xticks(x + width * (len(comparison_df.columns) - 1) / 2)
    ax.set_xticklabels(comparison_df.index, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Modality Performance Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  7. Calibration Curve
# --------------------------------------------------------------------------- #

def plot_calibration_curve(y_true, y_prob, n_bins=10, figsize=(8, 8)):
    """Reliability / calibration diagram.

    Parameters
    ----------
    y_true : array-like  (binary 0/1 or integer labels)
    y_prob : array-like  (probabilities for the positive class or max-prob)
    n_bins : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # For multiclass, flatten to binary: correct vs. incorrect with max prob
    if y_prob.ndim == 2:
        pred_class = np.argmax(y_prob, axis=1)
        y_prob_max = np.max(y_prob, axis=1)
        y_binary = (pred_class == y_true).astype(int)
    else:
        y_binary = y_true
        y_prob_max = y_prob

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_binary, y_prob_max, n_bins=n_bins, strategy="uniform"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.set(ylabel="Fraction of positives", title="Calibration Curve")
    ax1.legend()

    ax2.hist(y_prob_max, range=(0, 1), bins=n_bins, color="steelblue",
             edgecolor="white")
    ax2.set(xlabel="Mean predicted probability", ylabel="Count")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  8. Grad-CAM Overlay
# --------------------------------------------------------------------------- #

def plot_gradcam_overlay(image, heatmap, alpha=0.4, figsize=(8, 8)):
    """Overlay a Grad-CAM heatmap on a medical image.

    Parameters
    ----------
    image : np.ndarray
        Original image (H, W, 3) or (H, W) grayscale, values in [0, 255]
        or [0, 1].
    heatmap : np.ndarray
        Grad-CAM activation map (H', W'), values in [0, 1].
    alpha : float
        Blending weight for the heatmap overlay.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import cv2

    image = np.asarray(image, dtype=np.float32)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    # Normalize image to 0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Ensure 3-channel
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Resize heatmap to image size and apply colour map
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colour = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colour, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle("Grad-CAM Visualization", fontsize=14)
    fig.tight_layout()
    return fig
