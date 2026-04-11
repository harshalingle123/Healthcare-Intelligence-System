"""
Healthcare-Specific Evaluation Metrics
========================================
Provides clinical classification metrics (sensitivity, specificity, PPV, NPV,
Youden Index, diagnostic odds ratio) alongside standard sklearn metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
)


# --------------------------------------------------------------------------- #
#  Per-class helpers
# --------------------------------------------------------------------------- #

def _binary_confusion(y_true, y_pred, pos_label):
    """Return (TP, FP, FN, TN) treating *pos_label* as the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    return tp, fp, fn, tn


# --------------------------------------------------------------------------- #
#  Public metric functions
# --------------------------------------------------------------------------- #

def sensitivity_score(y_true, y_pred):
    """True-positive rate (recall) per class.

    Returns
    -------
    dict[label, float]
        Sensitivity for each unique label.
    """
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        tp, fp, fn, tn = _binary_confusion(y_true, y_pred, c)
        result[int(c) if np.issubdtype(type(c), np.integer) else c] = (
            tp / (tp + fn) if (tp + fn) > 0 else 0.0
        )
    return result


def specificity_score(y_true, y_pred):
    """True-negative rate per class.

    Returns
    -------
    dict[label, float]
        Specificity for each unique label.
    """
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        tp, fp, fn, tn = _binary_confusion(y_true, y_pred, c)
        result[int(c) if np.issubdtype(type(c), np.integer) else c] = (
            tn / (tn + fp) if (tn + fp) > 0 else 0.0
        )
    return result


def ppv_score(y_true, y_pred):
    """Positive predictive value (precision) per class.

    Returns
    -------
    dict[label, float]
    """
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        tp, fp, fn, tn = _binary_confusion(y_true, y_pred, c)
        result[int(c) if np.issubdtype(type(c), np.integer) else c] = (
            tp / (tp + fp) if (tp + fp) > 0 else 0.0
        )
    return result


def npv_score(y_true, y_pred):
    """Negative predictive value per class.

    Returns
    -------
    dict[label, float]
    """
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        tp, fp, fn, tn = _binary_confusion(y_true, y_pred, c)
        result[int(c) if np.issubdtype(type(c), np.integer) else c] = (
            tn / (tn + fn) if (tn + fn) > 0 else 0.0
        )
    return result


def youden_index(y_true, y_pred):
    """Youden's J statistic per class: Sensitivity + Specificity - 1.

    Returns
    -------
    dict[label, float]
    """
    sens = sensitivity_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    return {c: sens[c] + spec[c] - 1.0 for c in sens}


def diagnostic_odds_ratio(y_true, y_pred):
    """Diagnostic odds ratio per class: (TP*TN) / (FP*FN).

    Returns
    -------
    dict[label, float]
        DOR for each class.  Returns ``float('inf')`` when denominator is zero.
    """
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        tp, fp, fn, tn = _binary_confusion(y_true, y_pred, c)
        denom = fp * fn
        key = int(c) if np.issubdtype(type(c), np.integer) else c
        result[key] = (tp * tn) / denom if denom > 0 else float("inf")
    return result


# --------------------------------------------------------------------------- #
#  Aggregate convenience function
# --------------------------------------------------------------------------- #

def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Compute a comprehensive dictionary of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities (n_samples, n_classes).  Required for AUC-ROC
        and log-loss.

    Returns
    -------
    dict
        Dictionary with keys: accuracy, precision_macro, recall_macro,
        f1_macro, mcc, cohen_kappa, sensitivity, specificity, ppv, npv,
        youden_index, diagnostic_odds_ratio, and optionally auc_roc_macro
        and log_loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    avg = "macro"  # default averaging for multiclass

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "sensitivity": sensitivity_score(y_true, y_pred),
        "specificity": specificity_score(y_true, y_pred),
        "ppv": ppv_score(y_true, y_pred),
        "npv": npv_score(y_true, y_pred),
        "youden_index": youden_index(y_true, y_pred),
        "diagnostic_odds_ratio": diagnostic_odds_ratio(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["auc_roc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["auc_roc_macro"] = None
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except ValueError:
            metrics["log_loss"] = None

    return metrics
