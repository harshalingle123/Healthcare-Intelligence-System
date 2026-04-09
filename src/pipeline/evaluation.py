"""
Pipeline Evaluation Module
============================
Provides per-modality evaluation, ensemble evaluation, modality comparison,
healthcare-specific metrics, and report generation with plots.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.utils.metrics import (
    sensitivity_score,
    specificity_score,
    ppv_score,
    npv_score,
    youden_index,
    compute_all_metrics,
)
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_modality_comparison,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PipelineEvaluator:
    """Evaluate individual modalities, the ensemble, and generate reports."""

    # ------------------------------------------------------------------ #
    #  Per-modality metrics
    # ------------------------------------------------------------------ #
    def evaluate_per_modality(self, results_dict):
        """Compute standard + healthcare metrics for each modality.

        Parameters
        ----------
        results_dict : dict
            Mapping of modality name -> dict with keys:
                ``y_true``, ``y_pred``, ``y_prob`` (optional).

        Returns
        -------
        dict[str, dict]
            Modality name -> metrics dictionary.
        """
        modality_metrics = {}
        for modality, res in results_dict.items():
            y_true = np.asarray(res["y_true"])
            y_pred = np.asarray(res["y_pred"])
            y_prob = np.asarray(res["y_prob"]) if res.get("y_prob") is not None else None

            metrics = compute_all_metrics(y_true, y_pred, y_prob)
            modality_metrics[modality] = metrics
            logger.info(
                "Modality %-15s | Accuracy %.4f | F1-macro %.4f",
                modality, metrics["accuracy"], metrics["f1_macro"],
            )
        return modality_metrics

    # ------------------------------------------------------------------ #
    #  Ensemble metrics
    # ------------------------------------------------------------------ #
    def evaluate_ensemble(self, ensemble_preds, true_labels, ensemble_probs=None):
        """Full evaluation of ensemble predictions.

        Parameters
        ----------
        ensemble_preds : array-like
        true_labels : array-like
        ensemble_probs : array-like, optional

        Returns
        -------
        dict
        """
        metrics = compute_all_metrics(true_labels, ensemble_preds, ensemble_probs)
        logger.info(
            "Ensemble           | Accuracy %.4f | F1-macro %.4f",
            metrics["accuracy"], metrics["f1_macro"],
        )
        return metrics

    # ------------------------------------------------------------------ #
    #  Modality comparison table
    # ------------------------------------------------------------------ #
    def compare_modalities(self, results_dict):
        """Build a summary DataFrame comparing all modalities + ensemble.

        Parameters
        ----------
        results_dict : dict
            Same format as ``evaluate_per_modality``.

        Returns
        -------
        pandas.DataFrame
            Rows = modalities, columns = key metrics.
        """
        rows = []
        for modality, res in results_dict.items():
            y_true = np.asarray(res["y_true"])
            y_pred = np.asarray(res["y_pred"])
            y_prob = np.asarray(res["y_prob"]) if res.get("y_prob") is not None else None

            row = {
                "Modality": modality,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
            if y_prob is not None:
                try:
                    row["AUC-ROC (macro)"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro"
                    )
                except ValueError:
                    row["AUC-ROC (macro)"] = np.nan
            else:
                row["AUC-ROC (macro)"] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Modality")
        logger.info("\n%s", df.to_string())
        return df

    # ------------------------------------------------------------------ #
    #  Full evaluation report with plots
    # ------------------------------------------------------------------ #
    def generate_evaluation_report(self, results_dict, output_path):
        """Generate confusion matrices, ROC curves, PR curves, and save.

        Parameters
        ----------
        results_dict : dict
            Modality -> {y_true, y_pred, y_prob, classes}.
        output_path : str
            Directory to write PNG files and summary JSON.
        """
        os.makedirs(output_path, exist_ok=True)
        logger.info("Generating evaluation report in %s", output_path)

        # Evaluate all modalities
        all_metrics = self.evaluate_per_modality(results_dict)

        # Comparison table
        comparison_df = self.compare_modalities(results_dict)
        comparison_df.to_csv(os.path.join(output_path, "modality_comparison.csv"))

        # Per-modality plots
        for modality, res in results_dict.items():
            y_true = np.asarray(res["y_true"])
            y_pred = np.asarray(res["y_pred"])
            y_prob = np.asarray(res["y_prob"]) if res.get("y_prob") is not None else None
            classes = res.get("classes", [str(c) for c in sorted(np.unique(y_true))])

            # Confusion matrix
            fig = plot_confusion_matrix(y_true, y_pred, classes,
                                        title=f"{modality} - Confusion Matrix")
            fig.savefig(os.path.join(output_path, f"{modality}_confusion_matrix.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            if y_prob is not None:
                # ROC curves
                try:
                    fig = plot_roc_curves(y_true, y_prob, classes)
                    fig.savefig(os.path.join(output_path, f"{modality}_roc_curves.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    logger.warning("Could not plot ROC for %s: %s", modality, e)

                # PR curves
                try:
                    fig = plot_precision_recall_curves(y_true, y_prob, classes)
                    fig.savefig(os.path.join(output_path, f"{modality}_pr_curves.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    logger.warning("Could not plot PR for %s: %s", modality, e)

        # Modality comparison chart
        try:
            fig = plot_modality_comparison(comparison_df)
            if hasattr(fig, "savefig"):
                fig.savefig(os.path.join(output_path, "modality_comparison.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                # plotly figure
                fig.write_image(os.path.join(output_path, "modality_comparison.png"))
        except Exception as e:
            logger.warning("Could not plot modality comparison: %s", e)

        # Save metrics JSON (convert numpy types for serialisation)
        serialisable = {}
        for mod, m in all_metrics.items():
            serialisable[mod] = {
                k: (v if isinstance(v, (int, float, str, type(None), dict, list)) else str(v))
                for k, v in m.items()
            }
        with open(os.path.join(output_path, "metrics.json"), "w") as f:
            json.dump(serialisable, f, indent=2, default=str)

        logger.info("Evaluation report saved to %s", output_path)
        return all_metrics

    # ------------------------------------------------------------------ #
    #  Healthcare-specific metrics
    # ------------------------------------------------------------------ #
    def compute_healthcare_metrics(self, y_true, y_pred, y_prob=None):
        """Compute sensitivity, specificity, PPV, NPV, Youden index.

        Parameters
        ----------
        y_true, y_pred : array-like
        y_prob : array-like, optional

        Returns
        -------
        dict
        """
        return {
            "sensitivity": sensitivity_score(y_true, y_pred),
            "specificity": specificity_score(y_true, y_pred),
            "ppv": ppv_score(y_true, y_pred),
            "npv": npv_score(y_true, y_pred),
            "youden_index": youden_index(y_true, y_pred),
        }


# Need matplotlib imported for plt.close in generate_evaluation_report
import matplotlib.pyplot as plt
