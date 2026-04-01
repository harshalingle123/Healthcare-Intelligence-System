"""
Clinical Ensemble Model for multi-modal healthcare prediction fusion.
Combines structured data, NLP, lab anomaly, and imaging model predictions.
Part of the Healthcare Intelligence System models layer.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score


class ClinicalEnsemble:
    """Multi-modal ensemble combining structured, NLP, lab, and imaging predictions."""

    RISK_LEVELS = {
        "Critical": (0.75, 1.0),
        "High": (0.50, 0.75),
        "Moderate": (0.25, 0.50),
        "Low": (0.0, 0.25),
    }

    def __init__(self, weights=None):
        self.weights = weights or {
            "structured": 0.30,
            "nlp": 0.25,
            "lab": 0.25,
            "imaging": 0.20,
        }
        self.meta_learner = None
        self.calibrated_model = None
        self.modality_columns = ["structured_prob", "nlp_prob", "lab_prob", "imaging_prob"]

    def collect_modality_predictions(self, structured_preds, nlp_preds, lab_preds, image_preds):
        """
        Merge per-modality predictions by patient_id into a single pandas DataFrame.

        Each input should be a dict/DataFrame with at least 'patient_id' and a probability column.
        """
        def _to_df(preds, prob_col):
            if isinstance(preds, pd.DataFrame):
                df = preds.copy()
            elif isinstance(preds, dict):
                df = pd.DataFrame(preds)
            elif hasattr(preds, "toPandas"):
                df = preds.toPandas()
            else:
                df = pd.DataFrame(preds)
            return df

        structured_df = _to_df(structured_preds, "structured_prob")
        nlp_df = _to_df(nlp_preds, "nlp_prob")
        lab_df = _to_df(lab_preds, "lab_prob")
        image_df = _to_df(image_preds, "imaging_prob")

        # Rename probability columns for clarity
        rename_map = {
            "structured": "structured_prob",
            "nlp": "nlp_prob",
            "lab": "lab_prob",
            "imaging": "imaging_prob",
            "prob": None,  # generic
            "probability": None,
            "risk_score": None,
        }

        def _ensure_prob_col(df, target_col):
            if target_col in df.columns:
                return df
            for src in ["prob", "probability", "risk_score", "prediction"]:
                if src in df.columns:
                    df = df.rename(columns={src: target_col})
                    return df
            # If no matching column, use the last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_id_cols = [c for c in numeric_cols if c != "patient_id"]
            if non_id_cols:
                df = df.rename(columns={non_id_cols[-1]: target_col})
            return df

        structured_df = _ensure_prob_col(structured_df, "structured_prob")
        nlp_df = _ensure_prob_col(nlp_df, "nlp_prob")
        lab_df = _ensure_prob_col(lab_df, "lab_prob")
        image_df = _ensure_prob_col(image_df, "imaging_prob")

        # Merge on patient_id
        merged = structured_df[["patient_id", "structured_prob"]].copy()
        for df, col in [
            (nlp_df, "nlp_prob"),
            (lab_df, "lab_prob"),
            (image_df, "imaging_prob"),
        ]:
            if "patient_id" in df.columns and col in df.columns:
                merged = merged.merge(
                    df[["patient_id", col]],
                    on="patient_id",
                    how="outer"
                )

        # Fill missing modality predictions with 0.5 (neutral)
        for col in self.modality_columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.5)
            else:
                merged[col] = 0.5

        return merged

    def weighted_fusion(self, predictions_df):
        """Simple weighted average of per-modality risk probabilities."""
        result = predictions_df.copy()

        weighted_sum = np.zeros(len(result))
        for modality, weight in self.weights.items():
            col = f"{modality}_prob"
            if col in result.columns:
                weighted_sum += weight * result[col].values

        result["fused_probability"] = weighted_sum
        result["risk_category"] = result["fused_probability"].apply(self.assign_risk_level)

        return result

    def train_meta_learner(self, predictions_df, labels):
        """
        Train a Logistic Regression meta-learner on stacked base model predictions.
        Uses CalibratedClassifierCV for Platt scaling.
        """
        X = predictions_df[self.modality_columns].values
        y = np.asarray(labels)

        base_lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            multi_class="auto"
        )

        # CalibratedClassifierCV wraps the LR with Platt scaling
        self.calibrated_model = CalibratedClassifierCV(
            estimator=base_lr,
            cv=5,
            method="sigmoid"
        )
        self.calibrated_model.fit(X, y)
        self.meta_learner = self.calibrated_model

        # Cross-validation score
        scores = cross_val_score(base_lr, X, y, cv=5, scoring="accuracy")
        print(f"Meta-learner CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return self.meta_learner

    def predict_final(self, predictions_df):
        """
        Return final risk_category (Critical/High/Moderate/Low), confidence,
        and per-modality contributions.
        """
        result = predictions_df.copy()
        X = result[self.modality_columns].values

        if self.meta_learner is not None:
            probabilities = self.meta_learner.predict_proba(X)
            # Use max probability as the risk probability
            risk_prob = probabilities.max(axis=1)
            predicted_class = self.meta_learner.predict(X)
            result["predicted_class"] = predicted_class
            result["confidence"] = risk_prob
            result["fused_probability"] = risk_prob
        else:
            # Fall back to weighted fusion
            result = self.weighted_fusion(result)
            result["confidence"] = result["fused_probability"]

        result["risk_category"] = result["fused_probability"].apply(self.assign_risk_level)

        # Per-modality contributions
        for modality, weight in self.weights.items():
            col = f"{modality}_prob"
            if col in result.columns:
                result[f"{modality}_contribution"] = (
                    result[col] * weight / result["fused_probability"].clip(lower=1e-8)
                )

        return result

    def explain_prediction(self, patient_data):
        """
        Use SHAP on meta-learner to get feature contributions.
        Falls back to weight-based explanation if SHAP is unavailable.
        """
        try:
            import shap

            if self.meta_learner is None:
                raise RuntimeError("Meta-learner has not been trained.")

            X = np.array(
                [[patient_data.get(col, 0.5) for col in self.modality_columns]]
            )

            explainer = shap.LinearExplainer(
                self.meta_learner.calibrated_classifiers_[0].estimator,
                X
            )
            shap_values = explainer.shap_values(X)

            explanation = {}
            if isinstance(shap_values, list):
                # Multi-class: use the class with highest prediction
                pred_class = self.meta_learner.predict(X)[0]
                sv = shap_values[pred_class][0]
            else:
                sv = shap_values[0]

            for i, col in enumerate(self.modality_columns):
                modality = col.replace("_prob", "")
                explanation[modality] = {
                    "shap_value": float(sv[i]),
                    "feature_value": float(X[0][i]),
                    "contribution": "positive" if sv[i] > 0 else "negative"
                }

            return explanation

        except ImportError:
            print("SHAP not available. Using weight-based explanation.")
            explanation = {}
            for col in self.modality_columns:
                modality = col.replace("_prob", "")
                value = patient_data.get(col, 0.5)
                weight = self.weights.get(modality, 0.0)
                explanation[modality] = {
                    "weight": weight,
                    "feature_value": value,
                    "weighted_contribution": weight * value,
                }
            return explanation

        except Exception as e:
            print(f"SHAP explanation failed: {e}. Using weight-based explanation.")
            explanation = {}
            for col in self.modality_columns:
                modality = col.replace("_prob", "")
                value = patient_data.get(col, 0.5)
                weight = self.weights.get(modality, 0.0)
                explanation[modality] = {
                    "weight": weight,
                    "feature_value": value,
                    "weighted_contribution": weight * value,
                }
            return explanation

    def assign_risk_level(self, probability):
        """Map probability to risk level: >0.75=Critical, >0.5=High, >0.25=Moderate, <=0.25=Low."""
        if probability > 0.75:
            return "Critical"
        elif probability > 0.50:
            return "High"
        elif probability > 0.25:
            return "Moderate"
        else:
            return "Low"

    def generate_recommendations(self, risk_level, top_factors):
        """
        Return list of clinical action recommendations based on risk level
        and contributing factors.

        Args:
            risk_level: One of 'Critical', 'High', 'Moderate', 'Low'.
            top_factors: List of (factor_name, contribution_score) tuples.
        """
        recommendations = []

        # Base recommendations by risk level
        base_recommendations = {
            "Critical": [
                "URGENT: Immediate physician review required.",
                "Activate rapid response team for clinical assessment.",
                "Schedule emergency diagnostic workup within 1 hour.",
                "Notify attending physician and on-call specialist.",
                "Prepare for potential ICU admission.",
            ],
            "High": [
                "Priority physician review within 4 hours.",
                "Order comprehensive diagnostic panel.",
                "Increase monitoring frequency to every 2 hours.",
                "Schedule specialist consultation within 24 hours.",
                "Review and update current treatment plan.",
            ],
            "Moderate": [
                "Schedule physician review within 24 hours.",
                "Order follow-up lab tests for abnormal values.",
                "Continue standard monitoring protocol.",
                "Consider preventive interventions.",
                "Schedule follow-up appointment within 1 week.",
            ],
            "Low": [
                "Continue routine monitoring schedule.",
                "Maintain current care plan.",
                "Schedule standard follow-up as per protocol.",
                "Educate patient on warning signs to watch for.",
            ],
        }

        recommendations.extend(
            base_recommendations.get(risk_level, base_recommendations["Low"])
        )

        # Factor-specific recommendations
        factor_recommendations = {
            "structured": "Review structured clinical data and vital signs trend.",
            "nlp": "Review clinical notes for context on flagged conditions.",
            "lab": "Prioritize review of abnormal lab values and repeat critical tests.",
            "imaging": "Review imaging findings and consider additional imaging if indicated.",
        }

        if top_factors:
            recommendations.append("--- Key Contributing Factors ---")
            for factor_name, score in top_factors:
                clean_name = factor_name.replace("_prob", "").replace("_contribution", "")
                if clean_name in factor_recommendations:
                    recommendations.append(
                        f"  [{clean_name.upper()} (score: {score:.2f})]: "
                        f"{factor_recommendations[clean_name]}"
                    )

        return recommendations

    def save_model(self, path):
        """Save meta-learner and weights with joblib."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "meta_learner": self.meta_learner,
            "calibrated_model": self.calibrated_model,
            "weights": self.weights,
        }
        joblib.dump(data, path)

    def load_model(self, path):
        """Load meta-learner and weights from joblib."""
        data = joblib.load(path)
        self.meta_learner = data.get("meta_learner")
        self.calibrated_model = data.get("calibrated_model")
        self.weights = data.get("weights", self.weights)
        return self.meta_learner
