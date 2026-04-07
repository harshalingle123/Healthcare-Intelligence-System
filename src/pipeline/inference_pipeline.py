"""
Inference Pipeline for Healthcare Intelligence System
========================================================
Loads trained models and runs single-patient or batch predictions,
returning structured results with risk levels, confidence scores,
per-modality scores, top contributing factors, and recommended actions.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import joblib

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.utils.logger import setup_logger
from src.utils.medical_constants import DIAGNOSIS_CATEGORIES

logger = setup_logger(__name__)


class InferencePipeline:
    """End-to-end inference using all trained modality models and ensemble."""

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models_dir = self.config["paths"]["models"]
        self.symptom_classifier = None
        self.nlp_model = None
        self.lab_detector = None
        self.image_classifier = None
        self.ensemble = None
        self._models_loaded = False

        # Diagnosis label mapping
        self.label_map = {
            cat["label_index"]: cat["category"]
            for cat in DIAGNOSIS_CATEGORIES
        }

    # ------------------------------------------------------------------ #
    #  Load all persisted models
    # ------------------------------------------------------------------ #
    def load_models(self):
        """Load all saved models from the configured models directory."""
        logger.info("Loading models from %s", self.models_dir)

        # 1. Symptom classifier (PySpark PipelineModel)
        sc_path = os.path.join(self.models_dir, "symptom_classifier")
        if os.path.exists(sc_path):
            try:
                self.symptom_classifier = PipelineModel.load(sc_path)
                logger.info("Loaded SymptomClassifier")
            except Exception as e:
                logger.warning("Could not load SymptomClassifier: %s", e)

        # 2. Clinical NLP model (PyTorch / joblib)
        nlp_path = os.path.join(self.models_dir, "clinical_nlp_model.pkl")
        if os.path.exists(nlp_path):
            try:
                self.nlp_model = joblib.load(nlp_path)
                logger.info("Loaded ClinicalNLPModel")
            except Exception as e:
                logger.warning("Could not load ClinicalNLPModel: %s", e)

        # 3. Lab anomaly detector (joblib)
        lab_path = os.path.join(self.models_dir, "lab_anomaly_detector.pkl")
        if os.path.exists(lab_path):
            try:
                self.lab_detector = joblib.load(lab_path)
                logger.info("Loaded LabAnomalyDetector")
            except Exception as e:
                logger.warning("Could not load LabAnomalyDetector: %s", e)

        # 4. Medical image classifier (PyTorch state dict / joblib)
        img_path = os.path.join(self.models_dir, "image_classifier.pkl")
        if os.path.exists(img_path):
            try:
                self.image_classifier = joblib.load(img_path)
                logger.info("Loaded MedicalImageClassifier")
            except Exception as e:
                logger.warning("Could not load MedicalImageClassifier: %s", e)

        # 5. Ensemble meta-learner (joblib)
        ens_path = os.path.join(self.models_dir, "ensemble.pkl")
        if os.path.exists(ens_path):
            try:
                self.ensemble = joblib.load(ens_path)
                logger.info("Loaded ClinicalEnsemble")
            except Exception as e:
                logger.warning("Could not load ClinicalEnsemble: %s", e)

        self._models_loaded = True
        logger.info("Model loading complete")

    # ------------------------------------------------------------------ #
    #  Single-patient prediction
    # ------------------------------------------------------------------ #
    def predict_single_patient(self, patient_data: dict) -> dict:
        """Run the full multi-modal pipeline for one patient.

        Parameters
        ----------
        patient_data : dict
            Keys may include structured features, clinical note text,
            lab values, and an image array/path.

        Returns
        -------
        dict
            {
                "patient_id": str,
                "risk_level": str,         # Low / Medium / High / Critical
                "confidence": float,
                "predicted_diagnosis": str,
                "per_modality_scores": dict,
                "top_factors": list[str],
                "recommended_actions": list[str],
            }
        """
        if not self._models_loaded:
            self.load_models()

        per_modality_scores = {}
        modality_probs = []

        # --- Structured prediction -------------------------------------------
        structured_prob = self._predict_structured(patient_data)
        if structured_prob is not None:
            per_modality_scores["structured"] = structured_prob.tolist()
            modality_probs.append(("structured", structured_prob))

        # --- NLP prediction ---------------------------------------------------
        nlp_prob = self._predict_nlp(patient_data)
        if nlp_prob is not None:
            per_modality_scores["nlp"] = nlp_prob.tolist()
            modality_probs.append(("nlp", nlp_prob))

        # --- Lab anomaly prediction ------------------------------------------
        lab_prob = self._predict_lab(patient_data)
        if lab_prob is not None:
            per_modality_scores["lab"] = lab_prob.tolist()
            modality_probs.append(("lab", lab_prob))

        # --- Image prediction ------------------------------------------------
        image_prob = self._predict_image(patient_data)
        if image_prob is not None:
            per_modality_scores["imaging"] = image_prob.tolist()
            modality_probs.append(("imaging", image_prob))

        # --- Ensemble / weighted average fallback ----------------------------
        if self.ensemble is not None and modality_probs:
            meta_features = self._build_meta_features(modality_probs)
            ensemble_prob = self.ensemble.predict_proba(meta_features.reshape(1, -1))[0]
        elif modality_probs:
            weights = self.config.get("models", {}).get("ensemble", {}).get("weights", {})
            ensemble_prob = self._weighted_average(modality_probs, weights)
        else:
            ensemble_prob = np.array([0.2] * 5)

        predicted_class = int(np.argmax(ensemble_prob))
        confidence = float(np.max(ensemble_prob))
        risk_level = self._risk_level(confidence, predicted_class)
        diagnosis = self.label_map.get(predicted_class, f"Class {predicted_class}")

        top_factors = self._extract_top_factors(patient_data, per_modality_scores)
        recommended_actions = self._generate_recommendations(
            risk_level, diagnosis, top_factors
        )

        return {
            "patient_id": patient_data.get("patient_id", "unknown"),
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "predicted_diagnosis": diagnosis,
            "per_modality_scores": per_modality_scores,
            "top_factors": top_factors,
            "recommended_actions": recommended_actions,
        }

    # ------------------------------------------------------------------ #
    #  Batch prediction
    # ------------------------------------------------------------------ #
    def predict_batch(self, spark: SparkSession, patients_df: DataFrame) -> DataFrame:
        """Batch inference over a PySpark DataFrame.

        Parameters
        ----------
        spark : SparkSession
        patients_df : DataFrame
            Must contain at least ``patient_id`` and relevant feature columns.

        Returns
        -------
        DataFrame
            Original columns plus ``predicted_diagnosis``, ``risk_level``,
            ``confidence``.
        """
        if not self._models_loaded:
            self.load_models()

        # Collect to driver for per-patient inference (fine for moderate sizes)
        rows = patients_df.toPandas().to_dict(orient="records")
        results = [self.predict_single_patient(row) for row in rows]

        result_df = pd.DataFrame(results)
        spark_result = spark.createDataFrame(result_df)
        return spark_result

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #
    def _predict_structured(self, data):
        if self.symptom_classifier is None:
            return None
        try:
            # Build a single-row Spark DF is expensive; use cached pipeline weights
            # For single-patient we approximate via the RF model's predict
            feature_keys = [
                "age", "bmi", "heart_rate", "systolic_bp", "respiratory_rate",
                "temperature", "oxygen_saturation", "fever", "cough", "fatigue",
                "shortness_of_breath", "chest_pain", "headache", "nausea",
                "dizziness", "joint_pain", "weight_loss", "night_sweats",
                "abdominal_pain",
            ]
            features = np.array(
                [float(data.get(k, 0)) for k in feature_keys]
            ).reshape(1, -1)
            rf_model = self.symptom_classifier.stages[-1]
            from pyspark.ml.linalg import Vectors
            vec = Vectors.dense(features[0])
            prob = rf_model.predictProbability(vec)
            return np.array(prob.toArray())
        except Exception as e:
            logger.warning("Structured prediction failed: %s", e)
            return None

    def _predict_nlp(self, data):
        if self.nlp_model is None or "note_text" not in data:
            return None
        try:
            text = data["note_text"]
            if hasattr(self.nlp_model, "predict_proba"):
                return np.array(self.nlp_model.predict_proba([text])[0])
            elif hasattr(self.nlp_model, "predict"):
                pred = self.nlp_model.predict([text])[0]
                one_hot = np.zeros(5)
                one_hot[int(pred)] = 1.0
                return one_hot
        except Exception as e:
            logger.warning("NLP prediction failed: %s", e)
        return None

    def _predict_lab(self, data):
        if self.lab_detector is None:
            return None
        try:
            lab_keys = [
                "wbc", "rbc", "hemoglobin", "platelets", "glucose",
                "creatinine", "bun", "sodium", "potassium", "calcium",
            ]
            features = np.array(
                [float(data.get(k, 0)) for k in lab_keys]
            ).reshape(1, -1)
            if hasattr(self.lab_detector, "predict_proba"):
                return np.array(self.lab_detector.predict_proba(features)[0])
            elif hasattr(self.lab_detector, "decision_function"):
                score = self.lab_detector.decision_function(features)[0]
                prob = 1.0 / (1.0 + np.exp(score))  # anomaly probability
                return np.array([1 - prob, prob, 0, 0, 0])
        except Exception as e:
            logger.warning("Lab prediction failed: %s", e)
        return None

    def _predict_image(self, data):
        if self.image_classifier is None:
            return None
        try:
            if "image_array" in data:
                img = np.array(data["image_array"])
                if hasattr(self.image_classifier, "predict_proba"):
                    return np.array(self.image_classifier.predict_proba(img.reshape(1, -1))[0])
        except Exception as e:
            logger.warning("Image prediction failed: %s", e)
        return None

    def _build_meta_features(self, modality_probs):
        """Concatenate all modality probability vectors into a single feature vector."""
        return np.concatenate([p for _, p in modality_probs])

    def _weighted_average(self, modality_probs, weights):
        """Fallback weighted average when ensemble model is not available."""
        total_weight = 0.0
        avg = np.zeros_like(modality_probs[0][1])
        for name, prob in modality_probs:
            w = weights.get(name, 1.0 / len(modality_probs))
            avg += w * prob
            total_weight += w
        if total_weight > 0:
            avg /= total_weight
        return avg

    def _risk_level(self, confidence, predicted_class):
        """Map prediction confidence and class to a clinical risk level."""
        # High-severity classes
        high_severity_classes = {1, 4}  # Cardiovascular, Neurological
        if confidence >= 0.85 and predicted_class in high_severity_classes:
            return "Critical"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        else:
            return "Low"

    def _extract_top_factors(self, data, modality_scores, top_n=5):
        """Identify the most influential features from patient data."""
        factors = []
        # Simple heuristic: flag abnormal vitals / present symptoms
        vital_thresholds = {
            "heart_rate": (60, 100, "Abnormal heart rate"),
            "systolic_bp": (90, 140, "Abnormal blood pressure"),
            "temperature": (36.1, 37.2, "Abnormal temperature"),
            "oxygen_saturation": (95, 100, "Low oxygen saturation"),
            "respiratory_rate": (12, 20, "Abnormal respiratory rate"),
        }
        for vital, (lo, hi, desc) in vital_thresholds.items():
            val = data.get(vital)
            if val is not None:
                try:
                    val = float(val)
                    if val < lo or val > hi:
                        factors.append(f"{desc}: {val}")
                except (ValueError, TypeError):
                    pass

        symptom_flags = [
            "fever", "cough", "shortness_of_breath", "chest_pain",
            "fatigue", "night_sweats", "weight_loss",
        ]
        for s in symptom_flags:
            if data.get(s) and int(data[s]) == 1:
                factors.append(f"Symptom present: {s.replace('_', ' ')}")

        # Which modality contributed most
        if modality_scores:
            dominant = max(modality_scores, key=lambda m: max(modality_scores[m]))
            factors.append(f"Strongest signal from {dominant} modality")

        return factors[:top_n]

    def _generate_recommendations(self, risk_level, diagnosis, factors):
        """Generate clinical action recommendations based on risk."""
        actions = []
        if risk_level == "Critical":
            actions.append("URGENT: Immediate physician review required")
            actions.append("Consider ICU admission or rapid response team")
        elif risk_level == "High":
            actions.append("Schedule priority follow-up within 24 hours")
            actions.append("Order additional confirmatory diagnostics")
        elif risk_level == "Medium":
            actions.append("Schedule routine follow-up within 1 week")
            actions.append("Monitor vital signs closely")
        else:
            actions.append("Continue current management plan")
            actions.append("Routine follow-up as scheduled")

        actions.append(f"Evaluate for: {diagnosis}")

        if any("oxygen" in f.lower() for f in factors):
            actions.append("Assess respiratory function; consider ABG / pulse oximetry")
        if any("chest pain" in f.lower() for f in factors):
            actions.append("Obtain ECG and cardiac biomarkers")

        return actions
