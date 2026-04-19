"""
Tests for model modules: SymptomClassifier pipeline, LabAnomalyDetector,
and ClinicalEnsemble risk levels and weighted fusion.
"""

import pytest
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  SymptomClassifier Pipeline
# ------------------------------------------------------------------ #
class TestSymptomClassifierPipeline:

    def test_pipeline_creates_correct_stages(self):
        """Verify the pipeline has 4 stages: StringIndexer, VectorAssembler, StandardScaler, RF."""
        try:
            from pyspark.ml import Pipeline
            from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
            from pyspark.ml.classification import RandomForestClassifier
            from src.models.symptom_classifier import SymptomClassifier
        except ImportError:
            pytest.skip("PySpark is not installed")

        clf = SymptomClassifier()
        feature_cols = ["feat1", "feat2", "feat3"]
        pipeline = clf.build_pipeline(feature_cols, label_col="diagnosis")

        stages = pipeline.getStages()
        assert len(stages) == 4
        assert isinstance(stages[0], StringIndexer)
        assert isinstance(stages[1], VectorAssembler)
        assert isinstance(stages[2], StandardScaler)
        assert isinstance(stages[3], RandomForestClassifier)

    def test_pipeline_label_indexer_config(self):
        """Verify the label indexer targets the correct column."""
        try:
            from src.models.symptom_classifier import SymptomClassifier
        except ImportError:
            pytest.skip("PySpark is not installed")

        clf = SymptomClassifier()
        clf.build_pipeline(["f1", "f2"], label_col="my_label")

        indexer = clf.label_indexer
        assert indexer.getInputCol() == "my_label"
        assert indexer.getOutputCol() == "label_index"

    def test_pipeline_rf_hyperparameters(self):
        """Verify RF stage picks up config overrides."""
        try:
            from src.models.symptom_classifier import SymptomClassifier
        except ImportError:
            pytest.skip("PySpark is not installed")

        clf = SymptomClassifier(config={"max_depth": 15, "num_trees": 200})
        pipeline = clf.build_pipeline(["f1"], label_col="label")

        rf = pipeline.getStages()[-1]
        assert rf.getMaxDepth() == 15
        assert rf.getNumTrees() == 200


# ------------------------------------------------------------------ #
#  LabAnomalyDetector (IsolationForest)
# ------------------------------------------------------------------ #
class TestLabAnomalyDetector:

    def test_isolation_forest_on_synthetic_data(self):
        """Train IsolationForest on small synthetic data and verify anomaly detection."""
        try:
            from src.models.lab_anomaly_detector import LabAnomalyDetector
        except ImportError:
            pytest.skip("Required modules not available")

        np.random.seed(42)
        # 90 normal samples + 10 outliers
        normal = np.random.randn(90, 3) * 1.0 + 5.0
        outliers = np.random.randn(10, 3) * 1.0 + 20.0
        X = np.vstack([normal, outliers])

        pdf = pd.DataFrame(X, columns=["lab_a", "lab_b", "lab_c"])

        detector = LabAnomalyDetector(contamination=0.1, n_estimators=100)
        detector.train_isolation_forest(pdf)

        result = detector.detect_anomalies(pdf)
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns

        # At least some outliers should be detected
        anomaly_count = result["is_anomaly"].sum()
        assert anomaly_count >= 5  # reasonably, most of the 10 outliers

    def test_isolation_forest_scores_normalized(self):
        """Anomaly scores should be in [0, 1] range."""
        try:
            from src.models.lab_anomaly_detector import LabAnomalyDetector
        except ImportError:
            pytest.skip("Required modules not available")

        np.random.seed(42)
        pdf = pd.DataFrame(
            np.random.randn(50, 2) * 2.0 + 10.0,
            columns=["val1", "val2"],
        )

        detector = LabAnomalyDetector(contamination=0.1, n_estimators=50)
        detector.train_isolation_forest(pdf)
        result = detector.detect_anomalies(pdf)

        assert result["anomaly_score"].min() >= 0.0
        assert result["anomaly_score"].max() <= 1.0

    def test_detect_anomalies_before_training_raises(self):
        """Calling detect_anomalies without training should raise RuntimeError."""
        try:
            from src.models.lab_anomaly_detector import LabAnomalyDetector
        except ImportError:
            pytest.skip("Required modules not available")

        detector = LabAnomalyDetector()
        pdf = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(RuntimeError):
            detector.detect_anomalies(pdf)


# ------------------------------------------------------------------ #
#  ClinicalEnsemble - Risk Levels
# ------------------------------------------------------------------ #
class TestEnsembleRiskLevels:

    def test_risk_level_critical(self):
        from src.models.ensemble_model import ClinicalEnsemble
        ens = ClinicalEnsemble()
        assert ens.assign_risk_level(0.80) == "Critical"
        assert ens.assign_risk_level(0.99) == "Critical"

    def test_risk_level_high(self):
        from src.models.ensemble_model import ClinicalEnsemble
        ens = ClinicalEnsemble()
        assert ens.assign_risk_level(0.55) == "High"
        assert ens.assign_risk_level(0.75) == "High"

    def test_risk_level_moderate(self):
        from src.models.ensemble_model import ClinicalEnsemble
        ens = ClinicalEnsemble()
        assert ens.assign_risk_level(0.30) == "Moderate"
        assert ens.assign_risk_level(0.50) == "Moderate"

    def test_risk_level_low(self):
        from src.models.ensemble_model import ClinicalEnsemble
        ens = ClinicalEnsemble()
        assert ens.assign_risk_level(0.10) == "Low"
        assert ens.assign_risk_level(0.25) == "Low"
        assert ens.assign_risk_level(0.0) == "Low"

    def test_risk_level_boundary_075(self):
        """0.75 is not > 0.75, so should be High."""
        from src.models.ensemble_model import ClinicalEnsemble
        ens = ClinicalEnsemble()
        assert ens.assign_risk_level(0.75) == "High"

    def test_risk_level_boundary_025(self):
        """0.25 is not > 0.25, so should be Low."""
        from src.models.ensemble_model import ClinicalEnsemble
        ens = ClinicalEnsemble()
        assert ens.assign_risk_level(0.25) == "Low"


# ------------------------------------------------------------------ #
#  ClinicalEnsemble - Weighted Fusion
# ------------------------------------------------------------------ #
class TestEnsembleWeightedFusion:

    def test_weighted_fusion_computation(self):
        """Test that weighted_fusion computes correct weighted average."""
        from src.models.ensemble_model import ClinicalEnsemble

        weights = {
            "structured": 0.30,
            "nlp": 0.25,
            "lab": 0.25,
            "imaging": 0.20,
        }
        ens = ClinicalEnsemble(weights=weights)

        df = pd.DataFrame({
            "patient_id": ["P001"],
            "structured_prob": [0.8],
            "nlp_prob": [0.6],
            "lab_prob": [0.4],
            "imaging_prob": [0.2],
        })

        result = ens.weighted_fusion(df)
        expected = 0.30 * 0.8 + 0.25 * 0.6 + 0.25 * 0.4 + 0.20 * 0.2
        assert abs(result["fused_probability"].iloc[0] - expected) < 1e-6

    def test_weighted_fusion_has_risk_category(self):
        """Weighted fusion should add a risk_category column."""
        from src.models.ensemble_model import ClinicalEnsemble

        ens = ClinicalEnsemble()
        df = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "structured_prob": [0.9, 0.1],
            "nlp_prob": [0.8, 0.2],
            "lab_prob": [0.7, 0.1],
            "imaging_prob": [0.6, 0.05],
        })

        result = ens.weighted_fusion(df)
        assert "risk_category" in result.columns
        assert "fused_probability" in result.columns
        assert len(result) == 2

    def test_weighted_fusion_uniform_weights(self):
        """With uniform probabilities across modalities, fusion should equal that probability."""
        from src.models.ensemble_model import ClinicalEnsemble

        ens = ClinicalEnsemble()
        df = pd.DataFrame({
            "patient_id": ["P001"],
            "structured_prob": [0.5],
            "nlp_prob": [0.5],
            "lab_prob": [0.5],
            "imaging_prob": [0.5],
        })

        result = ens.weighted_fusion(df)
        # All inputs are 0.5, weights sum to 1.0, so result should be 0.5
        assert abs(result["fused_probability"].iloc[0] - 0.5) < 1e-6


# ------------------------------------------------------------------ #
#  Markers for heavy tests (NLP / Image)
# ------------------------------------------------------------------ #
@pytest.mark.skip(reason="NLP model requires transformers and GPU; skip in unit tests")
class TestClinicalNLPModel:
    def test_nlp_model_placeholder(self):
        pass


@pytest.mark.skip(reason="Image model requires torchvision and image data; skip in unit tests")
class TestMedicalImageClassifier:
    def test_image_classifier_placeholder(self):
        pass
