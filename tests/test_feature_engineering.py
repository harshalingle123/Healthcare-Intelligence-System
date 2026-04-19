"""
Tests for feature engineering modules: MEWS score, comorbidity index,
symptom clusters, lab deviation scores, and abnormality count.
"""

import pytest

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType,
        DoubleType, FloatType,
    )
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PYSPARK_AVAILABLE, reason="PySpark is not installed"
)


# ------------------------------------------------------------------ #
#  MEWS Score
# ------------------------------------------------------------------ #
class TestMEWSScoreComputation:

    def test_mews_score_column_exists(self, spark_session):
        """Verify that compute_vital_risk_scores adds a mews_score column."""
        from src.feature_engineering.structured_features import StructuredFeatureEngineer

        data = [
            ("P001", 120.0, 80, 16, 37.0, 98.0),
            ("P002", 85.0, 110, 25, 39.5, 88.0),
            ("P003", 65.0, 35, 8, 34.5, 82.0),
        ]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("systolic_bp", DoubleType(), True),
            StructField("heart_rate", IntegerType(), True),
            StructField("respiratory_rate", IntegerType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("oxygen_saturation", DoubleType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = StructuredFeatureEngineer()
        result = eng.compute_vital_risk_scores(df)

        assert "mews_score" in result.columns
        assert "qsofa_score" in result.columns

    def test_mews_score_ranges(self, spark_session):
        """Verify MEWS scores are within valid range (0-14 theoretical max)."""
        from src.feature_engineering.structured_features import StructuredFeatureEngineer

        data = [
            ("P001", 120.0, 80, 16, 37.0, 98.0),   # all normal -> low score
            ("P002", 65.0, 135, 32, 39.5, 82.0),    # all abnormal -> high score
        ]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("systolic_bp", DoubleType(), True),
            StructField("heart_rate", IntegerType(), True),
            StructField("respiratory_rate", IntegerType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("oxygen_saturation", DoubleType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = StructuredFeatureEngineer()
        result = eng.compute_vital_risk_scores(df)
        rows = result.select("patient_id", "mews_score").collect()
        scores = {row["patient_id"]: row["mews_score"] for row in rows}

        # Normal vitals should have low MEWS
        assert 0 <= scores["P001"] <= 2
        # Abnormal vitals should have high MEWS
        assert scores["P002"] >= 5

    def test_qsofa_score_range(self, spark_session):
        """qSOFA score should be between 0 and 3."""
        from src.feature_engineering.structured_features import StructuredFeatureEngineer

        data = [
            ("P001", 120.0, 80, 16, 37.0, 98.0),
            ("P002", 90.0, 100, 24, 38.0, 85.0),
        ]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("systolic_bp", DoubleType(), True),
            StructField("heart_rate", IntegerType(), True),
            StructField("respiratory_rate", IntegerType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("oxygen_saturation", DoubleType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = StructuredFeatureEngineer()
        result = eng.compute_vital_risk_scores(df)
        rows = result.select("qsofa_score").collect()
        for row in rows:
            assert 0 <= row["qsofa_score"] <= 3


# ------------------------------------------------------------------ #
#  Comorbidity Index
# ------------------------------------------------------------------ #
class TestComorbidityIndex:

    def test_comorbidity_index_with_known_inputs(self, spark_session):
        """Test Charlson-style comorbidity index with known binary inputs."""
        from src.feature_engineering.structured_features import StructuredFeatureEngineer

        # diabetes=1*1, hypertension=1*1, heart=1*2, liver=0*3, lung=0*1 = 4
        data = [
            ("P001", 1, 1, 1, 0, 0),
            ("P002", 0, 0, 0, 0, 0),
            ("P003", 1, 1, 1, 1, 1),  # 1+1+2+3+1 = 8
        ]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("diabetes", IntegerType(), True),
            StructField("hypertension", IntegerType(), True),
            StructField("heart_disease_history", IntegerType(), True),
            StructField("liver_disease_history", IntegerType(), True),
            StructField("lung_disease_history", IntegerType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = StructuredFeatureEngineer()
        result = eng.compute_comorbidity_index(df)
        rows = result.select("patient_id", "comorbidity_index").collect()
        index_map = {row["patient_id"]: row["comorbidity_index"] for row in rows}

        assert index_map["P001"] == 4
        assert index_map["P002"] == 0
        assert index_map["P003"] == 8

    def test_comorbidity_index_non_negative(self, spark_session):
        """Comorbidity index should never be negative."""
        from src.feature_engineering.structured_features import StructuredFeatureEngineer

        data = [("P001", 0, 0, 0, 0, 0)]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("diabetes", IntegerType(), True),
            StructField("hypertension", IntegerType(), True),
            StructField("heart_disease_history", IntegerType(), True),
            StructField("liver_disease_history", IntegerType(), True),
            StructField("lung_disease_history", IntegerType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = StructuredFeatureEngineer()
        result = eng.compute_comorbidity_index(df)
        row = result.select("comorbidity_index").collect()[0]
        assert row["comorbidity_index"] >= 0


# ------------------------------------------------------------------ #
#  Symptom Clusters
# ------------------------------------------------------------------ #
class TestSymptomClusters:

    def test_symptom_cluster_sums(self, spark_session):
        """Verify cluster sums match individual symptom values."""
        from src.feature_engineering.structured_features import StructuredFeatureEngineer

        data = [
            # respiratory: cough=1, wheezing=0, sob=1, sputum=0 -> 2
            # cardiac: chest_pain=1, palpitations=0, edema=0 -> 1
            # gi: nausea=1, vomiting=0, abdominal_pain=1 -> 2
            # metabolic: excessive_thirst=0, frequent_urination=0, weight_loss=1 -> 1
            ("P001", 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1),
        ]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("cough", IntegerType(), True),
            StructField("wheezing", IntegerType(), True),
            StructField("shortness_of_breath", IntegerType(), True),
            StructField("sputum_production", IntegerType(), True),
            StructField("chest_pain", IntegerType(), True),
            StructField("palpitations", IntegerType(), True),
            StructField("edema", IntegerType(), True),
            StructField("nausea", IntegerType(), True),
            StructField("vomiting", IntegerType(), True),
            StructField("abdominal_pain", IntegerType(), True),
            StructField("excessive_thirst", IntegerType(), True),
            StructField("frequent_urination", IntegerType(), True),
            StructField("weight_loss", IntegerType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = StructuredFeatureEngineer()
        result = eng.compute_symptom_clusters(df)
        row = result.collect()[0]

        assert row["respiratory_cluster"] == 2
        assert row["cardiac_cluster"] == 1
        assert row["gi_cluster"] == 2
        assert row["metabolic_cluster"] == 1

    def test_symptom_clusters_all_zero(self, spark_session):
        """All zeros should give zero cluster sums."""
        from src.feature_engineering.structured_features import StructuredFeatureEngineer

        data = [("P001", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("cough", IntegerType(), True),
            StructField("wheezing", IntegerType(), True),
            StructField("shortness_of_breath", IntegerType(), True),
            StructField("sputum_production", IntegerType(), True),
            StructField("chest_pain", IntegerType(), True),
            StructField("palpitations", IntegerType(), True),
            StructField("edema", IntegerType(), True),
            StructField("nausea", IntegerType(), True),
            StructField("vomiting", IntegerType(), True),
            StructField("abdominal_pain", IntegerType(), True),
            StructField("excessive_thirst", IntegerType(), True),
            StructField("frequent_urination", IntegerType(), True),
            StructField("weight_loss", IntegerType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = StructuredFeatureEngineer()
        result = eng.compute_symptom_clusters(df)
        row = result.collect()[0]

        assert row["respiratory_cluster"] == 0
        assert row["cardiac_cluster"] == 0
        assert row["gi_cluster"] == 0
        assert row["metabolic_cluster"] == 0


# ------------------------------------------------------------------ #
#  Lab Deviation Scores
# ------------------------------------------------------------------ #
class TestLabDeviationScores:

    def test_deviation_score_with_known_ranges(self, spark_session):
        """Test deviation = (value - midpoint) / half_range."""
        from src.feature_engineering.lab_features import LabFeatureEngineer

        # hemoglobin: range (12.0, 17.5), midpoint = 14.75, half_range = 2.75
        # value 14.75 -> deviation 0.0
        # value 17.5  -> deviation 1.0
        # value 12.0  -> deviation -1.0
        data = [
            ("P001", 14.75),
            ("P002", 17.5),
            ("P003", 12.0),
            ("P004", 20.25),  # outside normal -> deviation > 1
        ]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("hemoglobin", DoubleType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = LabFeatureEngineer()
        ref = {"hemoglobin": (12.0, 17.5)}
        result = eng.compute_deviation_scores(df, reference_ranges=ref)

        assert "hemoglobin_deviation" in result.columns
        rows = result.select("patient_id", "hemoglobin_deviation").collect()
        devs = {row["patient_id"]: row["hemoglobin_deviation"] for row in rows}

        assert abs(devs["P001"] - 0.0) < 0.01
        assert abs(devs["P002"] - 1.0) < 0.01
        assert abs(devs["P003"] - (-1.0)) < 0.01
        assert devs["P004"] > 1.0

    def test_deviation_score_skips_missing_columns(self, spark_session):
        """If a reference test is not in the DataFrame, it should be skipped."""
        from src.feature_engineering.lab_features import LabFeatureEngineer

        data = [("P001", 14.0)]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("hemoglobin", DoubleType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = LabFeatureEngineer()
        ref = {"hemoglobin": (12.0, 17.5), "nonexistent_test": (0.0, 10.0)}
        result = eng.compute_deviation_scores(df, reference_ranges=ref)

        assert "hemoglobin_deviation" in result.columns
        assert "nonexistent_test_deviation" not in result.columns


# ------------------------------------------------------------------ #
#  Abnormality Count
# ------------------------------------------------------------------ #
class TestAbnormalityCount:

    def test_abnormality_count_logic(self, spark_session):
        """Verify counting of out-of-range lab values."""
        from src.feature_engineering.lab_features import LabFeatureEngineer

        data = [
            # hemoglobin normal (12-17.5), glucose abnormal (70-100) -> 1 abnormal
            ("P001", 14.0, 110.0),
            # both abnormal -> 2
            ("P002", 10.0, 110.0),
            # both normal -> 0
            ("P003", 14.0, 85.0),
        ]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("hemoglobin", DoubleType(), True),
            StructField("glucose_fasting", DoubleType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = LabFeatureEngineer()
        ref = {"hemoglobin": (12.0, 17.5), "glucose_fasting": (70.0, 100.0)}
        result = eng.compute_abnormality_count(df, reference_ranges=ref)

        assert "abnormality_count" in result.columns
        rows = result.select("patient_id", "abnormality_count").collect()
        counts = {row["patient_id"]: row["abnormality_count"] for row in rows}

        assert counts["P001"] == 1
        assert counts["P002"] == 2
        assert counts["P003"] == 0

    def test_abnormality_count_no_matching_columns(self, spark_session):
        """When no reference columns exist, abnormality_count should be 0."""
        from src.feature_engineering.lab_features import LabFeatureEngineer

        data = [("P001", 14.0)]
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("some_other_col", DoubleType(), True),
        ])
        df = spark_session.createDataFrame(data, schema=schema)

        eng = LabFeatureEngineer()
        ref = {"hemoglobin": (12.0, 17.5)}
        result = eng.compute_abnormality_count(df, reference_ranges=ref)

        row = result.select("abnormality_count").collect()[0]
        assert row["abnormality_count"] == 0
