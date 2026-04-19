"""
Integration tests for the full training and inference pipelines.
Marked as slow since they involve Spark and model training.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.mark.slow
class TestFullPipelineMini:
    """Run the pipeline on a tiny subset (100 patients) and verify output format."""

    def test_full_pipeline_mini(self, spark_session):
        """Verify end-to-end pipeline produces expected output columns on 100-patient data."""
        try:
            from pyspark.sql import functions as F
            from pyspark.sql.types import (
                StructType, StructField, StringType, IntegerType,
                DoubleType, FloatType, DateType,
            )
            from src.data_processing.preprocessor import DataPreprocessor
        except ImportError:
            pytest.skip("Required modules not available")

        from datetime import date
        import random

        random.seed(42)
        np.random.seed(42)

        # Generate 100 synthetic patients
        patient_data = []
        for i in range(100):
            pid = f"P{i:04d}"
            age = random.randint(5, 90)
            gender = random.choice(["M", "F"])
            weight = round(random.uniform(40.0, 120.0), 1)
            height = round(random.uniform(140.0, 200.0), 1)
            bmi = round(weight / (height / 100.0) ** 2, 1)
            patient_data.append((
                pid, age, gender, weight, height, bmi,
                random.choice(["A+", "B+", "O+", "AB+"]),
                random.choice(["never", "current", "former"]),
                random.choice(["none", "light", "moderate", "heavy"]),
                random.choice(["low", "moderate", "high"]),
                "none", "none",
                date(2024, 1, random.randint(1, 28)),
            ))

        patients_schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("age", IntegerType(), True),
            StructField("gender", StringType(), True),
            StructField("weight", FloatType(), True),
            StructField("height", FloatType(), True),
            StructField("bmi", FloatType(), True),
            StructField("blood_type", StringType(), True),
            StructField("smoking_status", StringType(), True),
            StructField("alcohol_consumption", StringType(), True),
            StructField("physical_activity", StringType(), True),
            StructField("family_history", StringType(), True),
            StructField("previous_conditions", StringType(), True),
            StructField("admission_date", DateType(), True),
        ])
        patients_df = spark_session.createDataFrame(patient_data, schema=patients_schema)

        # Generate symptoms for those patients
        symptom_data = []
        for i in range(100):
            pid = f"P{i:04d}"
            symptoms = [random.randint(0, 1) for _ in range(12)]
            symptom_data.append(
                (pid, *symptoms, date(2024, 1, random.randint(1, 28)))
            )

        symptoms_schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("fever", IntegerType(), True),
            StructField("cough", IntegerType(), True),
            StructField("fatigue", IntegerType(), True),
            StructField("shortness_of_breath", IntegerType(), True),
            StructField("chest_pain", IntegerType(), True),
            StructField("headache", IntegerType(), True),
            StructField("nausea", IntegerType(), True),
            StructField("dizziness", IntegerType(), True),
            StructField("joint_pain", IntegerType(), True),
            StructField("weight_loss", IntegerType(), True),
            StructField("night_sweats", IntegerType(), True),
            StructField("abdominal_pain", IntegerType(), True),
            StructField("recorded_date", DateType(), True),
        ])
        symptoms_df = spark_session.createDataFrame(symptom_data, schema=symptoms_schema)

        # Preprocess
        processed_patients = DataPreprocessor.preprocess_patients(patients_df)
        processed_symptoms = DataPreprocessor.preprocess_symptoms(symptoms_df)

        # Join
        integrated = processed_patients.join(processed_symptoms, on="patient_id", how="left")

        # Verify output format
        assert integrated.count() == 100
        assert "patient_id" in integrated.columns
        assert "age_group" in integrated.columns
        assert "bmi_category" in integrated.columns
        assert "symptom_count" in integrated.columns

        # Verify no nulls in derived columns
        null_age_group = integrated.filter(F.col("age_group").isNull()).count()
        null_bmi_cat = integrated.filter(F.col("bmi_category").isNull()).count()
        assert null_age_group == 0
        assert null_bmi_cat == 0


@pytest.mark.slow
class TestInferencePipelineFormat:
    """Verify prediction output has required keys."""

    def test_inference_pipeline_format(self):
        """Check that InferencePipeline.predict_single_patient returns correct keys."""
        try:
            from src.pipeline.inference_pipeline import InferencePipeline
        except ImportError:
            pytest.skip("Required modules not available")

        # We cannot load actual models in a unit test, so we test
        # the output format with a mocked prediction.
        pipeline = InferencePipeline.__new__(InferencePipeline)
        pipeline.config = {"paths": {"models": "nonexistent"}, "models": {}}
        pipeline.models_dir = "nonexistent"
        pipeline.symptom_classifier = None
        pipeline.nlp_model = None
        pipeline.lab_detector = None
        pipeline.image_classifier = None
        pipeline.ensemble = None
        pipeline._models_loaded = True
        pipeline.label_map = {0: "Respiratory", 1: "Cardiovascular", 2: "Metabolic",
                              3: "Gastrointestinal", 4: "Neurological"}

        patient_data = {
            "patient_id": "TEST001",
            "age": 55,
            "bmi": 28.0,
            "heart_rate": 90,
            "systolic_bp": 140,
            "respiratory_rate": 18,
            "temperature": 37.5,
            "oxygen_saturation": 95,
            "fever": 1,
            "cough": 1,
        }

        result = pipeline.predict_single_patient(patient_data)

        required_keys = [
            "patient_id",
            "risk_level",
            "confidence",
            "predicted_diagnosis",
            "per_modality_scores",
            "top_factors",
            "recommended_actions",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        assert result["patient_id"] == "TEST001"
        assert isinstance(result["risk_level"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["predicted_diagnosis"], str)
        assert isinstance(result["per_modality_scores"], dict)
        assert isinstance(result["top_factors"], list)
        assert isinstance(result["recommended_actions"], list)

    def test_inference_risk_levels_valid(self):
        """Verify that risk levels are one of the expected values."""
        try:
            from src.pipeline.inference_pipeline import InferencePipeline
        except ImportError:
            pytest.skip("Required modules not available")

        pipeline = InferencePipeline.__new__(InferencePipeline)
        valid_levels = {"Low", "Medium", "High", "Critical"}

        # Test the _risk_level method directly
        assert pipeline._risk_level(0.3, 0) in valid_levels
        assert pipeline._risk_level(0.6, 0) in valid_levels
        assert pipeline._risk_level(0.8, 0) in valid_levels
        assert pipeline._risk_level(0.9, 1) in valid_levels  # high-severity class
