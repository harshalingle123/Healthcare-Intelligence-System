"""
Shared pytest fixtures for Healthcare Intelligence System tests.
"""

import pytest

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType,
        FloatType, DoubleType, DateType,
    )
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@pytest.fixture(scope="session")
def spark_session():
    """Create a lightweight SparkSession for testing (session-scoped)."""
    if not PYSPARK_AVAILABLE:
        pytest.skip("PySpark is not installed")

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("HealthcareTestSuite")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def sample_patients_df(spark_session):
    """Small DataFrame of 10 patients for unit testing."""
    from datetime import date

    data = [
        ("P001", 25, "M", 70.0, 175.0, 22.9, "A+", "never", "none", "moderate", "none", "none", date(2024, 1, 1)),
        ("P002", 45, "F", 85.0, 160.0, 33.2, "B+", "current", "moderate", "low", "diabetes", "hypertension", date(2024, 1, 2)),
        ("P003", 12, "M", 40.0, 150.0, 17.8, "O+", "never", "none", "high", "none", "none", date(2024, 1, 3)),
        ("P004", 67, "F", 60.0, 155.0, 24.9, "AB-", "former", "light", "low", "heart_disease", "copd", date(2024, 1, 4)),
        ("P005", 38, "M", 95.0, 180.0, 29.3, "A-", "never", "heavy", "moderate", "none", "none", date(2024, 1, 5)),
        ("P006", 55, "F", 72.0, 165.0, 26.4, "B-", "current", "none", "low", "diabetes", "none", date(2024, 1, 6)),
        ("P007", 80, "M", 65.0, 170.0, 22.5, "O-", "former", "light", "low", "heart_disease", "diabetes", date(2024, 1, 7)),
        ("P008", 30, "F", 58.0, 162.0, 22.1, "A+", "never", "none", "high", "none", "none", date(2024, 1, 8)),
        ("P009", 50, "M", 110.0, 178.0, 34.7, "AB+", "current", "heavy", "low", "hypertension", "diabetes", date(2024, 1, 9)),
        ("P010", 42, "F", 68.0, 168.0, 24.1, "B+", "never", "moderate", "moderate", "none", "none", date(2024, 1, 10)),
    ]

    schema = StructType([
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

    return spark_session.createDataFrame(data, schema=schema)


@pytest.fixture
def sample_symptoms_df(spark_session):
    """Small symptoms DataFrame for unit testing."""
    from datetime import date

    data = [
        ("P001", 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, date(2024, 1, 1)),
        ("P002", 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, date(2024, 1, 2)),
        ("P003", 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, date(2024, 1, 3)),
        ("P004", 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, date(2024, 1, 4)),
        ("P005", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, date(2024, 1, 5)),
        ("P006", 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, date(2024, 1, 6)),
        ("P007", 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, date(2024, 1, 7)),
        ("P008", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, date(2024, 1, 8)),
        ("P009", 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, date(2024, 1, 9)),
        ("P010", 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, date(2024, 1, 10)),
    ]

    schema = StructType([
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

    return spark_session.createDataFrame(data, schema=schema)


@pytest.fixture
def sample_lab_df(spark_session):
    """Small lab results DataFrame for unit testing."""
    from datetime import date

    data = [
        ("P001", "hemoglobin", 14.5, "g/dL", 12.0, 17.5, date(2024, 1, 1)),
        ("P001", "glucose_fasting", 95.0, "mg/dL", 70.0, 100.0, date(2024, 1, 1)),
        ("P002", "hemoglobin", 10.5, "g/dL", 12.0, 17.5, date(2024, 1, 2)),
        ("P002", "glucose_fasting", 180.0, "mg/dL", 70.0, 100.0, date(2024, 1, 2)),
        ("P003", "hemoglobin", 13.0, "g/dL", 12.0, 17.5, date(2024, 1, 3)),
        ("P003", "glucose_fasting", 88.0, "mg/dL", 70.0, 100.0, date(2024, 1, 3)),
        ("P004", "hemoglobin", 11.0, "g/dL", 12.0, 17.5, date(2024, 1, 4)),
        ("P004", "glucose_fasting", 250.0, "mg/dL", 70.0, 100.0, date(2024, 1, 4)),
        ("P005", "hemoglobin", 15.0, "g/dL", 12.0, 17.5, date(2024, 1, 5)),
        ("P005", "glucose_fasting", 72.0, "mg/dL", 70.0, 100.0, date(2024, 1, 5)),
    ]

    schema = StructType([
        StructField("patient_id", StringType(), False),
        StructField("test_name", StringType(), True),
        StructField("test_value", DoubleType(), True),
        StructField("unit", StringType(), True),
        StructField("reference_low", DoubleType(), True),
        StructField("reference_high", DoubleType(), True),
        StructField("test_date", DateType(), True),
    ])

    return spark_session.createDataFrame(data, schema=schema)
