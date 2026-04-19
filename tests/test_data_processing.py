"""
Tests for data processing modules: SparkSession creation, DataLoader schemas,
DataPreprocessor transformations, and DataIntegrator join logic.
"""

import pytest

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PYSPARK_AVAILABLE, reason="PySpark is not installed"
)


# ------------------------------------------------------------------ #
#  SparkSession creation
# ------------------------------------------------------------------ #
class TestSparkSessionCreation:

    def test_spark_session_creation(self, spark_session):
        """Verify that a SparkSession is successfully created."""
        assert spark_session is not None
        assert isinstance(spark_session, SparkSession)
        assert spark_session.sparkContext is not None

    def test_spark_session_app_name(self, spark_session):
        """Verify the Spark application name is set."""
        app_name = spark_session.sparkContext.appName
        assert app_name is not None and len(app_name) > 0


# ------------------------------------------------------------------ #
#  DataLoader schemas
# ------------------------------------------------------------------ #
class TestDataLoaderSchema:

    def test_patients_schema_is_struct_type(self):
        from src.data_processing.data_loader import HealthcareDataLoader
        assert isinstance(HealthcareDataLoader.PATIENTS_SCHEMA, StructType)

    def test_symptoms_schema_is_struct_type(self):
        from src.data_processing.data_loader import HealthcareDataLoader
        assert isinstance(HealthcareDataLoader.SYMPTOMS_SCHEMA, StructType)

    def test_lab_results_schema_is_struct_type(self):
        from src.data_processing.data_loader import HealthcareDataLoader
        assert isinstance(HealthcareDataLoader.LAB_RESULTS_SCHEMA, StructType)

    def test_clinical_notes_schema_is_struct_type(self):
        from src.data_processing.data_loader import HealthcareDataLoader
        assert isinstance(HealthcareDataLoader.CLINICAL_NOTES_SCHEMA, StructType)

    def test_image_metadata_schema_is_struct_type(self):
        from src.data_processing.data_loader import HealthcareDataLoader
        assert isinstance(HealthcareDataLoader.IMAGE_METADATA_SCHEMA, StructType)

    def test_ground_truth_schema_is_struct_type(self):
        from src.data_processing.data_loader import HealthcareDataLoader
        assert isinstance(HealthcareDataLoader.GROUND_TRUTH_SCHEMA, StructType)

    def test_patients_schema_has_patient_id(self):
        from src.data_processing.data_loader import HealthcareDataLoader
        field_names = [f.name for f in HealthcareDataLoader.PATIENTS_SCHEMA.fields]
        assert "patient_id" in field_names

    def test_symptoms_schema_has_symptom_columns(self):
        from src.data_processing.data_loader import HealthcareDataLoader
        field_names = [f.name for f in HealthcareDataLoader.SYMPTOMS_SCHEMA.fields]
        for col in ["fever", "cough", "fatigue", "shortness_of_breath", "chest_pain"]:
            assert col in field_names


# ------------------------------------------------------------------ #
#  DataPreprocessor - patients
# ------------------------------------------------------------------ #
class TestPreprocessorPatients:

    def test_age_group_creation(self, spark_session, sample_patients_df):
        from src.data_processing.preprocessor import DataPreprocessor
        result = DataPreprocessor.preprocess_patients(sample_patients_df)
        assert "age_group" in result.columns

        rows = result.select("patient_id", "age_group").collect()
        age_groups = {row["patient_id"]: row["age_group"] for row in rows}

        # P003 is 12 -> pediatric
        assert age_groups["P003"] == "pediatric"
        # P001 is 25 -> young_adult
        assert age_groups["P001"] == "young_adult"
        # P005 is 38 -> middle_aged
        assert age_groups["P005"] == "middle_aged"
        # P006 is 55 -> senior
        assert age_groups["P006"] == "senior"
        # P007 is 80 -> elderly
        assert age_groups["P007"] == "elderly"

    def test_bmi_category_creation(self, spark_session, sample_patients_df):
        from src.data_processing.preprocessor import DataPreprocessor
        result = DataPreprocessor.preprocess_patients(sample_patients_df)
        assert "bmi_category" in result.columns

        rows = result.select("patient_id", "bmi_category").collect()
        bmi_cats = {row["patient_id"]: row["bmi_category"] for row in rows}

        # P003 bmi=17.8 -> underweight
        assert bmi_cats["P003"] == "underweight"
        # P001 bmi=22.9 -> normal
        assert bmi_cats["P001"] == "normal"
        # P005 bmi=29.3 -> overweight
        assert bmi_cats["P005"] == "overweight"
        # P002 bmi=33.2 -> obese
        assert bmi_cats["P002"] == "obese"

    def test_preprocessed_row_count_unchanged(self, spark_session, sample_patients_df):
        from src.data_processing.preprocessor import DataPreprocessor
        result = DataPreprocessor.preprocess_patients(sample_patients_df)
        assert result.count() == sample_patients_df.count()


# ------------------------------------------------------------------ #
#  DataPreprocessor - symptoms
# ------------------------------------------------------------------ #
class TestPreprocessorSymptoms:

    def test_symptom_count_computation(self, spark_session, sample_symptoms_df):
        from src.data_processing.preprocessor import DataPreprocessor
        result = DataPreprocessor.preprocess_symptoms(sample_symptoms_df)
        assert "symptom_count" in result.columns

        rows = result.select("patient_id", "symptom_count").collect()
        counts = {row["patient_id"]: row["symptom_count"] for row in rows}

        # P005 has all zeros -> 0
        assert counts["P005"] == 0
        # P008 has all zeros -> 0
        assert counts["P008"] == 0
        # P001 has fever=1, cough=1, headache=1 -> 3
        assert counts["P001"] == 3
        # P009 has 11 symptoms -> 11
        assert counts["P009"] == 11

    def test_symptom_count_non_negative(self, spark_session, sample_symptoms_df):
        from src.data_processing.preprocessor import DataPreprocessor
        result = DataPreprocessor.preprocess_symptoms(sample_symptoms_df)
        rows = result.select("symptom_count").collect()
        for row in rows:
            assert row["symptom_count"] >= 0


# ------------------------------------------------------------------ #
#  DataIntegrator
# ------------------------------------------------------------------ #
class TestDataIntegration:

    def test_join_produces_correct_columns(self, spark_session, sample_patients_df, sample_symptoms_df):
        """Test that joining patients and symptoms produces expected columns."""
        joined = sample_patients_df.join(sample_symptoms_df, on="patient_id", how="left")

        assert "patient_id" in joined.columns
        assert "age" in joined.columns
        assert "fever" in joined.columns
        assert "cough" in joined.columns

    def test_left_join_preserves_all_patients(self, spark_session, sample_patients_df, sample_symptoms_df):
        """Left join should retain all patient rows."""
        joined = sample_patients_df.join(sample_symptoms_df, on="patient_id", how="left")
        assert joined.count() == sample_patients_df.count()

    def test_prefix_columns(self, spark_session, sample_symptoms_df):
        """Test that _prefix_columns adds prefixes correctly."""
        from src.data_processing.data_integrator import DataIntegrator
        prefixed = DataIntegrator._prefix_columns(sample_symptoms_df, "sym")

        assert "patient_id" in prefixed.columns
        assert "sym_fever" in prefixed.columns
        assert "sym_cough" in prefixed.columns
        assert "fever" not in prefixed.columns
