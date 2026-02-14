"""
Healthcare Data Loader Module
==============================
Provides schema-enforced CSV loading for each data source in the
Healthcare Intelligence System pipeline.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, DoubleType,
)


class HealthcareDataLoader:
    """Load and validate healthcare CSV datasets with explicit schemas."""

    # ------------------------------------------------------------------ #
    # Schema definitions (matched to generate_synthetic_data.py output)   #
    # ------------------------------------------------------------------ #
    PATIENTS_SCHEMA = StructType([
        StructField("patient_id", StringType(), False),
        StructField("age", IntegerType(), True),
        StructField("sex", StringType(), True),
        StructField("bmi", FloatType(), True),
        StructField("smoking_status", StringType(), True),
        StructField("diabetes", IntegerType(), True),
        StructField("hypertension", IntegerType(), True),
        StructField("heart_disease_history", IntegerType(), True),
        StructField("liver_disease_history", IntegerType(), True),
        StructField("lung_disease_history", IntegerType(), True),
        StructField("systolic_bp", IntegerType(), True),
        StructField("diastolic_bp", IntegerType(), True),
        StructField("heart_rate", IntegerType(), True),
        StructField("temperature", FloatType(), True),
        StructField("respiratory_rate", IntegerType(), True),
        StructField("oxygen_saturation", IntegerType(), True),
    ])

    SYMPTOMS_SCHEMA = StructType(
        [StructField("patient_id", StringType(), False),
         StructField("timestamp", StringType(), True)]
        + [StructField(s, IntegerType(), True) for s in [
            "cough", "fever", "chest_pain", "shortness_of_breath", "fatigue",
            "nausea", "vomiting", "abdominal_pain", "headache", "dizziness",
            "weight_loss", "night_sweats", "joint_pain", "swelling", "jaundice",
            "dark_urine", "loss_of_appetite", "excessive_thirst",
            "frequent_urination", "blurred_vision", "palpitations", "wheezing",
            "sputum_production", "hemoptysis", "edema", "confusion",
            "muscle_weakness", "numbness", "skin_rash",
        ]]
        + [StructField("primary_diagnosis", StringType(), True)]
    )

    LAB_RESULTS_SCHEMA = StructType([
        StructField("patient_id", StringType(), False),
        StructField("test_date", StringType(), True),
        StructField("test_name", StringType(), True),
        StructField("value", DoubleType(), True),
        StructField("unit", StringType(), True),
        StructField("reference_low", DoubleType(), True),
        StructField("reference_high", DoubleType(), True),
    ])

    CLINICAL_NOTES_SCHEMA = StructType([
        StructField("patient_id", StringType(), False),
        StructField("note_date", StringType(), True),
        StructField("note_type", StringType(), True),
        StructField("note_text", StringType(), True),
    ])

    IMAGE_METADATA_SCHEMA = StructType([
        StructField("patient_id", StringType(), False),
        StructField("image_id", StringType(), True),
        StructField("image_path", StringType(), True),
        StructField("modality", StringType(), True),
        StructField("body_part", StringType(), True),
        StructField("finding_label", StringType(), True),
    ])

    GROUND_TRUTH_SCHEMA = StructType([
        StructField("patient_id", StringType(), False),
        StructField("final_diagnosis", StringType(), True),
        StructField("risk_level", StringType(), True),
        StructField("requires_icu", IntegerType(), True),
    ])

    # ------------------------------------------------------------------ #
    # Loaders                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_csv(spark: SparkSession, path: str, schema: StructType) -> DataFrame:
        return (
            spark.read
            .option("header", "true")
            .option("enforceSchema", "true")
            .option("mode", "DROPMALFORMED")
            .schema(schema)
            .csv(path)
        )

    @classmethod
    def load_patients(cls, spark, path):
        return cls._read_csv(spark, path, cls.PATIENTS_SCHEMA)

    @classmethod
    def load_symptoms(cls, spark, path):
        return cls._read_csv(spark, path, cls.SYMPTOMS_SCHEMA)

    @classmethod
    def load_lab_results(cls, spark, path):
        return cls._read_csv(spark, path, cls.LAB_RESULTS_SCHEMA)

    @classmethod
    def load_clinical_notes(cls, spark, path):
        return cls._read_csv(spark, path, cls.CLINICAL_NOTES_SCHEMA)

    @classmethod
    def load_image_metadata(cls, spark, path):
        return cls._read_csv(spark, path, cls.IMAGE_METADATA_SCHEMA)

    @classmethod
    def load_ground_truth(cls, spark, path):
        return cls._read_csv(spark, path, cls.GROUND_TRUTH_SCHEMA)

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_data(df: DataFrame, name: str) -> None:
        from pyspark.sql import functions as F

        print(f"\n{'='*60}")
        print(f"  Validation Report: {name}")
        print(f"{'='*60}")

        row_count = df.count()
        print(f"  Total rows : {row_count}")
        print(f"  Columns    : {len(df.columns)}")

        print(f"\n  Schema:")
        for field in df.schema.fields:
            print(f"    {field.name:30s} {str(field.dataType):20s} nullable={field.nullable}")

        null_exprs = [
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in df.columns
        ]
        null_counts = df.select(null_exprs).collect()[0]

        print(f"\n  Null counts:")
        for col_name in df.columns:
            cnt = null_counts[col_name]
            pct = (cnt / row_count * 100) if row_count > 0 else 0.0
            print(f"    {col_name:30s} {cnt:>8d}  ({pct:5.1f}%)")

        print(f"{'='*60}\n")
