"""
Data Preprocessing Module
==========================
PySpark-only preprocessing pipelines for each data source in the
Healthcare Intelligence System.  No pandas dependency.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml.feature import Imputer


class DataPreprocessor:
    """Preprocessing transformations for every healthcare data source."""

    # ------------------------------------------------------------------ #
    # 1. Patients                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess_patients(df: DataFrame) -> DataFrame:
        """
        Preprocess patient demographics.

        Steps
        -----
        * Cast numeric columns to DoubleType (required by Imputer).
        * Impute missing numerics (age, weight, height, bmi) with median.
        * Create ``age_group`` categorical column.
        * Create ``bmi_category`` column based on WHO thresholds.
        """
        numeric_cols = ["age", "weight", "height", "bmi"]

        # Cast to double for Imputer compatibility
        for c in numeric_cols:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

        # Median imputation via PySpark ML Imputer
        imputer = Imputer(
            inputCols=numeric_cols,
            outputCols=[f"{c}_imputed" for c in numeric_cols],
            strategy="median",
        )
        model = imputer.fit(df)
        df = model.transform(df)

        # Replace originals with imputed values and drop temporaries
        for c in numeric_cols:
            df = df.withColumn(c, F.col(f"{c}_imputed")).drop(f"{c}_imputed")

        # Cast age back to integer
        df = df.withColumn("age", F.col("age").cast(IntegerType()))

        # Age group
        df = df.withColumn(
            "age_group",
            F.when(F.col("age") < 18, "pediatric")
            .when(F.col("age") < 35, "young_adult")
            .when(F.col("age") < 50, "middle_aged")
            .when(F.col("age") < 65, "senior")
            .otherwise("elderly"),
        )

        # BMI category (WHO)
        df = df.withColumn(
            "bmi_category",
            F.when(F.col("bmi") < 18.5, "underweight")
            .when(F.col("bmi") < 25.0, "normal")
            .when(F.col("bmi") < 30.0, "overweight")
            .otherwise("obese"),
        )

        return df

    # ------------------------------------------------------------------ #
    # 2. Symptoms                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess_symptoms(df: DataFrame) -> DataFrame:
        """
        Preprocess symptom indicators.

        Steps
        -----
        * Validate that binary columns contain only 0, 1, or null.
        * Replace invalid values with null.
        * Compute ``symptom_count`` as the sum of all symptom flags.
        """
        binary_cols = [
            c for c in df.columns
            if c not in ("patient_id", "recorded_date")
        ]

        # Clamp to valid binary range
        for c in binary_cols:
            df = df.withColumn(
                c,
                F.when(F.col(c).isin(0, 1), F.col(c)).otherwise(F.lit(None)).cast(IntegerType()),
            )

        # Total active symptoms (nulls treated as 0)
        df = df.withColumn(
            "symptom_count",
            sum(F.coalesce(F.col(c), F.lit(0)) for c in binary_cols),
        )

        return df

    # ------------------------------------------------------------------ #
    # 3. Lab results                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess_lab_results(df: DataFrame) -> DataFrame:
        """
        Preprocess laboratory results.

        Steps
        -----
        * Remove outliers beyond 3 IQR from Q1/Q3 per test_name.
        * Add ``is_abnormal`` flag (value outside reference range).
        * Pivot to wide format: one column per test_name holding the
          *latest* test value for each patient.
        """
        # -- Outlier removal (per test) -------------------------------- #
        stats_window = Window.partitionBy("test_name")

        df = df.withColumn("q1", F.percentile_approx("test_value", 0.25).over(stats_window))
        df = df.withColumn("q3", F.percentile_approx("test_value", 0.75).over(stats_window))
        df = df.withColumn("iqr", F.col("q3") - F.col("q1"))

        df = df.filter(
            (F.col("test_value") >= F.col("q1") - 3 * F.col("iqr"))
            & (F.col("test_value") <= F.col("q3") + 3 * F.col("iqr"))
        ).drop("q1", "q3", "iqr")

        # -- Abnormal flag --------------------------------------------- #
        df = df.withColumn(
            "is_abnormal",
            F.when(
                (F.col("test_value") < F.col("reference_low"))
                | (F.col("test_value") > F.col("reference_high")),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )

        # -- Keep latest result per patient per test ------------------- #
        row_window = Window.partitionBy("patient_id", "test_name").orderBy(F.col("test_date").desc())
        df = df.withColumn("rn", F.row_number().over(row_window)).filter(F.col("rn") == 1).drop("rn")

        # -- Pivot to wide format -------------------------------------- #
        value_pivot = (
            df.groupBy("patient_id")
            .pivot("test_name")
            .agg(F.first("test_value"))
        )

        abnormal_pivot = (
            df.groupBy("patient_id")
            .pivot("test_name")
            .agg(F.first("is_abnormal"))
        )

        # Rename abnormal columns to <test>_abnormal
        for c in abnormal_pivot.columns:
            if c != "patient_id":
                abnormal_pivot = abnormal_pivot.withColumnRenamed(c, f"{c}_abnormal")

        result = value_pivot.join(abnormal_pivot, on="patient_id", how="left")
        return result

    # ------------------------------------------------------------------ #
    # 4. Clinical notes                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess_clinical_notes(df: DataFrame) -> DataFrame:
        """
        Preprocess clinical notes text.

        Steps
        -----
        * Lowercase all text.
        * Remove special characters (keep letters, digits, spaces).
        * Normalize common medical abbreviations via a UDF.
        """
        # Map of common medical abbreviations -> expanded form
        _ABBREV_MAP = {
            "pt": "patient",
            "dx": "diagnosis",
            "hx": "history",
            "rx": "prescription",
            "sx": "symptoms",
            "tx": "treatment",
            "bp": "blood pressure",
            "hr": "heart rate",
            "rr": "respiratory rate",
            "sob": "shortness of breath",
            "nkda": "no known drug allergies",
            "prn": "as needed",
            "bid": "twice daily",
            "tid": "three times daily",
            "qid": "four times daily",
            "po": "by mouth",
            "iv": "intravenous",
            "im": "intramuscular",
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "cad": "coronary artery disease",
            "uti": "urinary tract infection",
            "bmp": "basic metabolic panel",
            "cbc": "complete blood count",
            "ct": "computed tomography",
            "mri": "magnetic resonance imaging",
        }

        @F.udf(StringType())
        def normalize_abbreviations(text):
            if text is None:
                return None
            tokens = text.split()
            normalized = [_ABBREV_MAP.get(t, t) for t in tokens]
            return " ".join(normalized)

        # Lowercase
        df = df.withColumn("note_text", F.lower(F.col("note_text")))

        # Remove special characters (retain alphanumerics and whitespace)
        df = df.withColumn("note_text", F.regexp_replace(F.col("note_text"), r"[^a-z0-9\s]", ""))

        # Collapse multiple spaces
        df = df.withColumn("note_text", F.regexp_replace(F.col("note_text"), r"\s+", " "))
        df = df.withColumn("note_text", F.trim(F.col("note_text")))

        # Abbreviation normalization
        df = df.withColumn("note_text", normalize_abbreviations(F.col("note_text")))

        # Compute word count as a basic feature
        df = df.withColumn("word_count", F.size(F.split(F.col("note_text"), " ")))

        return df

    # ------------------------------------------------------------------ #
    # 5. Image metadata                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess_images(metadata_df: DataFrame) -> DataFrame:
        """
        Preprocess medical image metadata.

        Steps
        -----
        * Validate that image_path is non-empty.
        * Standardize modality to uppercase.
        * Standardize body_part to lowercase.
        * Flag rows with missing or invalid paths.
        """
        # Flag invalid paths
        metadata_df = metadata_df.withColumn(
            "path_valid",
            F.when(
                F.col("image_path").isNotNull() & (F.length(F.trim(F.col("image_path"))) > 0),
                F.lit(True),
            ).otherwise(F.lit(False)),
        )

        # Standardize modality (e.g., "xray" -> "XRAY")
        metadata_df = metadata_df.withColumn(
            "modality", F.upper(F.trim(F.col("modality")))
        )

        # Standardize body part
        metadata_df = metadata_df.withColumn(
            "body_part", F.lower(F.trim(F.col("body_part")))
        )

        # Standardize format
        metadata_df = metadata_df.withColumn(
            "format", F.upper(F.trim(F.col("format")))
        )

        # Compute aspect ratio where dimensions are available
        metadata_df = metadata_df.withColumn(
            "aspect_ratio",
            F.when(
                (F.col("width").isNotNull()) & (F.col("height").isNotNull()) & (F.col("height") > 0),
                F.round(F.col("width") / F.col("height"), 2),
            ).otherwise(F.lit(None)),
        )

        return metadata_df
