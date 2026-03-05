"""
Structured Feature Engineering for Healthcare Intelligence System.

Computes clinical risk scores, comorbidity indices, interaction features,
symptom clusters, and assembles feature vectors using PySpark.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler


class StructuredFeatureEngineer:
    """Engineers features from structured clinical data using PySpark."""

    # ------------------------------------------------------------------ #
    #  MEWS & qSOFA
    # ------------------------------------------------------------------ #
    def compute_vital_risk_scores(self, df: DataFrame) -> DataFrame:
        """Compute Modified Early Warning Score (MEWS) and qSOFA from vitals.

        Expected columns: systolic_bp, heart_rate, respiratory_rate,
                          temperature, oxygen_saturation.
        Adds columns: mews_score, qsofa_score.
        """
        # --- MEWS components ------------------------------------------------
        # Systolic BP scoring
        sbp_score = (
            F.when(F.col("systolic_bp") <= 70, 3)
            .when(F.col("systolic_bp") <= 80, 2)
            .when(F.col("systolic_bp") <= 100, 1)
            .when(F.col("systolic_bp") <= 199, 0)
            .when(F.col("systolic_bp") >= 200, 2)
            .otherwise(0)
        )

        # Heart rate scoring
        hr_score = (
            F.when(F.col("heart_rate") <= 40, 2)
            .when(F.col("heart_rate") <= 50, 1)
            .when(F.col("heart_rate") <= 100, 0)
            .when(F.col("heart_rate") <= 110, 1)
            .when(F.col("heart_rate") <= 129, 2)
            .when(F.col("heart_rate") >= 130, 3)
            .otherwise(0)
        )

        # Respiratory rate scoring
        rr_score = (
            F.when(F.col("respiratory_rate") < 9, 2)
            .when(F.col("respiratory_rate") <= 14, 0)
            .when(F.col("respiratory_rate") <= 20, 1)
            .when(F.col("respiratory_rate") <= 29, 2)
            .when(F.col("respiratory_rate") >= 30, 3)
            .otherwise(0)
        )

        # Temperature scoring (Celsius)
        temp_score = (
            F.when(F.col("temperature") < 35.0, 2)
            .when(F.col("temperature") <= 38.4, 0)
            .when(F.col("temperature") <= 38.9, 1)
            .when(F.col("temperature") >= 39.0, 2)
            .otherwise(0)
        )

        # Oxygen saturation scoring
        spo2_score = (
            F.when(F.col("oxygen_saturation") < 85, 3)
            .when(F.col("oxygen_saturation") < 90, 2)
            .when(F.col("oxygen_saturation") < 94, 1)
            .when(F.col("oxygen_saturation") >= 94, 0)
            .otherwise(0)
        )

        df = df.withColumn(
            "mews_score",
            (sbp_score + hr_score + rr_score + temp_score + spo2_score).cast(IntegerType()),
        )

        # --- qSOFA (quick Sequential Organ Failure Assessment) ---------------
        # 1 point each for: SBP <= 100, RR >= 22, altered mentation (not available,
        # so we approximate with MEWS >= 3 or SpO2 < 90 as a proxy).
        qsofa = (
            F.when(F.col("systolic_bp") <= 100, 1).otherwise(0)
            + F.when(F.col("respiratory_rate") >= 22, 1).otherwise(0)
            + F.when(F.col("oxygen_saturation") < 90, 1).otherwise(0)
        )
        df = df.withColumn("qsofa_score", qsofa.cast(IntegerType()))

        return df

    # ------------------------------------------------------------------ #
    #  Charlson Comorbidity Index
    # ------------------------------------------------------------------ #
    def compute_comorbidity_index(self, df: DataFrame) -> DataFrame:
        """Compute simplified Charlson Comorbidity Index.

        Expected columns (binary 0/1): diabetes, hypertension,
            heart_disease_history, liver_disease_history, lung_disease_history.
        Adds column: comorbidity_index.
        """
        # Weights aligned with standard Charlson scoring (simplified)
        df = df.withColumn(
            "comorbidity_index",
            (
                F.coalesce(F.col("diabetes").cast(IntegerType()), F.lit(0)) * 1
                + F.coalesce(F.col("hypertension").cast(IntegerType()), F.lit(0)) * 1
                + F.coalesce(F.col("heart_disease_history").cast(IntegerType()), F.lit(0)) * 2
                + F.coalesce(F.col("liver_disease_history").cast(IntegerType()), F.lit(0)) * 3
                + F.coalesce(F.col("lung_disease_history").cast(IntegerType()), F.lit(0)) * 1
            ).cast(IntegerType()),
        )
        return df

    # ------------------------------------------------------------------ #
    #  Interaction Features
    # ------------------------------------------------------------------ #
    def compute_interaction_features(self, df: DataFrame) -> DataFrame:
        """Create clinically meaningful interaction terms.

        Adds columns: age_bmi_interaction, smoking_comorbidity_interaction,
            bp_heart_rate_interaction, age_comorbidity_interaction,
            bmi_diabetes_interaction.
        """
        df = df.withColumn(
            "age_bmi_interaction",
            (F.col("age").cast(DoubleType()) * F.col("bmi").cast(DoubleType())),
        )
        # Encode smoking_status as numeric: Current=2, Former=1, Never=0
        smoking_numeric = (
            F.when(F.col("smoking_status") == "Current", 2.0)
            .when(F.col("smoking_status") == "Former", 1.0)
            .otherwise(0.0)
        )
        df = df.withColumn(
            "smoking_comorbidity_interaction",
            smoking_numeric * F.col("comorbidity_index").cast(DoubleType()),
        )
        df = df.withColumn(
            "bp_heart_rate_interaction",
            (
                F.col("systolic_bp").cast(DoubleType())
                * F.col("heart_rate").cast(DoubleType())
            ),
        )
        df = df.withColumn(
            "age_comorbidity_interaction",
            (
                F.col("age").cast(DoubleType())
                * F.col("comorbidity_index").cast(DoubleType())
            ),
        )
        df = df.withColumn(
            "bmi_diabetes_interaction",
            (
                F.col("bmi").cast(DoubleType())
                * F.coalesce(F.col("diabetes").cast(DoubleType()), F.lit(0.0))
            ),
        )
        return df

    # ------------------------------------------------------------------ #
    #  Symptom Clusters
    # ------------------------------------------------------------------ #
    def compute_symptom_clusters(self, df: DataFrame) -> DataFrame:
        """Aggregate binary symptom columns into clinical clusters.

        Adds columns: respiratory_cluster, cardiac_cluster, gi_cluster,
                      metabolic_cluster.
        """

        def _safe_col(name):
            return F.coalesce(F.col(name).cast(IntegerType()), F.lit(0))

        # Respiratory cluster
        respiratory_symptoms = ["cough", "wheezing", "shortness_of_breath", "sputum_production"]
        df = df.withColumn(
            "respiratory_cluster",
            sum(_safe_col(s) for s in respiratory_symptoms).cast(IntegerType()),
        )

        # Cardiac cluster
        cardiac_symptoms = ["chest_pain", "palpitations", "edema"]
        df = df.withColumn(
            "cardiac_cluster",
            sum(_safe_col(s) for s in cardiac_symptoms).cast(IntegerType()),
        )

        # Gastrointestinal cluster
        gi_symptoms = ["nausea", "vomiting", "abdominal_pain"]
        df = df.withColumn(
            "gi_cluster",
            sum(_safe_col(s) for s in gi_symptoms).cast(IntegerType()),
        )

        # Metabolic cluster
        metabolic_symptoms = ["excessive_thirst", "frequent_urination", "weight_loss"]
        df = df.withColumn(
            "metabolic_cluster",
            sum(_safe_col(s) for s in metabolic_symptoms).cast(IntegerType()),
        )

        return df

    # ------------------------------------------------------------------ #
    #  Feature Vector Assembly
    # ------------------------------------------------------------------ #
    def build_feature_vector(
        self,
        df: DataFrame,
        feature_cols: list[str] | None = None,
        output_col: str = "features",
    ) -> DataFrame:
        """Assemble all numeric feature columns into a single vector column.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame with engineered features.
        feature_cols : list[str] | None
            Explicit list of columns. If None, auto-detects all numeric columns
            excluding identifiers (patient_id, encounter_id, label/target).
        output_col : str
            Name of the output vector column.
        """
        if feature_cols is None:
            exclude = {"patient_id", "encounter_id", "label", "target", "readmission",
                       "timestamp", "primary_diagnosis", "final_diagnosis",
                       "risk_level", "requires_icu", "label_index",
                       "features", "raw_features", "scaled_features",
                       "prediction", "rawPrediction", "probability"}
            numeric_types = {"int", "bigint", "double", "float", "decimal"}
            feature_cols = [
                f.name
                for f in df.schema.fields
                if f.dataType.simpleString().split("(")[0] in numeric_types
                and f.name not in exclude
            ]

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=output_col,
            handleInvalid="skip",
        )
        return assembler.transform(df)
