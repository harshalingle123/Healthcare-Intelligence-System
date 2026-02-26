"""
Data Integration Module
========================
Joins all preprocessed healthcare data sources into a single unified
DataFrame keyed by ``patient_id``, using PySpark window functions for
temporal alignment.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class DataIntegrator:
    """Integrate heterogeneous healthcare DataFrames into one unified view."""

    @staticmethod
    def _deduplicate_latest(df: DataFrame, id_col: str, date_col: str) -> DataFrame:
        """
        Keep only the most recent row per *id_col* based on *date_col*.

        Parameters
        ----------
        df : DataFrame
        id_col : str   – partition column (e.g. ``patient_id``)
        date_col : str – ordering column (most recent wins)

        Returns
        -------
        DataFrame with at most one row per *id_col*.
        """
        if date_col not in df.columns:
            return df

        w = Window.partitionBy(id_col).orderBy(F.col(date_col).desc())
        return (
            df.withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )

    @staticmethod
    def _prefix_columns(df: DataFrame, prefix: str, key_col: str = "patient_id") -> DataFrame:
        """
        Add a prefix to every column except *key_col* to prevent
        name collisions after joins.
        """
        for c in df.columns:
            if c != key_col:
                df = df.withColumnRenamed(c, f"{prefix}_{c}")
        return df

    @classmethod
    def integrate_patient_data(
        cls,
        patients_df: DataFrame,
        symptoms_df: DataFrame,
        lab_df: DataFrame,
        notes_df: DataFrame,
        image_meta_df: DataFrame,
        ground_truth_df: DataFrame,
        output_path: str = None,
    ) -> DataFrame:
        """
        Merge all data sources into a single patient-level DataFrame.

        Parameters
        ----------
        patients_df      : Preprocessed patient demographics.
        symptoms_df      : Preprocessed symptom indicators.
        lab_df           : Preprocessed lab results (already pivoted wide).
        notes_df         : Preprocessed clinical notes.
        image_meta_df    : Preprocessed image metadata.
        ground_truth_df  : Ground-truth diagnosis labels.
        output_path      : Optional path to write Parquet output.

        Returns
        -------
        DataFrame – unified, patient-level DataFrame.
        """
        # ------------------------------------------------------------ #
        # 1. Temporal alignment – keep latest record per patient        #
        # ------------------------------------------------------------ #
        symptoms_latest = cls._deduplicate_latest(symptoms_df, "patient_id", "recorded_date")
        notes_latest = cls._deduplicate_latest(notes_df, "patient_id", "created_date")
        images_latest = cls._deduplicate_latest(image_meta_df, "patient_id", "image_date")
        gt_latest = cls._deduplicate_latest(ground_truth_df, "patient_id", "diagnosis_date")

        # Lab DF is already one-row-per-patient after pivoting; no dedup needed.
        lab_latest = lab_df

        # ------------------------------------------------------------ #
        # 2. Prefix columns to avoid collisions                         #
        # ------------------------------------------------------------ #
        symptoms_latest = cls._prefix_columns(symptoms_latest, "sym")
        lab_latest = cls._prefix_columns(lab_latest, "lab")
        notes_latest = cls._prefix_columns(notes_latest, "note")
        images_latest = cls._prefix_columns(images_latest, "img")
        gt_latest = cls._prefix_columns(gt_latest, "gt")

        # ------------------------------------------------------------ #
        # 3. Left-join everything onto the patients spine               #
        # ------------------------------------------------------------ #
        unified = (
            patients_df
            .join(symptoms_latest, on="patient_id", how="left")
            .join(lab_latest, on="patient_id", how="left")
            .join(notes_latest, on="patient_id", how="left")
            .join(images_latest, on="patient_id", how="left")
            .join(gt_latest, on="patient_id", how="left")
        )

        # ------------------------------------------------------------ #
        # 4. Compute data completeness score                            #
        # ------------------------------------------------------------ #
        total_cols = len(unified.columns)

        non_null_expr = sum(
            F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
            for c in unified.columns
        )

        unified = unified.withColumn(
            "data_completeness_score",
            F.round(non_null_expr / F.lit(total_cols), 4),
        )

        # ------------------------------------------------------------ #
        # 5. Persist as Parquet (optional)                              #
        # ------------------------------------------------------------ #
        if output_path is not None:
            (
                unified.write
                .mode("overwrite")
                .partitionBy("patient_id")
                .parquet(output_path)
            )

        return unified
