"""
Lab-Result Feature Engineering for Healthcare Intelligence System.

Computes deviation scores, abnormality counts, critical flags, organ panel
aggregate scores, and temporal trends from laboratory test results using
PySpark.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType


# Default clinical reference ranges: {test_name: (low, high)}
DEFAULT_REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    # Complete blood count
    "hemoglobin": (12.0, 17.5),
    "white_blood_cells": (4.5, 11.0),
    "platelets": (150.0, 400.0),
    "hematocrit": (36.0, 50.0),
    # Metabolic panel
    "glucose_fasting": (70.0, 100.0),
    "hba1c": (4.0, 5.6),
    "sodium": (136.0, 145.0),
    "potassium": (3.5, 5.0),
    "calcium": (8.5, 10.5),
    "chloride": (98.0, 106.0),
    "bicarbonate": (22.0, 29.0),
    # Renal panel
    "creatinine": (0.6, 1.2),
    "blood_urea_nitrogen": (7.0, 20.0),
    # Liver panel
    "alt": (7.0, 56.0),
    "ast": (10.0, 40.0),
    "total_bilirubin": (0.1, 1.2),
    "albumin": (3.5, 5.5),
    "alkaline_phosphatase": (44.0, 147.0),
    # Inflammatory markers
    "crp": (0.0, 3.0),
    "esr": (0.0, 20.0),
    # Lipid panel
    "total_cholesterol": (0.0, 200.0),
    "ldl": (0.0, 100.0),
    "hdl": (40.0, 60.0),
    "triglycerides": (0.0, 150.0),
    # Thyroid
    "tsh": (0.4, 4.0),
}

# Critical thresholds: {test_name: (crit_low, crit_high)}
CRITICAL_THRESHOLDS: dict[str, tuple[float, float]] = {
    "hemoglobin": (7.0, 20.0),
    "white_blood_cells": (2.0, 30.0),
    "platelets": (50.0, 1000.0),
    "glucose_fasting": (40.0, 400.0),
    "sodium": (120.0, 160.0),
    "potassium": (2.5, 6.5),
    "creatinine": (0.0, 10.0),
    "blood_urea_nitrogen": (0.0, 100.0),
    "alt": (0.0, 1000.0),
    "ast": (0.0, 1000.0),
    "total_bilirubin": (0.0, 10.0),
    "crp": (0.0, 100.0),
    "hba1c": (0.0, 14.0),
}


class LabFeatureEngineer:
    """Engineers features from laboratory test results using PySpark."""

    def __init__(
        self,
        reference_ranges: dict[str, tuple[float, float]] | None = None,
        critical_thresholds: dict[str, tuple[float, float]] | None = None,
    ):
        self.reference_ranges = reference_ranges or DEFAULT_REFERENCE_RANGES
        self.critical_thresholds = critical_thresholds or CRITICAL_THRESHOLDS

    # ------------------------------------------------------------------ #
    #  Deviation Scores
    # ------------------------------------------------------------------ #
    def compute_deviation_scores(
        self,
        df: DataFrame,
        reference_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> DataFrame:
        """Normalised deviation from the midpoint of the normal range.

        For each lab test column present in *reference_ranges*, adds
        ``<test>_deviation`` = (value - midpoint) / half_range.

        A score of 0 means exactly at the midpoint; +/-1 means at the
        boundary; >1 or <-1 means outside normal limits.
        """
        ref = reference_ranges or self.reference_ranges
        existing_cols = set(df.columns)

        for test, (low, high) in ref.items():
            if test not in existing_cols:
                continue
            midpoint = (low + high) / 2.0
            half_range = (high - low) / 2.0
            if half_range == 0:
                half_range = 1.0  # avoid division by zero
            df = df.withColumn(
                f"{test}_deviation",
                ((F.col(test).cast(DoubleType()) - F.lit(midpoint)) / F.lit(half_range)),
            )
        return df

    # ------------------------------------------------------------------ #
    #  Abnormality Count
    # ------------------------------------------------------------------ #
    def compute_abnormality_count(
        self,
        df: DataFrame,
        reference_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> DataFrame:
        """Count how many lab values fall outside normal range per row.

        Adds column: abnormality_count.
        """
        ref = reference_ranges or self.reference_ranges
        existing_cols = set(df.columns)

        flags = []
        for test, (low, high) in ref.items():
            if test not in existing_cols:
                continue
            flag = F.when(
                (F.col(test).cast(DoubleType()) < low)
                | (F.col(test).cast(DoubleType()) > high),
                1,
            ).otherwise(0)
            flags.append(flag)

        if flags:
            df = df.withColumn("abnormality_count", sum(flags).cast(IntegerType()))
        else:
            df = df.withColumn("abnormality_count", F.lit(0).cast(IntegerType()))
        return df

    # ------------------------------------------------------------------ #
    #  Critical Flags
    # ------------------------------------------------------------------ #
    def compute_critical_flags(
        self,
        df: DataFrame,
        critical_thresholds: dict[str, tuple[float, float]] | None = None,
    ) -> DataFrame:
        """Binary flags for critically abnormal values.

        For each lab test in *critical_thresholds*, adds
        ``<test>_critical`` (0 or 1).  Also adds ``total_critical_flags``.
        """
        thresholds = critical_thresholds or self.critical_thresholds
        existing_cols = set(df.columns)
        flag_cols: list[str] = []

        for test, (crit_low, crit_high) in thresholds.items():
            if test not in existing_cols:
                continue
            col_name = f"{test}_critical"
            df = df.withColumn(
                col_name,
                F.when(
                    (F.col(test).cast(DoubleType()) < crit_low)
                    | (F.col(test).cast(DoubleType()) > crit_high),
                    1,
                ).otherwise(0).cast(IntegerType()),
            )
            flag_cols.append(col_name)

        if flag_cols:
            df = df.withColumn(
                "total_critical_flags",
                sum(F.col(c) for c in flag_cols).cast(IntegerType()),
            )
        else:
            df = df.withColumn("total_critical_flags", F.lit(0).cast(IntegerType()))
        return df

    # ------------------------------------------------------------------ #
    #  Organ-Panel Aggregate Scores
    # ------------------------------------------------------------------ #
    def compute_organ_panel_scores(self, df: DataFrame) -> DataFrame:
        """Aggregate deviation scores into organ-system panel scores.

        Panels:
          - liver_panel_score: alt, ast, total_bilirubin, albumin
          - renal_panel_score: creatinine, blood_urea_nitrogen
          - metabolic_panel_score: glucose_fasting, hba1c
          - inflammatory_score: white_blood_cells, crp

        Each panel score is the mean of the absolute deviation scores of
        its constituent tests.  Missing deviation columns are skipped.
        """
        panels = {
            "liver_panel_score": ["alt_deviation", "ast_deviation",
                                  "total_bilirubin_deviation", "albumin_deviation"],
            "renal_panel_score": ["creatinine_deviation", "blood_urea_nitrogen_deviation"],
            "metabolic_panel_score": ["glucose_fasting_deviation", "hba1c_deviation"],
            "inflammatory_score": ["white_blood_cells_deviation", "crp_deviation"],
        }
        existing_cols = set(df.columns)

        for panel_name, dev_cols in panels.items():
            present = [c for c in dev_cols if c in existing_cols]
            if present:
                abs_sum = sum(F.abs(F.coalesce(F.col(c), F.lit(0.0))) for c in present)
                df = df.withColumn(
                    panel_name,
                    (abs_sum / F.lit(float(len(present)))).cast(DoubleType()),
                )
            else:
                df = df.withColumn(panel_name, F.lit(0.0).cast(DoubleType()))
        return df

    # ------------------------------------------------------------------ #
    #  Temporal Trends (slope via window functions)
    # ------------------------------------------------------------------ #
    def compute_temporal_trends(
        self,
        df: DataFrame,
        patient_col: str = "patient_id",
        time_col: str = "lab_date",
        value_cols: list[str] | None = None,
    ) -> DataFrame:
        """Compute per-patient temporal slope for lab values using windows.

        For each value column, adds ``<col>_trend`` representing the slope
        (change per unit time ordinal) calculated via linear regression
        over the patient's time-ordered readings.

        Uses the formula:  slope = (N*sum(x*y) - sum(x)*sum(y)) /
                                   (N*sum(x^2) - (sum(x))^2)
        where x is a row-number ordinal within the patient window.
        """
        if value_cols is None:
            value_cols = [
                c for c in df.columns
                if c not in {patient_col, time_col, "encounter_id"}
                and not c.endswith("_deviation")
                and not c.endswith("_critical")
            ]

        existing_cols = set(df.columns)
        value_cols = [c for c in value_cols if c in existing_cols]

        if not value_cols:
            return df

        window = Window.partitionBy(patient_col).orderBy(time_col)
        patient_window = Window.partitionBy(patient_col)

        # Row number as the time ordinal
        df = df.withColumn("_row_num", F.row_number().over(window).cast(DoubleType()))
        df = df.withColumn("_n", F.count("*").over(patient_window).cast(DoubleType()))

        for col_name in value_cols:
            y = F.col(col_name).cast(DoubleType())
            x = F.col("_row_num")

            df = df.withColumn("_xy", x * y)
            df = df.withColumn("_x2", x * x)

            sum_x = F.sum(x).over(patient_window)
            sum_y = F.sum(y).over(patient_window)
            sum_xy = F.sum(F.col("_xy")).over(patient_window)
            sum_x2 = F.sum(F.col("_x2")).over(patient_window)
            n = F.col("_n")

            denominator = n * sum_x2 - sum_x * sum_x
            numerator = n * sum_xy - sum_x * sum_y

            slope = F.when(denominator != 0, numerator / denominator).otherwise(0.0)
            df = df.withColumn(f"{col_name}_trend", slope.cast(DoubleType()))

        # Clean up helper columns
        df = df.drop("_row_num", "_n", "_xy", "_x2")
        return df
