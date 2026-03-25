"""
Lab Anomaly Detector combining Isolation Forest, rule-based risk scoring, and KMeans clustering.
Part of the Healthcare Intelligence System models layer.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator


class LabAnomalyDetector:
    """Detect anomalous lab results using ML and rule-based approaches."""

    def __init__(self, contamination=0.1, n_estimators=200):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.isolation_forest = None
        self.scaler = None
        self.kmeans_model = None
        self.feature_cols = None

    def train_isolation_forest(self, features_df):
        """Fit Isolation Forest on lab features (collects PySpark DataFrame to pandas)."""
        if hasattr(features_df, "toPandas"):
            pdf = features_df.toPandas()
        else:
            pdf = features_df

        # Select only numeric columns
        numeric_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = numeric_cols

        X = pdf[numeric_cols].fillna(0).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)

        return self.isolation_forest

    def detect_anomalies(self, features_df):
        """Predict anomaly scores, return DataFrame with anomaly_score and is_anomaly columns."""
        if self.isolation_forest is None:
            raise RuntimeError("Isolation Forest has not been trained. Call train_isolation_forest() first.")

        if hasattr(features_df, "toPandas"):
            pdf = features_df.toPandas()
        else:
            pdf = features_df.copy()

        X = pdf[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        # score_samples returns negative scores; more negative = more anomalous
        raw_scores = self.isolation_forest.score_samples(X_scaled)
        # Normalize to 0-1 range where 1 = most anomalous
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score - min_score > 0:
            anomaly_scores = 1.0 - (raw_scores - min_score) / (max_score - min_score)
        else:
            anomaly_scores = np.zeros_like(raw_scores)

        predictions = self.isolation_forest.predict(X_scaled)

        pdf["anomaly_score"] = anomaly_scores
        pdf["is_anomaly"] = (predictions == -1).astype(int)

        return pdf

    def compute_rule_based_risk(self, lab_df, reference_ranges):
        """
        Count critically abnormal values and return a risk score between 0 and 1.

        Args:
            lab_df: pandas DataFrame with lab test columns.
            reference_ranges: dict mapping column name to (low, high) tuple of normal ranges.
                Example: {"glucose": (70, 100), "hemoglobin": (12.0, 17.5)}

        Returns:
            pandas Series of risk scores (0 to 1) per row.
        """
        if hasattr(lab_df, "toPandas"):
            pdf = lab_df.toPandas()
        else:
            pdf = lab_df

        total_tests = len(reference_ranges)
        if total_tests == 0:
            return pd.Series(np.zeros(len(pdf)), index=pdf.index)

        abnormal_counts = pd.Series(np.zeros(len(pdf)), index=pdf.index)

        for col, (low, high) in reference_ranges.items():
            if col not in pdf.columns:
                continue
            values = pdf[col].fillna((low + high) / 2)
            # Critical if value is more than 50% outside normal range
            range_span = high - low
            critically_low = low - 0.5 * range_span
            critically_high = high + 0.5 * range_span
            is_critical = (values < critically_low) | (values > critically_high)
            is_abnormal = (values < low) | (values > high)
            # Critical counts as 1.0, abnormal as 0.5
            abnormal_counts += is_critical.astype(float) * 1.0
            abnormal_counts += (~is_critical & is_abnormal).astype(float) * 0.5

        risk_scores = (abnormal_counts / total_tests).clip(0.0, 1.0)
        return risk_scores

    def compute_combined_risk(self, ml_score, rule_score, ml_weight=0.6, rule_weight=0.4):
        """Weighted combination of ML anomaly score and rule-based risk."""
        combined = ml_weight * np.asarray(ml_score) + rule_weight * np.asarray(rule_score)
        return np.clip(combined, 0.0, 1.0)

    def train_kmeans_clusters(self, spark, features_df, k=4):
        """PySpark MLlib KMeans clustering on lab features, return cluster assignments."""
        if hasattr(features_df, "toPandas"):
            # Already a Spark DataFrame
            sdf = features_df
        else:
            # Convert pandas to Spark
            sdf = spark.createDataFrame(features_df)

        numeric_cols = [
            f.name for f in sdf.schema.fields
            if str(f.dataType) in ("DoubleType()", "FloatType()", "IntegerType()", "LongType()")
        ]

        assembler = VectorAssembler(
            inputCols=numeric_cols,
            outputCol="features",
            handleInvalid="keep"
        )
        assembled_df = assembler.transform(sdf)

        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=k,
            seed=42,
            maxIter=50
        )
        self.kmeans_model = kmeans.fit(assembled_df)

        clustered_df = self.kmeans_model.transform(assembled_df)

        # Evaluate clustering
        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="cluster"
        )
        silhouette = evaluator.evaluate(clustered_df)
        print(f"KMeans Silhouette Score: {silhouette:.4f}")

        return clustered_df

    def save_model(self, path):
        """Save Isolation Forest with joblib and KMeans with PySpark."""
        os.makedirs(path, exist_ok=True)

        if self.isolation_forest is not None:
            joblib.dump(
                {
                    "isolation_forest": self.isolation_forest,
                    "scaler": self.scaler,
                    "feature_cols": self.feature_cols,
                },
                os.path.join(path, "isolation_forest.joblib")
            )

        if self.kmeans_model is not None:
            kmeans_path = os.path.join(path, "kmeans_model")
            self.kmeans_model.write().overwrite().save(kmeans_path)

    def load_model(self, path):
        """Load Isolation Forest with joblib and KMeans with PySpark."""
        iforest_path = os.path.join(path, "isolation_forest.joblib")
        if os.path.exists(iforest_path):
            data = joblib.load(iforest_path)
            self.isolation_forest = data["isolation_forest"]
            self.scaler = data["scaler"]
            self.feature_cols = data["feature_cols"]

        kmeans_path = os.path.join(path, "kmeans_model")
        if os.path.exists(kmeans_path):
            self.kmeans_model = KMeansModel.load(kmeans_path)
