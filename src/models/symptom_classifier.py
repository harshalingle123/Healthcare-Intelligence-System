"""
Symptom Classifier using PySpark MLlib RandomForestClassifier.
Part of the Healthcare Intelligence System models layer.
"""

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class SymptomClassifier:
    """PySpark-based symptom classification using Random Forest with hyperparameter tuning."""

    def __init__(self, config=None):
        self.config = config or {}
        self.max_depth = self.config.get("max_depth", 10)
        self.num_trees = self.config.get("num_trees", 150)
        self.pipeline = None
        self.cv_model = None
        self.best_model = None
        self.label_indexer = None

    def build_pipeline(self, feature_cols, label_col):
        """Create PySpark ML Pipeline: StringIndexer -> VectorAssembler -> StandardScaler -> RandomForest."""
        self.label_indexer = StringIndexer(
            inputCol=label_col,
            outputCol="label_index",
            handleInvalid="keep"
        )

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="raw_features",
            handleInvalid="keep"
        )

        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=False
        )

        rf = RandomForestClassifier(
            featuresCol="scaled_features",
            labelCol="label_index",
            maxDepth=self.max_depth,
            numTrees=self.num_trees,
            seed=42
        )

        self.pipeline = Pipeline(stages=[self.label_indexer, assembler, scaler, rf])
        return self.pipeline

    def train(self, train_df, feature_cols, label_col):
        """Build pipeline, run CrossValidator with ParamGrid, store best model."""
        self.build_pipeline(feature_cols, label_col)

        rf_stage = self.pipeline.getStages()[-1]

        param_grid = (
            ParamGridBuilder()
            .addGrid(rf_stage.maxDepth, [5, 8, 10])
            .addGrid(rf_stage.numTrees, [50, 100, 200])
            .build()
        )

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label_index",
            predictionCol="prediction",
            metricName="f1"
        )

        cross_validator = CrossValidator(
            estimator=self.pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=5,
            parallelism=2,
            seed=42
        )

        self.cv_model = cross_validator.fit(train_df)
        self.best_model = self.cv_model.bestModel
        return self.best_model

    def predict(self, test_df):
        """Transform test data with best model, return predictions DataFrame."""
        if self.best_model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        predictions = self.best_model.transform(test_df)
        return predictions

    def evaluate(self, predictions_df):
        """Compute accuracy, F1, weighted precision, weighted recall."""
        metrics = {}
        for metric_name in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label_index",
                predictionCol="prediction",
                metricName=metric_name
            )
            metrics[metric_name] = evaluator.evaluate(predictions_df)
        return metrics

    def get_feature_importance(self):
        """Extract feature importances from the Random Forest model."""
        if self.best_model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        rf_model = self.best_model.stages[-1]
        return rf_model.featureImportances.toArray()

    def save_model(self, path):
        """Save PySpark PipelineModel to disk."""
        if self.best_model is None:
            raise RuntimeError("No model to save. Call train() first.")
        self.best_model.write().overwrite().save(path)

    def load_model(self, path):
        """Load PySpark PipelineModel from disk."""
        self.best_model = PipelineModel.load(path)
        return self.best_model
