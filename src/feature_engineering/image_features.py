"""
Image Feature Engineering for Healthcare Intelligence System.

Extracts CNN features from medical images (e.g., chest X-rays) using a
pretrained DenseNet-121 backbone via PyTorch, then converts them to a
PySpark DataFrame for downstream joining.
"""

from __future__ import annotations

import os
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    StringType,
    StructField,
    StructType,
)

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class ImageFeatureEngineer:
    """Extracts deep features from medical images using DenseNet-121."""

    # ImageNet normalisation statistics
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        if not _TORCH_AVAILABLE:
            print(
                "[ImageFeatureEngineer] PyTorch / torchvision / Pillow not "
                "available. Image feature extraction will be disabled."
            )

    # ------------------------------------------------------------------ #
    #  Build Feature-Extractor Model
    # ------------------------------------------------------------------ #
    def build_model(self, pretrained: bool = True) -> Optional["nn.Module"]:
        """Load DenseNet-121 and remove the final classifier layer.

        Returns a feature extractor that outputs 1024-dim vectors.
        """
        if not _TORCH_AVAILABLE:
            print("[ImageFeatureEngineer] torch not available; cannot build model.")
            return None

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)

        # Remove the final classification layer; keep the feature backbone.
        # DenseNet-121's features are followed by a ReLU + AdaptiveAvgPool
        # producing a 1024-dim vector.
        feature_extractor = nn.Sequential(
            densenet.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        feature_extractor.eval()
        return feature_extractor

    # ------------------------------------------------------------------ #
    #  Extract CNN Features
    # ------------------------------------------------------------------ #
    def extract_cnn_features(
        self,
        image_paths: dict[str, str],
        model: Optional["nn.Module"] = None,
        batch_size: int = 16,
    ) -> dict[str, list[float]]:
        """Extract 1024-dim DenseNet-121 features from images.

        Parameters
        ----------
        image_paths : dict[str, str]
            Mapping of ``{patient_id: file_path}`` for each image.
        model : nn.Module | None
            A prebuilt feature extractor. If None, one is built with
            pretrained weights.
        batch_size : int
            Number of images per forward pass.

        Returns
        -------
        dict[str, list[float]]
            Mapping of ``{patient_id: 1024-dim feature vector}``.
        """
        if not _TORCH_AVAILABLE:
            print(
                "[ImageFeatureEngineer] torch not available; "
                "returning empty features."
            )
            return {}

        if model is None:
            model = self.build_model(pretrained=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])

        patient_ids = list(image_paths.keys())
        features_dict: dict[str, list[float]] = {}

        for start in range(0, len(patient_ids), batch_size):
            batch_ids = patient_ids[start : start + batch_size]
            tensors = []

            for pid in batch_ids:
                path = image_paths[pid]
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(preprocess(img))
                except Exception as exc:
                    print(f"[ImageFeatureEngineer] Failed to load {path}: {exc}")
                    # Use a zero tensor as placeholder
                    tensors.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(tensors).to(device)

            with torch.no_grad():
                batch_features = model(batch_tensor)  # (B, 1024)

            batch_features_np = batch_features.cpu().numpy()
            for idx, pid in enumerate(batch_ids):
                features_dict[pid] = batch_features_np[idx].tolist()

        return features_dict

    # ------------------------------------------------------------------ #
    #  Convert to Spark DataFrame
    # ------------------------------------------------------------------ #
    def create_image_feature_df(
        self,
        spark: SparkSession,
        features_dict: dict[str, list[float]],
        id_col: str = "patient_id",
    ) -> DataFrame:
        """Convert the features dictionary to a Spark DataFrame.

        Returns a DataFrame with columns ``[id_col, image_features]``
        where ``image_features`` is an array of 1024 floats.
        """
        schema = StructType([
            StructField(id_col, StringType(), False),
            StructField("image_features", ArrayType(FloatType()), False),
        ])

        rows = [
            (str(pid), [float(v) for v in vec])
            for pid, vec in features_dict.items()
        ]

        return spark.createDataFrame(rows, schema)
