"""
Training Pipeline for Healthcare Intelligence System
=======================================================
Orchestrates end-to-end model training across four modalities
(structured, NLP, lab-anomaly, imaging) and a meta-learning ensemble.
"""

import os
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
import joblib

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="data/outputs/training.log")


class TrainingPipeline:
    """End-to-end training orchestration for the Healthcare Intelligence System."""

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.spark = None

        self.raw_data_path = self.config["paths"]["raw_data"]
        self.models_path = self.config["paths"]["models"]
        self.outputs_path = self.config["paths"]["outputs"]

        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.outputs_path, exist_ok=True)

    def run(self):
        pipeline_start = time.time()
        print("=" * 70)
        print("  Healthcare Intelligence System - Training Pipeline")
        print("=" * 70)

        # ---- Step 1: Spark session ----
        self._step("1/14", "Creating Spark session")
        from src.data_processing.spark_session import create_spark_session
        self.spark = create_spark_session(self.config.get("spark", {}))
        print(f"  Spark version: {self.spark.version}")

        # ---- Step 2: Load data ----
        self._step("2/14", "Loading datasets")
        from src.data_processing.data_loader import HealthcareDataLoader
        loader = HealthcareDataLoader

        patients_df = loader.load_patients(self.spark, os.path.join(self.raw_data_path, "patients.csv"))
        symptoms_df = loader.load_symptoms(self.spark, os.path.join(self.raw_data_path, "symptoms.csv"))
        lab_df = loader.load_lab_results(self.spark, os.path.join(self.raw_data_path, "lab_results.csv"))
        notes_df = loader.load_clinical_notes(self.spark, os.path.join(self.raw_data_path, "clinical_notes.csv"))
        images_df = loader.load_image_metadata(self.spark, os.path.join(self.raw_data_path, "image_metadata.csv"))
        labels_df = loader.load_ground_truth(self.spark, os.path.join(self.raw_data_path, "ground_truth.csv"))

        p_count = patients_df.count()
        s_count = symptoms_df.count()
        l_count = lab_df.count()
        n_count = notes_df.count()
        i_count = images_df.count()
        gt_count = labels_df.count()
        print(f"  Patients: {p_count}  Symptoms: {s_count}  Labs: {l_count}")
        print(f"  Notes: {n_count}  Images: {i_count}  Labels: {gt_count}")

        # ---- Step 3: Preprocess ----
        self._step("3/14", "Preprocessing data")
        from pyspark.sql import functions as F

        patients_clean = patients_df.na.fill({"smoking_status": "Unknown"})
        symptoms_clean = symptoms_df.na.fill(0)
        # Add symptom_count
        symptom_cols = [c for c in symptoms_clean.columns
                        if c not in ("patient_id", "timestamp", "primary_diagnosis")]
        symptoms_clean = symptoms_clean.withColumn(
            "symptom_count",
            sum(F.col(c) for c in symptom_cols)
        )
        print(f"  Preprocessing complete. Symptom columns: {len(symptom_cols)}")

        # ---- Step 4: Integrate data ----
        self._step("4/14", "Integrating data sources")
        # Join patients + symptoms + ground_truth
        integrated_df = patients_clean.join(symptoms_clean, on="patient_id", how="inner")
        integrated_df = integrated_df.join(labels_df, on="patient_id", how="inner")
        int_count = integrated_df.count()
        int_cols = len(integrated_df.columns)
        print(f"  Integrated dataset: {int_count} rows, {int_cols} columns")

        # ---- Step 5: Feature engineering ----
        self._step("5/14", "Engineering features")
        from src.feature_engineering.structured_features import StructuredFeatureEngineer
        feat_eng = StructuredFeatureEngineer()

        featured_df = integrated_df

        # Vital risk scores
        featured_df = feat_eng.compute_vital_risk_scores(featured_df)
        print("  Computed MEWS and qSOFA scores")

        # Comorbidity index
        featured_df = feat_eng.compute_comorbidity_index(featured_df)
        featured_df = feat_eng.compute_interaction_features(featured_df)
        print("  Computed comorbidity index and interaction features")

        # Symptom clusters
        featured_df = feat_eng.compute_symptom_clusters(featured_df)
        print("  Computed symptom clusters")

        # Build feature vector
        label_col = "final_diagnosis"
        featured_df = feat_eng.build_feature_vector(featured_df, output_col="features")
        print("  Feature vector assembled")

        # ---- Step 6: Stratified split ----
        self._step("6/14", "Splitting data 80/20 stratified")
        from functools import reduce

        train_frames, test_frames = [], []
        distinct_labels = [
            row[label_col]
            for row in featured_df.select(label_col).distinct().collect()
        ]
        for lbl in distinct_labels:
            subset = featured_df.filter(F.col(label_col) == lbl)
            tr, te = subset.randomSplit([0.8, 0.2], seed=42)
            train_frames.append(tr)
            test_frames.append(te)

        train_df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), train_frames)
        test_df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), test_frames)
        train_df.cache()
        test_df.cache()
        train_count = train_df.count()
        test_count = test_df.count()
        print(f"  Train: {train_count}   Test: {test_count}")

        # ---- Step 7: Train SymptomClassifier (PySpark MLlib) ----
        structured_metrics = {}
        structured_preds = None
        self._step("7/14", "Training SymptomClassifier (PySpark MLlib Random Forest)")
        try:
            from src.models.symptom_classifier import SymptomClassifier
            sc_config = self.config.get("models", {}).get("symptom_classifier", {})
            symptom_clf = SymptomClassifier(config=sc_config)

            feature_cols = self._get_numeric_feature_cols(train_df)
            print(f"  Using {len(feature_cols)} numeric features")

            symptom_clf.train(train_df, feature_cols, label_col)
            structured_preds = symptom_clf.predict(test_df)
            structured_metrics = symptom_clf.evaluate(structured_preds)
            print(f"  SymptomClassifier Accuracy: {structured_metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  SymptomClassifier F1: {structured_metrics.get('f1', 'N/A'):.4f}")

            symptom_clf.save_model(os.path.join(self.models_path, "symptom_classifier"))
            print("  Model saved to data/models/symptom_classifier")
        except Exception as e:
            print(f"  ERROR: SymptomClassifier training failed: {e}")
            logger.error("SymptomClassifier failed: %s", e, exc_info=True)

        # ---- Step 8: Train NLP Model ----
        nlp_preds = None
        nlp_metrics = {}
        self._step("8/14", "Training Clinical NLP Model (TF-IDF + LogisticRegression)")
        try:
            nlp_preds, nlp_metrics = self._train_nlp_model(notes_df, labels_df, label_col)
            if nlp_metrics:
                print(f"  NLP Model Accuracy: {nlp_metrics.get('accuracy', 'N/A'):.4f}")
                print(f"  NLP Model F1: {nlp_metrics.get('f1', 'N/A'):.4f}")
        except Exception as e:
            print(f"  ERROR: NLP training failed: {e}")
            logger.error("NLP failed: %s", e, exc_info=True)

        # ---- Step 9: Train Lab Anomaly Detector ----
        lab_preds = None
        lab_metrics = {}
        self._step("9/14", "Training Lab Anomaly Detector (RandomForest on lab features)")
        try:
            lab_preds, lab_metrics = self._train_lab_model(lab_df, labels_df, label_col)
            if lab_metrics:
                print(f"  Lab Model Accuracy: {lab_metrics.get('accuracy', 'N/A'):.4f}")
                print(f"  Lab Model F1: {lab_metrics.get('f1', 'N/A'):.4f}")
        except Exception as e:
            print(f"  ERROR: Lab model training failed: {e}")
            logger.error("Lab model failed: %s", e, exc_info=True)

        # ---- Step 10: Image Classifier ----
        image_preds = None
        image_metrics = {}
        self._step("10/14", "Training Medical Image Classifier (DenseNet-121)")
        try:
            image_preds, image_metrics = self._train_image_model(images_df, labels_df, label_col)
            if image_metrics:
                print(f"  Image Model Accuracy: {image_metrics.get('accuracy', 'N/A'):.4f}")
        except Exception as e:
            print(f"  ERROR: Image model training failed: {e}")
            logger.error("Image model failed: %s", e, exc_info=True)

        # ---- Step 11: Collect held-out predictions ----
        self._step("11/14", "Generating held-out predictions for ensemble")
        held_out = self._collect_held_out_predictions(
            test_df, structured_preds, nlp_preds, lab_preds, image_preds, label_col
        )
        if held_out is not None:
            print(f"  Meta-feature matrix shape: {held_out['X_meta'].shape}")

        # ---- Step 12: Train Ensemble ----
        ensemble_preds = None
        ensemble_metrics = {}
        self._step("12/14", "Training Ensemble Meta-Learner")
        try:
            if held_out is not None:
                ensemble_preds, ensemble_metrics = self._train_ensemble(held_out)
                if ensemble_metrics:
                    print(f"  Ensemble Accuracy: {ensemble_metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"  Ensemble F1: {ensemble_metrics.get('f1', 'N/A'):.4f}")
        except Exception as e:
            print(f"  ERROR: Ensemble training failed: {e}")
            logger.error("Ensemble failed: %s", e, exc_info=True)

        # ---- Step 13: Evaluate ----
        self._step("13/14", "Evaluating full pipeline")
        all_metrics = {
            "structured": structured_metrics,
            "nlp": nlp_metrics,
            "lab": lab_metrics,
            "imaging": image_metrics,
            "ensemble": ensemble_metrics,
        }
        self._print_comparison_table(all_metrics)

        # ---- Step 14: Save metrics ----
        self._step("14/14", "Saving metrics and predictions")
        self._save_metrics(all_metrics, os.path.join(self.outputs_path, "evaluation_metrics.json"))

        # Save predictions for dashboard
        if ensemble_preds is not None:
            self._save_predictions(ensemble_preds, held_out,
                                   os.path.join(self.outputs_path, "predictions.csv"))

        elapsed = time.time() - pipeline_start
        print("\n" + "=" * 70)
        print(f"  Pipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print("=" * 70)

        # Stop Spark
        self.spark.stop()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _step(self, step_id, description):
        print(f"\n{'-'*50}")
        print(f"  [{step_id}] {description}")
        print(f"{'-'*50}")

    def _get_numeric_feature_cols(self, df):
        exclude = {"patient_id", "timestamp", "primary_diagnosis",
                    "final_diagnosis", "risk_level", "requires_icu",
                    "features", "raw_features", "scaled_features",
                    "prediction", "rawPrediction", "probability",
                    "label_index", "label"}
        numeric_types = {"int", "bigint", "double", "float", "decimal"}
        return [
            f.name for f in df.schema.fields
            if f.dataType.simpleString().split("(")[0] in numeric_types
            and f.name not in exclude
        ]

    # -- NLP --
    def _train_nlp_model(self, notes_df, labels_df, label_col):
        """Train TF-IDF + LogisticRegression on clinical notes."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        # Join notes with labels
        notes_with_labels = notes_df.join(labels_df, on="patient_id", how="inner")
        notes_pd = notes_with_labels.select("patient_id", "note_text", label_col).toPandas()
        notes_pd = notes_pd.dropna(subset=["note_text"])

        # Aggregate notes per patient (concatenate multiple notes)
        grouped = notes_pd.groupby("patient_id").agg({
            "note_text": " ".join,
            label_col: "first"
        }).reset_index()

        print(f"  NLP training samples: {len(grouped)}")

        le = LabelEncoder()
        y = le.fit_transform(grouped[label_col])
        X_train, X_test, y_train, y_test = train_test_split(
            grouped["note_text"], y, test_size=0.2, random_state=42, stratify=y
        )

        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
        clf.fit(X_train_tfidf, y_train)

        y_pred = clf.predict(X_test_tfidf)
        y_prob = clf.predict_proba(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # Save model
        joblib.dump({"tfidf": tfidf, "clf": clf, "le": le},
                    os.path.join(self.models_path, "nlp_model.pkl"))
        print("  NLP model saved to data/models/nlp_model.pkl")

        metrics = {"accuracy": acc, "f1": f1}
        preds = {"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}
        return preds, metrics

    # -- Lab --
    def _train_lab_model(self, lab_df, labels_df, label_col):
        """Train RandomForest on pivoted lab features."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        # Pivot lab results: one column per test
        lab_pd = lab_df.toPandas()
        lab_pivot = lab_pd.pivot_table(
            index="patient_id", columns="test_name",
            values="value", aggfunc="mean"
        ).reset_index()
        lab_pivot.columns = [str(c) for c in lab_pivot.columns]

        # Join with labels
        labels_pd = labels_df.select("patient_id", label_col).toPandas()
        merged = lab_pivot.merge(labels_pd, on="patient_id", how="inner")
        merged = merged.dropna()

        feature_cols = [c for c in merged.columns if c not in ("patient_id", label_col)]
        print(f"  Lab features: {len(feature_cols)} tests, {len(merged)} patients")

        le = LabelEncoder()
        y = le.fit_transform(merged[label_col])
        X = merged[feature_cols].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        joblib.dump({"clf": clf, "le": le, "feature_cols": feature_cols},
                    os.path.join(self.models_path, "lab_model.pkl"))
        print("  Lab model saved to data/models/lab_model.pkl")

        metrics = {"accuracy": acc, "f1": f1}
        preds = {"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}
        return preds, metrics

    # -- Image --
    def _train_image_model(self, images_df, labels_df, label_col):
        """Train DenseNet-121 on chest X-ray images."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset
            from torchvision import models, transforms
            from PIL import Image
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score, f1_score
        except ImportError:
            print("  PyTorch/torchvision not available; skipping image model")
            return None, {}

        # Join image metadata with labels
        img_with_labels = images_df.join(labels_df, on="patient_id", how="inner")
        img_pd = img_with_labels.toPandas()
        img_pd = img_pd.dropna(subset=["image_path", label_col])

        # Fix image paths to be relative to data/raw/
        img_pd["full_path"] = img_pd["image_path"].apply(
            lambda p: os.path.join(self.raw_data_path, p)
        )
        # Filter to existing files
        img_pd = img_pd[img_pd["full_path"].apply(os.path.exists)]
        print(f"  Image samples with valid paths: {len(img_pd)}")

        if len(img_pd) < 10:
            print("  Not enough images; skipping")
            return None, {}

        le = LabelEncoder()
        img_pd["label_idx"] = le.fit_transform(img_pd[label_col])
        num_classes = len(le.classes_)

        # Split
        from sklearn.model_selection import train_test_split
        train_img, test_img = train_test_split(
            img_pd, test_size=0.2, random_state=42, stratify=img_pd["label_idx"]
        )

        # Dataset
        class XrayDataset(Dataset):
            def __init__(self, df, transform):
                self.paths = df["full_path"].tolist()
                self.labels = df["label_idx"].tolist()
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert("RGB")
                img = self.transform(img)
                return img, self.labels[idx]

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_loader = DataLoader(XrayDataset(train_img, train_transform),
                                  batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(XrayDataset(test_img, val_transform),
                                 batch_size=32, shuffle=False, num_workers=0)

        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        # Train for limited epochs
        epochs = min(self.config.get("models", {}).get("image_classifier", {}).get("epochs", 5), 5)
        best_val_acc = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = torch.as_tensor(labels, dtype=torch.long).to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = correct / total

            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = torch.as_tensor(labels, dtype=torch.long).to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = val_correct / val_total
            scheduler.step(val_loss)
            print(f"  Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                           os.path.join(self.models_path, "image_classifier.pth"))

        # Final evaluation
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels)
                all_probs.extend(probs)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        joblib.dump({"le": le, "num_classes": num_classes},
                    os.path.join(self.models_path, "image_classifier_meta.pkl"))
        print(f"  Image model saved. Best val acc: {best_val_acc:.4f}")

        metrics = {"accuracy": acc, "f1": f1}
        preds = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
        return preds, metrics

    # -- Held-out predictions --
    def _collect_held_out_predictions(self, test_df, structured, nlp, lab, image, label_col):
        """Build meta-feature matrix from all modality predictions."""
        from sklearn.preprocessing import LabelEncoder

        n_classes = self.config.get("models", {}).get("symptom_classifier", {}).get("num_classes", 5)

        # Get true labels from structured predictions (PySpark) or test_df
        if structured is not None:
            try:
                preds_pd = structured.select("label_index", "probability").toPandas()
                y_true = preds_pd["label_index"].values.astype(int)
                struct_probs = np.array([row.toArray() for row in preds_pd["probability"]])
                n_samples = len(y_true)
            except Exception as e:
                print(f"  Warning: Could not extract structured preds: {e}")
                structured = None

        if structured is None:
            y_true_pd = test_df.select(label_col).toPandas()
            le = LabelEncoder()
            y_true = le.fit_transform(y_true_pd[label_col])
            n_samples = len(y_true)
            struct_probs = np.full((n_samples, n_classes), 1.0 / n_classes)

        meta_blocks = [struct_probs if structured is not None
                       else np.full((n_samples, n_classes), 1.0 / n_classes)]

        for preds_dict in [nlp, lab, image]:
            if preds_dict is not None and preds_dict.get("y_prob") is not None:
                prob = np.asarray(preds_dict["y_prob"])
                if prob.shape[1] == n_classes:
                    # Align size by padding/truncating
                    if len(prob) == n_samples:
                        meta_blocks.append(prob)
                    else:
                        meta_blocks.append(np.full((n_samples, n_classes), 1.0 / n_classes))
                else:
                    meta_blocks.append(np.full((n_samples, n_classes), 1.0 / n_classes))
            else:
                meta_blocks.append(np.full((n_samples, n_classes), 1.0 / n_classes))

        X_meta = np.hstack(meta_blocks)
        return {"X_meta": X_meta, "y_true": y_true}

    # -- Ensemble --
    def _train_ensemble(self, held_out):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, f1_score

        X_meta = held_out["X_meta"]
        y_true = held_out["y_true"]

        meta_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        cv_scores = cross_val_score(meta_clf, X_meta, y_true, cv=5, scoring="f1_macro")
        print(f"  Ensemble CV F1 (macro): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        meta_clf.fit(X_meta, y_true)
        y_pred = meta_clf.predict(X_meta)
        y_prob = meta_clf.predict_proba(X_meta)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        joblib.dump(meta_clf, os.path.join(self.models_path, "ensemble.pkl"))
        print("  Ensemble meta-learner saved to data/models/ensemble.pkl")

        metrics = {"accuracy": acc, "f1": f1}
        preds = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
        return preds, metrics

    # -- Print results --
    def _print_comparison_table(self, all_metrics):
        print("\n" + "=" * 60)
        print("  MODEL COMPARISON")
        print("=" * 60)
        print(f"  {'Model':<25} {'Accuracy':>10} {'F1 (macro)':>12}")
        print(f"  {'-'*25} {'-'*10} {'-'*12}")
        for name, metrics in all_metrics.items():
            if metrics:
                acc = metrics.get("accuracy", 0)
                f1 = metrics.get("f1", 0)
                print(f"  {name:<25} {acc:>10.4f} {f1:>12.4f}")
            else:
                print(f"  {name:<25} {'N/A':>10} {'N/A':>12}")
        print("=" * 60)

    # -- Save metrics --
    def _save_metrics(self, metrics, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=_convert)
        print(f"  Metrics saved to {path}")

    # -- Save predictions --
    def _save_predictions(self, ensemble_preds, held_out, path):
        try:
            df = pd.DataFrame({
                "y_true": held_out["y_true"],
                "y_pred": ensemble_preds["y_pred"],
            })
            df.to_csv(path, index=False)
            print(f"  Predictions saved to {path}")
        except Exception as e:
            print(f"  Warning: Could not save predictions: {e}")


# -------------------------------------------------------------------- #
#  CLI entry point                                                       #
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
