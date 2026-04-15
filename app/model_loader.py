"""
Model loading utilities for the Healthcare Intelligence System Streamlit dashboard.
Handles data loading, prediction loading, metrics loading, and demo score simulation.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_cached_data(data_dir: str) -> dict:
    """
    Load all CSVs from data/raw/ into pandas DataFrames.
    Returns a dict keyed by dataset name.
    """
    datasets = {}
    csv_files = {
        "patients": "patients.csv",
        "symptoms": "symptoms.csv",
        "lab_results": "lab_results.csv",
        "clinical_notes": "clinical_notes.csv",
        "image_metadata": "image_metadata.csv",
        "ground_truth": "ground_truth.csv",
    }

    for name, filename in csv_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                datasets[name] = pd.read_csv(filepath)
            except Exception as e:
                datasets[name] = pd.DataFrame()
                st.warning(f"Could not load {filename}: {e}")
        else:
            datasets[name] = pd.DataFrame()

    return datasets


@st.cache_data
def load_predictions(output_dir: str) -> pd.DataFrame | None:
    """Load prediction results from data/outputs/ if available."""
    pred_path = os.path.join(output_dir, "predictions.csv")
    if os.path.exists(pred_path):
        try:
            return pd.read_csv(pred_path)
        except Exception:
            return None
    return None


@st.cache_data
def load_metrics(output_dir: str) -> dict | None:
    """Load evaluation metrics JSON from data/outputs/ if available."""
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def simulate_risk_scores(
    patient_id: str,
    patients_df: pd.DataFrame,
    symptoms_df: pd.DataFrame,
    lab_df: pd.DataFrame,
) -> dict:
    """
    Generate plausible risk scores for demo when no trained models exist.
    Uses simple heuristics based on vital signs, symptom count, and lab results.
    Returns dict with overall score, modality scores, contributing factors, and risk level.
    """
    np.random.seed(hash(patient_id) % (2**31))

    # ----- Structured data risk (vitals + demographics) -----
    structured_score = 30.0
    factors = []

    patient_row = patients_df[patients_df["patient_id"] == patient_id]
    if not patient_row.empty:
        p = patient_row.iloc[0]

        # Age contribution
        age = p.get("age", 50)
        if age > 65:
            structured_score += 15
            factors.append(f"Advanced age ({age})")
        elif age > 50:
            structured_score += 8

        # BMI
        bmi = p.get("bmi", 25)
        if bmi > 30:
            structured_score += 8
            factors.append(f"Elevated BMI ({bmi:.1f})")
        elif bmi < 18.5:
            structured_score += 6
            factors.append(f"Low BMI ({bmi:.1f})")

        # Comorbidities
        for cond in ["diabetes", "hypertension", "heart_disease_history",
                      "liver_disease_history", "lung_disease_history"]:
            if p.get(cond, 0) == 1:
                structured_score += 5
                factors.append(f"History: {cond.replace('_', ' ').title()}")

        # Vital signs
        hr = p.get("heart_rate", 80)
        if hr > 100:
            structured_score += 8
            factors.append(f"Tachycardia (HR {hr})")
        elif hr < 60:
            structured_score += 6
            factors.append(f"Bradycardia (HR {hr})")

        sbp = p.get("systolic_bp", 120)
        if sbp > 140:
            structured_score += 7
            factors.append(f"Hypertensive (SBP {sbp})")
        elif sbp < 90:
            structured_score += 10
            factors.append(f"Hypotension (SBP {sbp})")

        temp = p.get("temperature", 37.0)
        if temp > 38.5:
            structured_score += 8
            factors.append(f"Fever ({temp:.1f} C)")
        elif temp < 36:
            structured_score += 6
            factors.append(f"Hypothermia ({temp:.1f} C)")

        spo2 = p.get("oxygen_saturation", 98)
        if spo2 < 90:
            structured_score += 15
            factors.append(f"Severe hypoxia (SpO2 {spo2}%)")
        elif spo2 < 95:
            structured_score += 8
            factors.append(f"Low SpO2 ({spo2}%)")

        rr = p.get("respiratory_rate", 16)
        if rr > 20:
            structured_score += 6
            factors.append(f"Tachypnea (RR {rr})")

        smoking = p.get("smoking_status", "Never")
        if smoking == "Current":
            structured_score += 6
            factors.append("Current smoker")
        elif smoking == "Former":
            structured_score += 3

    structured_score = min(structured_score, 100)

    # ----- NLP risk (symptom count) -----
    nlp_score = 25.0
    sym_row = symptoms_df[symptoms_df["patient_id"] == patient_id]
    symptom_cols = [c for c in symptoms_df.columns
                    if c not in ("patient_id", "timestamp", "primary_diagnosis")]
    if not sym_row.empty:
        active = int(sym_row.iloc[0][symptom_cols].sum())
        nlp_score += active * 4.5
        if active >= 8:
            factors.append(f"High symptom burden ({active} symptoms)")
        elif active >= 5:
            factors.append(f"Moderate symptom burden ({active} symptoms)")
    nlp_score = min(nlp_score, 100)

    # ----- Lab risk -----
    lab_score = 25.0
    lab_rows = lab_df[lab_df["patient_id"] == patient_id]
    if not lab_rows.empty:
        abnormal_count = 0
        for _, row in lab_rows.iterrows():
            val = row.get("value", 0)
            lo = row.get("reference_low", 0)
            hi = row.get("reference_high", 0)
            if val < lo or val > hi:
                abnormal_count += 1
        lab_score += abnormal_count * 6
        if abnormal_count >= 4:
            factors.append(f"Multiple abnormal labs ({abnormal_count})")
        elif abnormal_count >= 2:
            factors.append(f"Abnormal lab values ({abnormal_count})")
    lab_score = min(lab_score, 100)

    # ----- Imaging risk (simulated) -----
    imaging_score = 20.0 + np.random.uniform(0, 30)
    imaging_score = min(imaging_score, 100)

    # ----- Overall ensemble -----
    overall = (
        0.30 * structured_score
        + 0.25 * nlp_score
        + 0.25 * lab_score
        + 0.20 * imaging_score
    )
    overall = min(overall, 100)

    if overall >= 75:
        risk_level = "Critical"
    elif overall >= 50:
        risk_level = "High"
    elif overall >= 25:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "overall_score": round(overall, 1),
        "risk_level": risk_level,
        "modality_scores": {
            "Structured Data": round(structured_score, 1),
            "NLP (Clinical Notes)": round(nlp_score, 1),
            "Lab Results": round(lab_score, 1),
            "Medical Imaging": round(imaging_score, 1),
        },
        "contributing_factors": factors[:8],
    }
