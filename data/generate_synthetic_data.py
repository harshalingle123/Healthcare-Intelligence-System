"""
Synthetic Data Generation for Healthcare Intelligence System
=============================================================
Generates medically plausible, cross-correlated synthetic data for 10,000
patients across 5 diagnosis categories:
    Pneumonia, Heart_Disease, Diabetes_Complication, Liver_Disease, Healthy

Usage:
    python data/generate_synthetic_data.py
    (run from the Project root directory)

Output (all written to data/raw/):
    patients.csv, symptoms.csv, lab_results.csv, clinical_notes.csv,
    image_metadata.csv, ground_truth.csv, images/*.png
"""

import os
import sys
import random
import datetime
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from faker import Faker

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ---------------------------------------------------------------------------
# Paths (relative to THIS script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
IMAGE_DIR = os.path.join(RAW_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_PATIENTS = 10_000
PATIENTS_PER_CLASS = 2_000
DIAGNOSES = [
    "Pneumonia",
    "Heart_Disease",
    "Diabetes_Complication",
    "Liver_Disease",
    "Healthy",
]
LABS_PER_PATIENT = 5
IMAGE_PATIENTS = 1_000  # number of patients who get imaging
IMAGE_SIZE = 224

# ---------------------------------------------------------------------------
# Helper: clamp
# ---------------------------------------------------------------------------
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ===================================================================
# 1. PATIENTS.CSV
# ===================================================================
def generate_patients():
    print("[1/6] Generating patients.csv ...")
    rows = []
    pid = 1
    for dx in DIAGNOSES:
        for _ in range(PATIENTS_PER_CLASS):
            # --- Age ---
            if dx == "Pneumonia":
                age = int(clamp(np.random.normal(55, 18), 1, 95))
            elif dx == "Heart_Disease":
                age = int(clamp(np.random.normal(62, 12), 30, 95))
            elif dx == "Diabetes_Complication":
                age = int(clamp(np.random.normal(58, 14), 25, 90))
            elif dx == "Liver_Disease":
                age = int(clamp(np.random.normal(52, 14), 20, 85))
            else:  # Healthy
                age = int(clamp(np.random.normal(40, 15), 18, 80))

            sex = np.random.choice(["M", "F"], p=[0.55, 0.45] if dx == "Heart_Disease" else [0.50, 0.50])

            # --- BMI ---
            if dx == "Diabetes_Complication":
                bmi = round(clamp(np.random.normal(32, 5), 18, 55), 1)
            elif dx == "Heart_Disease":
                bmi = round(clamp(np.random.normal(29, 5), 17, 50), 1)
            elif dx == "Liver_Disease":
                bmi = round(clamp(np.random.normal(27, 6), 16, 50), 1)
            else:
                bmi = round(clamp(np.random.normal(25, 4), 16, 45), 1)

            smoking = np.random.choice(
                ["Never", "Former", "Current"],
                p={
                    "Pneumonia": [0.30, 0.30, 0.40],
                    "Heart_Disease": [0.20, 0.35, 0.45],
                    "Liver_Disease": [0.35, 0.30, 0.35],
                    "Diabetes_Complication": [0.40, 0.30, 0.30],
                    "Healthy": [0.65, 0.25, 0.10],
                }[dx],
            )

            diabetes = int(dx == "Diabetes_Complication" or np.random.rand() < (0.25 if dx in ("Heart_Disease", "Liver_Disease") else 0.08))
            hypertension = int(dx in ("Heart_Disease",) or np.random.rand() < (0.40 if dx == "Diabetes_Complication" else 0.15))
            heart_hx = int(dx == "Heart_Disease" or np.random.rand() < 0.08)
            liver_hx = int(dx == "Liver_Disease" or np.random.rand() < 0.05)
            lung_hx = int(dx == "Pneumonia" and np.random.rand() < 0.45 or np.random.rand() < 0.06)

            # --- Vitals ---
            if dx == "Pneumonia":
                sbp = int(clamp(np.random.normal(125, 15), 90, 180))
                dbp = int(clamp(np.random.normal(78, 10), 55, 110))
                hr = int(clamp(np.random.normal(100, 14), 60, 140))
                temp = round(clamp(np.random.normal(38.5, 0.8), 36.0, 41.0), 1)
                rr = int(clamp(np.random.normal(24, 5), 12, 40))
                spo2 = int(clamp(np.random.normal(93, 3), 80, 100))
            elif dx == "Heart_Disease":
                sbp = int(clamp(np.random.normal(148, 18), 100, 200))
                dbp = int(clamp(np.random.normal(92, 12), 60, 120))
                hr = int(clamp(np.random.normal(88, 18), 45, 150))
                temp = round(clamp(np.random.normal(36.8, 0.3), 36.0, 38.0), 1)
                rr = int(clamp(np.random.normal(20, 4), 12, 35))
                spo2 = int(clamp(np.random.normal(96, 2), 88, 100))
            elif dx == "Diabetes_Complication":
                sbp = int(clamp(np.random.normal(138, 16), 95, 190))
                dbp = int(clamp(np.random.normal(86, 10), 58, 115))
                hr = int(clamp(np.random.normal(84, 12), 55, 130))
                temp = round(clamp(np.random.normal(37.0, 0.4), 36.0, 38.5), 1)
                rr = int(clamp(np.random.normal(18, 3), 12, 30))
                spo2 = int(clamp(np.random.normal(97, 2), 90, 100))
            elif dx == "Liver_Disease":
                sbp = int(clamp(np.random.normal(118, 14), 85, 170))
                dbp = int(clamp(np.random.normal(72, 10), 50, 100))
                hr = int(clamp(np.random.normal(82, 14), 55, 130))
                temp = round(clamp(np.random.normal(37.2, 0.6), 36.0, 39.0), 1)
                rr = int(clamp(np.random.normal(18, 3), 12, 30))
                spo2 = int(clamp(np.random.normal(97, 2), 90, 100))
            else:  # Healthy
                sbp = int(clamp(np.random.normal(118, 10), 95, 140))
                dbp = int(clamp(np.random.normal(75, 8), 55, 90))
                hr = int(clamp(np.random.normal(72, 10), 50, 100))
                temp = round(clamp(np.random.normal(36.8, 0.3), 36.0, 37.5), 1)
                rr = int(clamp(np.random.normal(16, 2), 12, 22))
                spo2 = int(clamp(np.random.normal(98, 1), 95, 100))

            rows.append({
                "patient_id": f"P{pid:05d}",
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "smoking_status": smoking,
                "diabetes": diabetes,
                "hypertension": hypertension,
                "heart_disease_history": heart_hx,
                "liver_disease_history": liver_hx,
                "lung_disease_history": lung_hx,
                "systolic_bp": sbp,
                "diastolic_bp": dbp,
                "heart_rate": hr,
                "temperature": temp,
                "respiratory_rate": rr,
                "oxygen_saturation": spo2,
                "_diagnosis": dx,  # internal, will be dropped before saving
            })
            pid += 1

    df = pd.DataFrame(rows)
    df.drop(columns=["_diagnosis"]).to_csv(os.path.join(RAW_DIR, "patients.csv"), index=False)
    print(f"    -> {len(df)} patients written.")
    return df


# ===================================================================
# 2. SYMPTOMS.CSV
# ===================================================================
# Probabilities per symptom per diagnosis class
SYMPTOM_COLS = [
    "cough", "fever", "chest_pain", "shortness_of_breath", "fatigue",
    "nausea", "vomiting", "abdominal_pain", "headache", "dizziness",
    "weight_loss", "night_sweats", "joint_pain", "swelling", "jaundice",
    "dark_urine", "loss_of_appetite", "excessive_thirst", "frequent_urination",
    "blurred_vision", "palpitations", "wheezing", "sputum_production",
    "hemoptysis", "edema", "confusion", "muscle_weakness", "numbness",
    "skin_rash",
]

SYMPTOM_PROBS = {
    "Pneumonia": {
        "cough": 0.92, "fever": 0.85, "chest_pain": 0.45, "shortness_of_breath": 0.78,
        "fatigue": 0.65, "nausea": 0.20, "vomiting": 0.10, "abdominal_pain": 0.05,
        "headache": 0.30, "dizziness": 0.15, "weight_loss": 0.10, "night_sweats": 0.40,
        "joint_pain": 0.12, "swelling": 0.05, "jaundice": 0.02, "dark_urine": 0.05,
        "loss_of_appetite": 0.45, "excessive_thirst": 0.08, "frequent_urination": 0.05,
        "blurred_vision": 0.03, "palpitations": 0.10, "wheezing": 0.55,
        "sputum_production": 0.80, "hemoptysis": 0.15, "edema": 0.05,
        "confusion": 0.12, "muscle_weakness": 0.20, "numbness": 0.03, "skin_rash": 0.03,
    },
    "Heart_Disease": {
        "cough": 0.15, "fever": 0.08, "chest_pain": 0.82, "shortness_of_breath": 0.75,
        "fatigue": 0.72, "nausea": 0.25, "vomiting": 0.10, "abdominal_pain": 0.08,
        "headache": 0.20, "dizziness": 0.50, "weight_loss": 0.10, "night_sweats": 0.18,
        "joint_pain": 0.10, "swelling": 0.45, "jaundice": 0.03, "dark_urine": 0.04,
        "loss_of_appetite": 0.25, "excessive_thirst": 0.10, "frequent_urination": 0.10,
        "blurred_vision": 0.08, "palpitations": 0.78, "wheezing": 0.12,
        "sputum_production": 0.08, "hemoptysis": 0.02, "edema": 0.60,
        "confusion": 0.10, "muscle_weakness": 0.20, "numbness": 0.25, "skin_rash": 0.03,
    },
    "Diabetes_Complication": {
        "cough": 0.08, "fever": 0.10, "chest_pain": 0.10, "shortness_of_breath": 0.20,
        "fatigue": 0.70, "nausea": 0.35, "vomiting": 0.25, "abdominal_pain": 0.20,
        "headache": 0.25, "dizziness": 0.30, "weight_loss": 0.40, "night_sweats": 0.15,
        "joint_pain": 0.15, "swelling": 0.25, "jaundice": 0.03, "dark_urine": 0.10,
        "loss_of_appetite": 0.30, "excessive_thirst": 0.82, "frequent_urination": 0.80,
        "blurred_vision": 0.55, "palpitations": 0.12, "wheezing": 0.05,
        "sputum_production": 0.04, "hemoptysis": 0.01, "edema": 0.30,
        "confusion": 0.15, "muscle_weakness": 0.40, "numbness": 0.55, "skin_rash": 0.12,
    },
    "Liver_Disease": {
        "cough": 0.05, "fever": 0.20, "chest_pain": 0.05, "shortness_of_breath": 0.15,
        "fatigue": 0.75, "nausea": 0.65, "vomiting": 0.40, "abdominal_pain": 0.72,
        "headache": 0.15, "dizziness": 0.20, "weight_loss": 0.35, "night_sweats": 0.12,
        "joint_pain": 0.18, "swelling": 0.55, "jaundice": 0.70, "dark_urine": 0.60,
        "loss_of_appetite": 0.65, "excessive_thirst": 0.08, "frequent_urination": 0.06,
        "blurred_vision": 0.05, "palpitations": 0.08, "wheezing": 0.04,
        "sputum_production": 0.03, "hemoptysis": 0.02, "edema": 0.50,
        "confusion": 0.25, "muscle_weakness": 0.35, "numbness": 0.08, "skin_rash": 0.20,
    },
    "Healthy": {
        "cough": 0.08, "fever": 0.04, "chest_pain": 0.03, "shortness_of_breath": 0.05,
        "fatigue": 0.15, "nausea": 0.05, "vomiting": 0.02, "abdominal_pain": 0.04,
        "headache": 0.12, "dizziness": 0.05, "weight_loss": 0.03, "night_sweats": 0.03,
        "joint_pain": 0.06, "swelling": 0.02, "jaundice": 0.01, "dark_urine": 0.02,
        "loss_of_appetite": 0.05, "excessive_thirst": 0.04, "frequent_urination": 0.04,
        "blurred_vision": 0.03, "palpitations": 0.04, "wheezing": 0.03,
        "sputum_production": 0.02, "hemoptysis": 0.005, "edema": 0.02,
        "confusion": 0.02, "muscle_weakness": 0.05, "numbness": 0.04, "skin_rash": 0.04,
    },
}


def generate_symptoms(patients_df):
    print("[2/6] Generating symptoms.csv ...")
    rows = []
    base_date = datetime.date(2024, 1, 1)
    for _, pat in patients_df.iterrows():
        dx = pat["_diagnosis"]
        probs = SYMPTOM_PROBS[dx]
        ts = base_date + datetime.timedelta(days=int(np.random.randint(0, 365)))
        row = {
            "patient_id": pat["patient_id"],
            "timestamp": ts.isoformat(),
        }
        for sym in SYMPTOM_COLS:
            row[sym] = int(np.random.rand() < probs[sym])
        row["primary_diagnosis"] = dx
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "symptoms.csv"), index=False)
    print(f"    -> {len(df)} symptom records written.")
    return df


# ===================================================================
# 3. LAB_RESULTS.CSV
# ===================================================================
# Each test: (test_name, unit, ref_low, ref_high, {dx: (mean, std)})
LAB_SPECS = [
    ("hemoglobin", "g/dL", 12.0, 17.5, {
        "Pneumonia": (12.5, 1.8), "Heart_Disease": (13.0, 1.5),
        "Diabetes_Complication": (12.0, 2.0), "Liver_Disease": (10.5, 2.0), "Healthy": (14.5, 1.2),
    }),
    ("white_blood_cells", "10^3/uL", 4.5, 11.0, {
        "Pneumonia": (15.0, 4.0), "Heart_Disease": (8.5, 2.5),
        "Diabetes_Complication": (9.0, 2.5), "Liver_Disease": (7.0, 3.0), "Healthy": (7.0, 1.5),
    }),
    ("platelets", "10^3/uL", 150, 400, {
        "Pneumonia": (280, 80), "Heart_Disease": (250, 60),
        "Diabetes_Complication": (260, 70), "Liver_Disease": (120, 50), "Healthy": (260, 50),
    }),
    ("glucose_fasting", "mg/dL", 70, 100, {
        "Pneumonia": (105, 20), "Heart_Disease": (110, 25),
        "Diabetes_Complication": (210, 60), "Liver_Disease": (95, 20), "Healthy": (88, 8),
    }),
    ("hba1c", "%", 4.0, 5.6, {
        "Pneumonia": (5.8, 0.6), "Heart_Disease": (6.0, 0.8),
        "Diabetes_Complication": (9.2, 1.8), "Liver_Disease": (5.5, 0.6), "Healthy": (5.0, 0.3),
    }),
    ("creatinine", "mg/dL", 0.6, 1.2, {
        "Pneumonia": (1.0, 0.3), "Heart_Disease": (1.2, 0.4),
        "Diabetes_Complication": (1.8, 0.7), "Liver_Disease": (1.1, 0.4), "Healthy": (0.9, 0.15),
    }),
    ("blood_urea_nitrogen", "mg/dL", 7, 20, {
        "Pneumonia": (18, 6), "Heart_Disease": (22, 8),
        "Diabetes_Complication": (28, 10), "Liver_Disease": (16, 6), "Healthy": (13, 3),
    }),
    ("sodium", "mEq/L", 136, 145, {
        "Pneumonia": (137, 4), "Heart_Disease": (139, 3),
        "Diabetes_Complication": (135, 4), "Liver_Disease": (133, 5), "Healthy": (140, 2),
    }),
    ("potassium", "mEq/L", 3.5, 5.0, {
        "Pneumonia": (4.0, 0.5), "Heart_Disease": (4.3, 0.6),
        "Diabetes_Complication": (4.8, 0.7), "Liver_Disease": (4.2, 0.6), "Healthy": (4.2, 0.3),
    }),
    ("alt", "U/L", 7, 56, {
        "Pneumonia": (30, 12), "Heart_Disease": (35, 15),
        "Diabetes_Complication": (38, 15), "Liver_Disease": (120, 60), "Healthy": (25, 8),
    }),
    ("ast", "U/L", 10, 40, {
        "Pneumonia": (28, 10), "Heart_Disease": (32, 12),
        "Diabetes_Complication": (30, 12), "Liver_Disease": (110, 55), "Healthy": (22, 6),
    }),
    ("total_bilirubin", "mg/dL", 0.1, 1.2, {
        "Pneumonia": (0.8, 0.3), "Heart_Disease": (0.9, 0.3),
        "Diabetes_Complication": (0.7, 0.3), "Liver_Disease": (4.5, 2.5), "Healthy": (0.6, 0.2),
    }),
    ("albumin", "g/dL", 3.5, 5.5, {
        "Pneumonia": (3.4, 0.5), "Heart_Disease": (3.8, 0.4),
        "Diabetes_Complication": (3.6, 0.5), "Liver_Disease": (2.8, 0.6), "Healthy": (4.5, 0.3),
    }),
    ("total_cholesterol", "mg/dL", 0, 200, {
        "Pneumonia": (185, 35), "Heart_Disease": (245, 40),
        "Diabetes_Complication": (230, 40), "Liver_Disease": (170, 40), "Healthy": (180, 25),
    }),
    ("crp", "mg/L", 0, 3.0, {
        "Pneumonia": (85, 45), "Heart_Disease": (12, 8),
        "Diabetes_Complication": (8, 5), "Liver_Disease": (15, 10), "Healthy": (1.5, 1.0),
    }),
]


def generate_lab_results(patients_df):
    print("[3/6] Generating lab_results.csv ...")
    rows = []
    base_date = datetime.date(2024, 1, 1)
    for _, pat in patients_df.iterrows():
        dx = pat["_diagnosis"]
        for lab_idx in range(LABS_PER_PATIENT):
            test_date = base_date + datetime.timedelta(days=int(np.random.randint(0, 365)))
            for test_name, unit, ref_lo, ref_hi, dx_params in LAB_SPECS:
                mean, std = dx_params[dx]
                value = round(float(np.random.normal(mean, std)), 2)
                # Ensure non-negative for values that must be positive
                if value < 0:
                    value = round(abs(value), 2)
                rows.append({
                    "patient_id": pat["patient_id"],
                    "test_date": test_date.isoformat(),
                    "test_name": test_name,
                    "value": value,
                    "unit": unit,
                    "reference_low": ref_lo,
                    "reference_high": ref_hi,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "lab_results.csv"), index=False)
    print(f"    -> {len(df)} lab records written.")
    return df


# ===================================================================
# 4. CLINICAL_NOTES.CSV
# ===================================================================
NOTE_TEMPLATES = {
    "Pneumonia": {
        "admission": [
            "Pt is a {age} y/o {sex} presenting with {days}-day hx of productive cough, fever (Tmax {temp}F), and dyspnea. PMH significant for {pmh}. CXR shows {xray_finding}. Started on empiric abx (ceftriaxone + azithromycin). O2 sat {spo2}% on RA.",
            "Chief complaint: SOB, cough x {days} days. {age} y/o {sex} with worsening respiratory symptoms. Exam: bilateral crackles, decreased breath sounds R base. WBC {wbc}. CRP elevated at {crp}. Dx: Community-acquired pneumonia. Plan: IV antibiotics, supplemental O2, incentive spirometry.",
            "Patient {age} y/o {sex} admitted with acute onset fever, chills, productive cough with purulent sputum. Tachypneic at {rr} breaths/min. SpO2 {spo2}%. Chest auscultation reveals rhonchi and rales bilaterally. CURB-65 score {curb65}. Initiated on levofloxacin 750mg IV daily.",
        ],
        "progress": [
            "Day {day} of admission. Pt continues on IV abx. Fever curve trending down. O2 requirement decreased from {o2_prev}L to {o2_curr}L NC. WBC downtrending. Repeat CXR shows interval improvement. Continue current management.",
            "Progress note: Respiratory status improving. Cough less productive. Afebrile x 24h. SpO2 {spo2}% on {o2_curr}L NC. Plan to transition to PO antibiotics and consider discharge in 24-48h if stable.",
        ],
        "discharge": [
            "Discharge summary: {age} y/o {sex} admitted for CAP. Treated with IV ceftriaxone/azithromycin x {days} days, transitioned to PO amoxicillin-clavulanate. Afebrile, SpO2 {spo2}% on RA at discharge. Follow-up CXR in 6 weeks. Return precautions given.",
        ],
    },
    "Heart_Disease": {
        "admission": [
            "Pt is a {age} y/o {sex} presenting with acute onset substernal chest pain, radiating to L arm, associated with diaphoresis and nausea. PMH: HTN, HLD, {pmh}. ECG shows ST elevation in leads II, III, aVF. Troponin I elevated at {trop}. Dx: STEMI. Cardiology consulted for emergent cath.",
            "Chief complaint: Chest pressure x {days} hours. {age} y/o {sex} with hx of CAD, HTN. BP {sbp}/{dbp}, HR {hr} (irregular). BNP {bnp}. Echo shows EF {ef}%. Dx: Acute decompensated heart failure. Started on IV furosemide, ACEi, and continuous telemetry.",
            "{age} y/o {sex} with known CHF (EF {ef}%) presents with worsening dyspnea on exertion, orthopnea, and bilateral LE edema. JVD present. Lungs with bilateral rales. Pro-BNP significantly elevated. Admit for acute HF exacerbation. IV diuresis initiated.",
        ],
        "progress": [
            "Day {day}: Patient on telemetry. No arrhythmias overnight. Diuresis ongoing with net negative {fluid}mL. Weight down {wt_loss} kg. BNP trending down. Continue current regimen. Cardiology following.",
            "Hemodynamically stable. BP {sbp}/{dbp}. HR {hr}, regular. Chest pain resolved. Serial troponins downtrending. Echo scheduled for today. Continue dual antiplatelet therapy, beta-blocker, and statin.",
        ],
        "discharge": [
            "Discharge: {age} y/o {sex} s/p NSTEMI / CHF exacerbation. Underwent cardiac cath with PCI to {artery}. EF {ef}% on discharge echo. Discharged on ASA, clopidogrel, metoprolol, lisinopril, atorvastatin. Cardiac rehab referral placed. Follow-up with cardiology in 2 weeks.",
        ],
    },
    "Diabetes_Complication": {
        "admission": [
            "Pt is a {age} y/o {sex} with poorly controlled T2DM (HbA1c {hba1c}%) presenting with {complication}. Glucose on admission {glucose} mg/dL. Cr {cr}, eGFR {egfr}. Urine protein +{prot}. Dx: Diabetic {dx_detail}. Endocrine and {consult} consulted.",
            "Chief complaint: {symptom} x {days} days. {age} y/o {sex} with longstanding DM2, noncompliant with insulin. Exam: {exam_finding}. Labs: Glucose {glucose}, HbA1c {hba1c}%, BMP notable for {bmp_finding}. Assessment: Diabetic {dx_detail} with metabolic derangement.",
            "{age} y/o {sex} admitted with DKA. pH {ph}, bicarb {bicarb}, anion gap {ag}. Blood glucose {glucose} mg/dL. Started on insulin drip per DKA protocol. Aggressive IV fluid resuscitation with NS. Potassium replacement initiated. Monitoring q1h BMP.",
        ],
        "progress": [
            "Day {day}: Glucose range {glu_lo}-{glu_hi} mg/dL on current insulin regimen. Anion gap closed. Transitioned from insulin drip to subcutaneous basal-bolus. Diabetes educator consulted. Renal function stable, Cr {cr}.",
            "Endocrine progress note: Adjusting basal insulin to {basal} units. Sliding scale modified. Patient educated on carb counting. Ophthalmology cleared for {eye_finding}. A1c target discussed. CGM placement considered.",
        ],
        "discharge": [
            "Discharge: {age} y/o {sex} admitted for diabetic {dx_detail}. HbA1c {hba1c}%. Discharged on insulin glargine {basal}u QHS, lispro per sliding scale, metformin 1000mg BID. Endocrine follow-up in 1 week. Diabetic shoe fitting ordered. Home glucose monitoring reviewed.",
        ],
    },
    "Liver_Disease": {
        "admission": [
            "Pt is a {age} y/o {sex} presenting with {days}-day hx of worsening jaundice, RUQ pain, and abdominal distension. PMH: {pmh}. Exam: icteric sclera, hepatomegaly, ascites. Labs: AST {ast}, ALT {alt}, T.bili {tbili}, albumin {alb}, INR {inr}. Dx: {liver_dx}. MELD score {meld}.",
            "Chief complaint: Abdominal swelling, jaundice. {age} y/o {sex} with hx of {etiology} cirrhosis. Presents with increasing abdominal girth, LE edema. Paracentesis performed: {tap_vol}L removed, fluid sent for cell count/culture (r/o SBP). Albumin replacement given.",
            "{age} y/o {sex} admitted with acute hepatic decompensation. Encephalopathy grade {enc_grade}. Asterixis present. NH3 level {nh3}. Lactulose and rifaximin initiated. Child-Pugh class {cp_class}. GI consulted for variceal screening.",
        ],
        "progress": [
            "Day {day}: Liver function tests trending. AST {ast} -> {ast2}, ALT {alt} -> {alt2}. Bilirubin {tbili}. Mental status {ms_status}. Continued lactulose with {bm} BMs/day (goal 3-4). Diuretics adjusted. Na {na}.",
            "Hepatology progress note: Ascites management with spironolactone {spiro}mg and furosemide {lasix}mg. 2g Na diet. Daily weights and I/Os monitored. Transplant evaluation {tx_status}. Nutrition consult for protein optimization.",
        ],
        "discharge": [
            "Discharge: {age} y/o {sex} with {etiology} cirrhosis, admitted for {reason}. MELD {meld} at discharge. Discharged on lactulose 30mL TID, rifaximin 550mg BID, spironolactone/furosemide. Transplant listing {tx_status}. GI follow-up in 1 week. Return for fever, worsening confusion, or GI bleeding.",
        ],
    },
    "Healthy": {
        "admission": [
            "Pt is a {age} y/o {sex} presenting for routine annual physical examination. No acute complaints. PMH: unremarkable. Medications: none. Vitals WNL. Comprehensive metabolic panel, CBC, lipid panel all within normal limits. Assessment: Healthy adult. Plan: Continue preventive care, age-appropriate screenings.",
            "Routine wellness visit. {age} y/o {sex}. ROS negative. Physical exam unremarkable. BMI {bmi}. BP {sbp}/{dbp}. Labs reviewed and within normal ranges. Immunizations up to date. Counseled on diet, exercise, and preventive health measures.",
        ],
        "progress": [
            "Follow-up visit: Patient reports feeling well. No new symptoms. Vitals stable. Labs from last visit reviewed - all WNL. Continue current health maintenance. Return in 12 months or PRN.",
        ],
        "discharge": [
            "Visit summary: {age} y/o healthy {sex}. Routine checkup with no abnormalities identified. Screening labs normal. Encouraged regular exercise (150 min/wk moderate intensity) and balanced diet. Next visit in 1 year.",
        ],
    },
}


def _fill_template(template, pat, dx):
    """Fill in template placeholders with plausible values."""
    sex_word = "male" if pat["sex"] == "M" else "female"
    replacements = {
        "{age}": str(pat["age"]),
        "{sex}": sex_word,
        "{bmi}": str(pat["bmi"]),
        "{sbp}": str(pat["systolic_bp"]),
        "{dbp}": str(pat["diastolic_bp"]),
        "{hr}": str(pat["heart_rate"]),
        "{spo2}": str(pat["oxygen_saturation"]),
        "{rr}": str(pat["respiratory_rate"]),
        "{temp}": str(round(pat["temperature"] * 9 / 5 + 32, 1)),
        "{days}": str(np.random.randint(1, 14)),
        "{day}": str(np.random.randint(2, 7)),
        "{pmh}": random.choice(["HTN", "DM2", "COPD", "HLD", "CAD", "no significant PMH"]),
        "{xray_finding}": random.choice([
            "RLL consolidation", "bilateral infiltrates", "LLL opacity",
            "R middle lobe consolidation with air bronchograms",
        ]),
        "{wbc}": str(round(np.random.normal(15, 4), 1)),
        "{crp}": str(round(np.random.normal(80, 30), 1)),
        "{curb65}": str(np.random.randint(1, 4)),
        "{o2_prev}": str(np.random.randint(3, 6)),
        "{o2_curr}": str(np.random.randint(1, 3)),
        "{trop}": str(round(np.random.uniform(0.04, 12.0), 2)),
        "{bnp}": str(int(np.random.uniform(400, 2500))),
        "{ef}": str(np.random.randint(20, 55)),
        "{fluid}": str(np.random.randint(500, 2500)),
        "{wt_loss}": str(round(np.random.uniform(0.5, 3.0), 1)),
        "{artery}": random.choice(["LAD", "RCA", "LCx", "diagonal branch"]),
        "{hba1c}": str(round(np.random.normal(9.2, 1.5), 1)),
        "{glucose}": str(int(np.random.normal(250, 80))),
        "{cr}": str(round(np.random.normal(1.8, 0.5), 1)),
        "{egfr}": str(int(np.random.normal(45, 15))),
        "{prot}": str(np.random.randint(1, 4)),
        "{complication}": random.choice([
            "diabetic ketoacidosis", "hyperosmolar hyperglycemic state",
            "diabetic nephropathy exacerbation", "diabetic foot ulcer with cellulitis",
        ]),
        "{dx_detail}": random.choice(["nephropathy", "neuropathy", "ketoacidosis", "retinopathy"]),
        "{consult}": random.choice(["nephrology", "podiatry", "ophthalmology"]),
        "{symptom}": random.choice(["worsening vision", "foot numbness", "polyuria/polydipsia", "wound on foot"]),
        "{exam_finding}": random.choice([
            "decreased sensation bilateral feet", "non-healing ulcer R great toe",
            "cotton-wool spots on fundoscopy", "Kussmaul respirations",
        ]),
        "{bmp_finding}": random.choice(["elevated anion gap", "hyponatremia", "hyperkalemia", "elevated Cr"]),
        "{ph}": str(round(np.random.uniform(7.1, 7.3), 2)),
        "{bicarb}": str(int(np.random.uniform(8, 18))),
        "{ag}": str(int(np.random.uniform(16, 28))),
        "{glu_lo}": str(int(np.random.uniform(120, 180))),
        "{glu_hi}": str(int(np.random.uniform(220, 350))),
        "{basal}": str(int(np.random.uniform(20, 60))),
        "{eye_finding}": random.choice(["non-proliferative retinopathy", "proliferative retinopathy", "macular edema", "no acute findings"]),
        "{ast}": str(int(np.random.normal(110, 50))),
        "{ast2}": str(int(np.random.normal(85, 40))),
        "{alt}": str(int(np.random.normal(120, 55))),
        "{alt2}": str(int(np.random.normal(90, 40))),
        "{tbili}": str(round(np.random.normal(4.5, 2.0), 1)),
        "{alb}": str(round(np.random.normal(2.8, 0.5), 1)),
        "{inr}": str(round(np.random.uniform(1.3, 2.5), 1)),
        "{liver_dx}": random.choice([
            "hepatic cirrhosis decompensation", "acute-on-chronic liver failure",
            "alcoholic hepatitis", "hepatocellular carcinoma",
        ]),
        "{meld}": str(np.random.randint(12, 35)),
        "{etiology}": random.choice(["EtOH", "NASH", "HCV", "HBV", "cryptogenic"]),
        "{tap_vol}": str(round(np.random.uniform(2, 8), 1)),
        "{enc_grade}": str(np.random.randint(1, 4)),
        "{nh3}": str(int(np.random.uniform(60, 180))),
        "{cp_class}": random.choice(["B", "C"]),
        "{ms_status}": random.choice(["improved", "unchanged", "slightly confused"]),
        "{bm}": str(np.random.randint(2, 6)),
        "{na}": str(int(np.random.normal(134, 4))),
        "{spiro}": str(random.choice([50, 100, 200])),
        "{lasix}": str(random.choice([20, 40, 80])),
        "{tx_status}": random.choice(["initiated", "pending workup", "deferred", "listed"]),
        "{reason}": random.choice(["ascites management", "hepatic encephalopathy", "variceal bleed", "SBP"]),
    }
    text = template
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def generate_clinical_notes(patients_df):
    print("[4/6] Generating clinical_notes.csv ...")
    rows = []
    base_date = datetime.date(2024, 1, 1)
    note_types = ["admission", "progress", "discharge"]

    for _, pat in patients_df.iterrows():
        dx = pat["_diagnosis"]
        templates = NOTE_TEMPLATES[dx]
        # Generate 1-3 notes per patient
        num_notes = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
        chosen_types = np.random.choice(note_types, size=num_notes, replace=False)

        for nt in chosen_types:
            note_date = base_date + datetime.timedelta(days=int(np.random.randint(0, 365)))
            template = random.choice(templates[nt])
            note_text = _fill_template(template, pat, dx)
            rows.append({
                "patient_id": pat["patient_id"],
                "note_date": note_date.isoformat(),
                "note_type": nt,
                "note_text": note_text,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "clinical_notes.csv"), index=False)
    print(f"    -> {len(df)} clinical notes written.")
    return df


# ===================================================================
# 5. IMAGE_METADATA.CSV + SYNTHETIC IMAGES
# ===================================================================
def _generate_synthetic_xray(dx, image_path):
    """
    Generate a 224x224 grayscale image that simulates different
    chest X-ray patterns using Gaussian noise with diagnosis-specific
    characteristics.
    """
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)

    # Base tissue-like background
    img += np.random.normal(128, 20, (IMAGE_SIZE, IMAGE_SIZE))

    # Create a basic chest-like structure (elliptical lung fields)
    y, x = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE), indexing="ij")
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2

    # Mediastinum (bright vertical strip in center)
    mediastinum_mask = np.abs(x - cx) < 20
    img[mediastinum_mask] += 40

    # Lung fields (darker regions on either side)
    left_lung = ((x - cx + 45) ** 2 / 50 ** 2 + (y - cy) ** 2 / 80 ** 2) < 1
    right_lung = ((x - cx - 45) ** 2 / 50 ** 2 + (y - cy) ** 2 / 80 ** 2) < 1
    img[left_lung] -= 30
    img[right_lung] -= 30

    if dx == "Pneumonia":
        # Add focal consolidation (bright patch in one lung)
        side = np.random.choice([-1, 1])
        cons_cx = cx + side * np.random.randint(30, 55)
        cons_cy = cy + np.random.randint(10, 50)
        cons_r = np.random.randint(20, 40)
        consolidation = ((x - cons_cx) ** 2 + (y - cons_cy) ** 2) < cons_r ** 2
        img[consolidation] += np.random.normal(50, 15, img[consolidation].shape)
        # Air bronchograms (small darker streaks within consolidation)
        for _ in range(3):
            sx = cons_cx + np.random.randint(-10, 10)
            sy = cons_cy + np.random.randint(-10, 10)
            streak = ((x - sx) ** 2 / 8 ** 2 + (y - sy) ** 2 / 2 ** 2) < 1
            img[streak] -= 20

    elif dx == "Heart_Disease":
        # Cardiomegaly: enlarged cardiac silhouette
        heart_r = np.random.randint(55, 75)
        heart_mask = ((x - cx) ** 2 / heart_r ** 2 + (y - (cy + 15)) ** 2 / (heart_r - 10) ** 2) < 1
        img[heart_mask] += np.random.normal(35, 10, img[heart_mask].shape)
        # Pulmonary edema (bilateral haziness)
        edema_noise = np.random.normal(0, 12, (IMAGE_SIZE, IMAGE_SIZE))
        lung_mask = left_lung | right_lung
        img[lung_mask] += edema_noise[lung_mask]

    elif dx == "Diabetes_Complication":
        # Generally normal-appearing CXR with subtle findings
        img += np.random.normal(0, 5, (IMAGE_SIZE, IMAGE_SIZE))

    elif dx == "Liver_Disease":
        # Possible pleural effusion (bright region at base)
        effusion_mask = y > (IMAGE_SIZE - np.random.randint(30, 55))
        img[effusion_mask] += np.random.normal(40, 10, img[effusion_mask].shape)

    else:  # Healthy
        # Clean image with minimal noise
        img += np.random.normal(0, 3, (IMAGE_SIZE, IMAGE_SIZE))

    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(image_path)


def generate_images_and_metadata(patients_df):
    print("[5/6] Generating image_metadata.csv and synthetic X-ray images ...")

    # Select IMAGE_PATIENTS patients (200 per class)
    samples = []
    for dx_name in DIAGNOSES:
        group = patients_df[patients_df["_diagnosis"] == dx_name]
        samples.append(group.sample(IMAGE_PATIENTS // len(DIAGNOSES), random_state=SEED))
    selected = pd.concat(samples, ignore_index=True)

    finding_map = {
        "Pneumonia": ["consolidation", "infiltrate", "air_bronchograms", "lobar_opacity"],
        "Heart_Disease": ["cardiomegaly", "pulmonary_edema", "pleural_effusion", "enlarged_cardiac_silhouette"],
        "Diabetes_Complication": ["no_acute_finding", "normal", "mild_atherosclerosis"],
        "Liver_Disease": ["pleural_effusion", "elevated_hemidiaphragm", "hepatomegaly_shadow"],
        "Healthy": ["normal", "no_acute_finding"],
    }

    rows = []
    for idx, (_, pat) in enumerate(selected.iterrows()):
        dx = pat["_diagnosis"]
        image_id = f"IMG{idx + 1:05d}"
        image_filename = f"{image_id}.png"
        image_path_on_disk = os.path.join(IMAGE_DIR, image_filename)
        relative_image_path = f"images/{image_filename}"

        _generate_synthetic_xray(dx, image_path_on_disk)

        rows.append({
            "patient_id": pat["patient_id"],
            "image_id": image_id,
            "image_path": relative_image_path,
            "modality": "X-ray",
            "body_part": "Chest",
            "finding_label": random.choice(finding_map[dx]),
        })

        if (idx + 1) % 200 == 0:
            print(f"    -> {idx + 1}/{IMAGE_PATIENTS} images generated ...")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "image_metadata.csv"), index=False)
    print(f"    -> {len(df)} image records written.")
    return df


# ===================================================================
# 6. GROUND_TRUTH.CSV
# ===================================================================
def generate_ground_truth(patients_df):
    print("[6/6] Generating ground_truth.csv ...")
    rows = []
    for _, pat in patients_df.iterrows():
        dx = pat["_diagnosis"]

        # Risk level based on diagnosis and vitals
        if dx == "Healthy":
            risk = "Low"
            icu = 0
        else:
            spo2 = pat["oxygen_saturation"]
            hr = pat["heart_rate"]
            temp = pat["temperature"]

            # Scoring for severity
            severity_score = 0
            if spo2 < 90:
                severity_score += 3
            elif spo2 < 94:
                severity_score += 2
            elif spo2 < 96:
                severity_score += 1

            if hr > 120 or hr < 50:
                severity_score += 2
            elif hr > 100 or hr < 60:
                severity_score += 1

            if temp > 39.5:
                severity_score += 2
            elif temp > 38.5:
                severity_score += 1

            if pat["age"] > 70:
                severity_score += 1

            if severity_score >= 5:
                risk = "Critical"
            elif severity_score >= 3:
                risk = "High"
            elif severity_score >= 1:
                risk = "Moderate"
            else:
                risk = "Low"

            icu = int(risk in ("Critical",) or (risk == "High" and np.random.rand() < 0.3))

        rows.append({
            "patient_id": pat["patient_id"],
            "final_diagnosis": dx,
            "risk_level": risk,
            "requires_icu": icu,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "ground_truth.csv"), index=False)
    print(f"    -> {len(df)} ground truth records written.")

    # Print distribution summary
    print("\n--- Risk Level Distribution ---")
    print(df["risk_level"].value_counts().to_string())
    print("\n--- ICU Requirement ---")
    print(df["requires_icu"].value_counts().to_string())
    return df


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("  Healthcare Intelligence System - Synthetic Data Generator")
    print("=" * 60)
    print(f"  Patients: {NUM_PATIENTS} ({PATIENTS_PER_CLASS} per class)")
    print(f"  Diagnoses: {', '.join(DIAGNOSES)}")
    print(f"  Output dir: {RAW_DIR}")
    print("=" * 60)
    print()

    patients_df = generate_patients()
    print()
    generate_symptoms(patients_df)
    print()
    generate_lab_results(patients_df)
    print()
    generate_clinical_notes(patients_df)
    print()
    generate_images_and_metadata(patients_df)
    print()
    generate_ground_truth(patients_df)

    print()
    print("=" * 60)
    print("  All synthetic data generated successfully!")
    print(f"  Files written to: {RAW_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
