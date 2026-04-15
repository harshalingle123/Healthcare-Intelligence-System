"""
Healthcare Intelligence System - Main Streamlit Dashboard
Run with: streamlit run app/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure app package is importable when running from project root
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from app.components import (
    render_clinical_note,
    render_lab_results_table,
    render_modality_scores,
    render_patient_card,
    render_risk_gauge,
    render_symptoms_display,
    render_vital_signs,
)
from app.model_loader import (
    load_cached_data,
    load_metrics,
    load_predictions,
    simulate_risk_scores,
)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "outputs")
IMAGES_DIR = os.path.join(PROJECT_DIR, "data", "raw", "images")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    [data-testid="stMetric"] {
        background: rgba(52, 152, 219, 0.10);
        border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #3498db;
    }
    [data-testid="stMetric"] label {
        color: inherit !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard Overview",
        "Patient Analysis",
        "Risk Assessment",
        "Model Performance",
        "Batch Analysis",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Healthcare Intelligence System**")
st.sidebar.caption("Multi-modal ML pipeline for clinical risk prediction")
st.sidebar.caption(f"Data dir: `{DATA_DIR}`")

# ── Load data ──────────────────────────────────────────────────────────────
data = load_cached_data(DATA_DIR)
patients_df = data.get("patients", pd.DataFrame())
symptoms_df = data.get("symptoms", pd.DataFrame())
lab_df = data.get("lab_results", pd.DataFrame())
notes_df = data.get("clinical_notes", pd.DataFrame())
images_df = data.get("image_metadata", pd.DataFrame())
ground_truth_df = data.get("ground_truth", pd.DataFrame())


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 - Dashboard Overview
# ═══════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.title("Healthcare Intelligence System")
    st.markdown("Multi-modal clinical risk prediction dashboard combining structured data, "
                "NLP, lab results, and medical imaging.")

    if patients_df.empty and ground_truth_df.empty:
        st.warning("No data files found. Please ensure CSV files exist in data/raw/.")
        return

    # ── Key metrics ────────────────────────────────────────────────────────
    total_patients = len(patients_df) if not patients_df.empty else 0
    models_trained = 0
    metrics = load_metrics(OUTPUT_DIR)
    if metrics:
        models_trained = len(metrics) if isinstance(metrics, dict) else 1

    # Compute average simulated risk for a sample (cache-friendly)
    avg_risk = 0.0
    critical_pct = 0.0
    if not ground_truth_df.empty:
        risk_counts = ground_truth_df["risk_level"].value_counts()
        critical_count = risk_counts.get("Critical", 0)
        critical_pct = round(100 * critical_count / len(ground_truth_df), 1)
        risk_map = {"Low": 15, "Moderate": 40, "High": 65, "Critical": 85}
        avg_risk = round(
            ground_truth_df["risk_level"].map(risk_map).mean(), 1
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{total_patients:,}")
    c2.metric("Models Trained", models_trained if models_trained else "0 (demo)")
    c3.metric("Avg Risk Score", f"{avg_risk}")
    c4.metric("Critical Patients", f"{critical_pct}%")

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Level Distribution")
        if not ground_truth_df.empty:
            risk_counts = ground_truth_df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            color_map = {"Low": "#2ecc71", "Moderate": "#f1c40f",
                         "High": "#e67e22", "Critical": "#e74c3c"}
            fig = px.pie(
                risk_counts, names="Risk Level", values="Count",
                color="Risk Level", color_discrete_map=color_map,
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ground truth data available.")

    with col_right:
        st.subheader("Diagnosis Distribution")
        if not ground_truth_df.empty and "final_diagnosis" in ground_truth_df.columns:
            diag_counts = ground_truth_df["final_diagnosis"].value_counts().head(10).reset_index()
            diag_counts.columns = ["Diagnosis", "Count"]
            fig = px.bar(
                diag_counts, x="Count", y="Diagnosis", orientation="h",
                color="Count", color_continuous_scale="Blues",
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        elif not symptoms_df.empty and "primary_diagnosis" in symptoms_df.columns:
            diag_counts = symptoms_df["primary_diagnosis"].value_counts().head(10).reset_index()
            diag_counts.columns = ["Diagnosis", "Count"]
            fig = px.bar(
                diag_counts, x="Count", y="Diagnosis", orientation="h",
                color="Count", color_continuous_scale="Blues",
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No diagnosis data available.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 - Patient Analysis
# ═══════════════════════════════════════════════════════════════════════════
def page_patient_analysis():
    st.title("Patient Analysis")

    if patients_df.empty:
        st.warning("No patient data found.")
        return

    patient_ids = sorted(patients_df["patient_id"].unique().tolist())
    selected_id = st.sidebar.selectbox("Select Patient", patient_ids)

    patient_row = patients_df[patients_df["patient_id"] == selected_id].iloc[0]

    # Demographics
    render_patient_card(patient_row.to_dict())
    st.markdown("---")

    # Vital signs
    render_vital_signs(patient_row.to_dict())
    st.markdown("---")

    # Symptoms
    sym_row = symptoms_df[symptoms_df["patient_id"] == selected_id]
    if not sym_row.empty:
        render_symptoms_display(sym_row.iloc[0].to_dict())
    else:
        st.info("No symptoms data for this patient.")

    st.markdown("---")

    # Lab results
    patient_labs = lab_df[lab_df["patient_id"] == selected_id]
    render_lab_results_table(patient_labs)
    st.markdown("---")

    # Clinical notes
    patient_notes = notes_df[notes_df["patient_id"] == selected_id]
    if not patient_notes.empty:
        st.subheader("Clinical Notes")
        for _, note in patient_notes.iterrows():
            render_clinical_note(
                note.get("note_text", ""),
                note.get("note_type", "general"),
            )
    else:
        st.info("No clinical notes for this patient.")

    # Chest X-ray
    patient_images = images_df[images_df["patient_id"] == selected_id] \
        if not images_df.empty else pd.DataFrame()
    if not patient_images.empty:
        st.subheader("Chest X-ray")
        for _, img_row in patient_images.iterrows():
            img_path = os.path.join(PROJECT_DIR, "data", "raw", img_row["image_path"])
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{img_row.get('finding_label', '')} "
                         f"({img_row.get('image_id', '')})", width=400)
            else:
                st.caption(f"Image file not found: {img_row['image_path']}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 - Risk Assessment
# ═══════════════════════════════════════════════════════════════════════════
def page_risk_assessment():
    st.title("Risk Assessment")

    if patients_df.empty:
        st.warning("No patient data found.")
        return

    patient_ids = sorted(patients_df["patient_id"].unique().tolist())
    selected_id = st.sidebar.selectbox("Select Patient", patient_ids, key="risk_patient")

    # Try ensemble predictions first
    predictions = load_predictions(OUTPUT_DIR)
    if (predictions is not None
            and "patient_id" in predictions.columns
            and selected_id in predictions["patient_id"].values):
        pred_row = predictions[predictions["patient_id"] == selected_id].iloc[0]
        scores = {
            "overall_score": float(pred_row.get("risk_score", 50)),
            "risk_level": pred_row.get("risk_level", "Moderate"),
            "modality_scores": {
                "Structured Data": float(pred_row.get("structured_score", 50)),
                "NLP (Clinical Notes)": float(pred_row.get("nlp_score", 50)),
                "Lab Results": float(pred_row.get("lab_score", 50)),
                "Medical Imaging": float(pred_row.get("imaging_score", 50)),
            },
            "contributing_factors": pred_row.get("factors", "").split(";")
            if "factors" in pred_row.index else [],
        }
        st.info("Showing predictions from trained ensemble model.")
    else:
        scores = simulate_risk_scores(selected_id, patients_df, symptoms_df, lab_df)
        st.info("No trained model predictions found. Showing heuristic-based demo scores.")

    # Gauge
    render_risk_gauge(scores["overall_score"], scores["risk_level"])

    # Modality scores
    col1, col2 = st.columns([3, 2])
    with col1:
        render_modality_scores(scores["modality_scores"])
    with col2:
        st.subheader("Top Contributing Factors")
        if scores["contributing_factors"]:
            for i, factor in enumerate(scores["contributing_factors"], 1):
                st.markdown(f"**{i}.** {factor}")
        else:
            st.caption("No specific risk factors identified.")

    # Clinical actions
    st.markdown("---")
    st.subheader("Recommended Clinical Actions")
    level = scores["risk_level"]
    if level == "Critical":
        st.error("CRITICAL RISK - Immediate intervention required")
        actions = [
            "Escalate to attending physician immediately",
            "Consider ICU admission",
            "Continuous vital sign monitoring",
            "Stat labs and imaging",
            "Initiate rapid response protocol",
        ]
    elif level == "High":
        st.warning("HIGH RISK - Close monitoring needed")
        actions = [
            "Notify attending physician within 1 hour",
            "Increase monitoring frequency to q2h",
            "Review and optimize current treatment plan",
            "Consider specialist consultation",
            "Repeat labs in 6 hours",
        ]
    elif level == "Moderate":
        st.info("MODERATE RISK - Standard monitoring with vigilance")
        actions = [
            "Continue current monitoring schedule (q4h)",
            "Review lab trends at next rounding",
            "Ensure follow-up appointments scheduled",
            "Patient education on warning signs",
        ]
    else:
        st.success("LOW RISK - Routine care")
        actions = [
            "Continue routine monitoring",
            "Standard follow-up in 2 weeks",
            "Reinforce preventive care and lifestyle modifications",
        ]
    for a in actions:
        st.markdown(f"- {a}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 - Model Performance
# ═══════════════════════════════════════════════════════════════════════════
def page_model_performance():
    st.title("Model Performance")

    metrics = load_metrics(OUTPUT_DIR)

    if metrics is None:
        st.info("No trained model metrics found. Displaying placeholder performance data "
                "for demonstration purposes.")
        metrics = {
            "structured_model": {
                "accuracy": 0.82, "f1_score": 0.79, "auc_roc": 0.88,
                "fpr": [0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0],
                "tpr": [0, 0.4, 0.6, 0.78, 0.88, 0.94, 1.0],
                "confusion_matrix": [[140, 35], [28, 147]],
            },
            "nlp_model": {
                "accuracy": 0.78, "f1_score": 0.75, "auc_roc": 0.84,
                "fpr": [0, 0.08, 0.15, 0.25, 0.45, 0.65, 1.0],
                "tpr": [0, 0.35, 0.55, 0.72, 0.84, 0.92, 1.0],
                "confusion_matrix": [[132, 43], [34, 141]],
            },
            "lab_model": {
                "accuracy": 0.80, "f1_score": 0.77, "auc_roc": 0.86,
                "fpr": [0, 0.06, 0.12, 0.22, 0.42, 0.62, 1.0],
                "tpr": [0, 0.38, 0.58, 0.75, 0.86, 0.93, 1.0],
                "confusion_matrix": [[136, 39], [31, 144]],
            },
            "imaging_model": {
                "accuracy": 0.76, "f1_score": 0.73, "auc_roc": 0.82,
                "fpr": [0, 0.1, 0.18, 0.28, 0.48, 0.68, 1.0],
                "tpr": [0, 0.32, 0.5, 0.68, 0.82, 0.90, 1.0],
                "confusion_matrix": [[128, 47], [37, 138]],
            },
            "ensemble": {
                "accuracy": 0.87, "f1_score": 0.85, "auc_roc": 0.93,
                "fpr": [0, 0.03, 0.07, 0.14, 0.3, 0.5, 1.0],
                "tpr": [0, 0.5, 0.7, 0.85, 0.93, 0.97, 1.0],
                "confusion_matrix": [[152, 23], [22, 153]],
            },
        }

    # ── Comparison table ──────────────────────────────────────────────────
    st.subheader("Model Comparison")
    table_data = []
    for model_name, m in metrics.items():
        table_data.append({
            "Model": model_name.replace("_", " ").title(),
            "Accuracy": f"{m.get('accuracy', 0):.3f}",
            "F1 Score": f"{m.get('f1_score', 0):.3f}",
            "AUC-ROC": f"{m.get('auc_roc', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── ROC Curves ────────────────────────────────────────────────────────
    with col1:
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        colors = px.colors.qualitative.Set2
        for idx, (model_name, m) in enumerate(metrics.items()):
            fpr = m.get("fpr", [0, 1])
            tpr = m.get("tpr", [0, 1])
            auc_val = m.get("auc_roc", 0)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{model_name.replace('_', ' ').title()} (AUC={auc_val:.2f})",
                line=dict(color=colors[idx % len(colors)], width=2),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random", line=dict(color="gray", dash="dash"),
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=420, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Confusion matrix ──────────────────────────────────────────────────
    with col2:
        st.subheader("Confusion Matrix")
        model_names = list(metrics.keys())
        sel_model = st.selectbox("Select model", model_names,
                                 format_func=lambda x: x.replace("_", " ").title())
        cm = metrics[sel_model].get("confusion_matrix", [[0, 0], [0, 0]])
        cm_arr = np.array(cm)
        fig_cm = px.imshow(
            cm_arr,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Negative", "Positive"], y=["Negative", "Positive"],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(height=400, margin=dict(t=20, b=40))
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Modality comparison grouped bar ────────────────────────────────────
    st.markdown("---")
    st.subheader("Modality Comparison")
    metric_names = ["Accuracy", "F1 Score", "AUC-ROC"]
    fig_bar = go.Figure()
    for metric_key, metric_label in [("accuracy", "Accuracy"),
                                      ("f1_score", "F1 Score"),
                                      ("auc_roc", "AUC-ROC")]:
        fig_bar.add_trace(go.Bar(
            x=[mn.replace("_", " ").title() for mn in metrics],
            y=[metrics[mn].get(metric_key, 0) for mn in metrics],
            name=metric_label,
        ))
    fig_bar.update_layout(
        barmode="group", yaxis_title="Score", height=400,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 - Batch Analysis
# ═══════════════════════════════════════════════════════════════════════════
def page_batch_analysis():
    st.title("Batch Analysis")
    st.markdown("Upload a CSV of patient records to run batch risk predictions.")

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return

        st.subheader("Data Preview")
        st.dataframe(batch_df.head(20), use_container_width=True, hide_index=True)
        st.caption(f"{len(batch_df)} rows x {len(batch_df.columns)} columns")

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Running predictions..."):
                results = []
                for _, row in batch_df.iterrows():
                    pid = row.get("patient_id", f"BATCH_{_}")
                    # Simple heuristic scoring for demo
                    score = 30.0
                    if "heart_rate" in row and pd.notna(row["heart_rate"]):
                        hr = float(row["heart_rate"])
                        if hr > 100 or hr < 60:
                            score += 12
                    if "systolic_bp" in row and pd.notna(row["systolic_bp"]):
                        sbp = float(row["systolic_bp"])
                        if sbp > 140 or sbp < 90:
                            score += 10
                    if "oxygen_saturation" in row and pd.notna(row["oxygen_saturation"]):
                        spo2 = float(row["oxygen_saturation"])
                        if spo2 < 95:
                            score += 15
                    if "temperature" in row and pd.notna(row["temperature"]):
                        temp = float(row["temperature"])
                        if temp > 38.5:
                            score += 10
                    if "age" in row and pd.notna(row["age"]):
                        age = float(row["age"])
                        if age > 65:
                            score += 10
                        elif age > 50:
                            score += 5
                    # Add some variance
                    np.random.seed(hash(str(pid)) % (2**31))
                    score += np.random.uniform(-5, 10)
                    score = max(0, min(100, score))

                    if score >= 75:
                        level = "Critical"
                    elif score >= 50:
                        level = "High"
                    elif score >= 25:
                        level = "Moderate"
                    else:
                        level = "Low"

                    results.append({
                        "patient_id": pid,
                        "risk_score": round(score, 1),
                        "risk_level": level,
                    })

                results_df = pd.DataFrame(results)

            st.subheader("Prediction Results")

            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Summary
            st.markdown("---")
            summary_cols = st.columns(4)
            for i, level in enumerate(["Low", "Moderate", "High", "Critical"]):
                count = len(results_df[results_df["risk_level"] == level])
                summary_cols[i].metric(f"{level} Risk", count)

            # Download
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload a CSV file with patient data columns (patient_id, age, sex, "
                "heart_rate, systolic_bp, oxygen_saturation, temperature, etc.) "
                "to generate batch risk predictions.")


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════
if page == "Dashboard Overview":
    page_dashboard()
elif page == "Patient Analysis":
    page_patient_analysis()
elif page == "Risk Assessment":
    page_risk_assessment()
elif page == "Model Performance":
    page_model_performance()
elif page == "Batch Analysis":
    page_batch_analysis()
