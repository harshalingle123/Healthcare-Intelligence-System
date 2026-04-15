"""
Reusable Streamlit UI components for the Healthcare Intelligence System dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Patient demographics card
# ---------------------------------------------------------------------------
def render_patient_card(patient_data: dict):
    """Compact card showing patient demographics using st.columns."""
    st.subheader(f"Patient {patient_data.get('patient_id', 'N/A')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", patient_data.get("age", "N/A"))
    c2.metric("Sex", patient_data.get("sex", "N/A"))
    c3.metric("BMI", f"{patient_data.get('bmi', 0):.1f}")
    c4.metric("Smoking", patient_data.get("smoking_status", "N/A"))

    # Comorbidities row
    comorbidities = []
    for cond in ["diabetes", "hypertension", "heart_disease_history",
                  "liver_disease_history", "lung_disease_history"]:
        if patient_data.get(cond, 0) == 1:
            comorbidities.append(cond.replace("_", " ").title())
    if comorbidities:
        st.markdown(
            "**Comorbidities:** " + " | ".join(
                f":red[{c}]" for c in comorbidities
            )
        )
    else:
        st.markdown("**Comorbidities:** None recorded")


# ---------------------------------------------------------------------------
# Vital signs with color-coded indicators
# ---------------------------------------------------------------------------
_VITAL_THRESHOLDS = {
    "heart_rate": {"low": 60, "high": 100, "unit": "bpm", "label": "Heart Rate"},
    "systolic_bp": {"low": 90, "high": 140, "unit": "mmHg", "label": "Systolic BP"},
    "diastolic_bp": {"low": 60, "high": 90, "unit": "mmHg", "label": "Diastolic BP"},
    "temperature": {"low": 36.0, "high": 38.5, "unit": "C", "label": "Temperature"},
    "respiratory_rate": {"low": 12, "high": 20, "unit": "/min", "label": "Resp Rate"},
    "oxygen_saturation": {"low": 95, "high": 100, "unit": "%", "label": "SpO2"},
}


def render_vital_signs(vitals_dict: dict):
    """Display vital signs with red/green color coding based on normal ranges."""
    st.subheader("Vital Signs")
    cols = st.columns(len(_VITAL_THRESHOLDS))
    for col, (key, thresh) in zip(cols, _VITAL_THRESHOLDS.items()):
        val = vitals_dict.get(key)
        if val is None:
            col.metric(thresh["label"], "N/A")
            continue
        val = float(val)
        is_normal = thresh["low"] <= val <= thresh["high"]
        # SpO2 has no real "high" abnormality; only low matters
        if key == "oxygen_saturation":
            is_normal = val >= thresh["low"]
        color = "#27ae60" if is_normal else "#e74c3c"
        status = "Normal" if is_normal else "Abnormal"
        col.markdown(
            f"**{thresh['label']}**<br>"
            f"<span style='font-size:1.6rem;color:{color};font-weight:700'>"
            f"{val:.1f}</span> {thresh['unit']}<br>"
            f"<span style='color:{color};font-weight:600'>{status}</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Lab results table with conditional formatting
# ---------------------------------------------------------------------------
def render_lab_results_table(lab_df: pd.DataFrame):
    """Styled DataFrame with red/green highlighting for abnormal/normal."""
    st.subheader("Lab Results")
    if lab_df.empty:
        st.info("No lab results available for this patient.")
        return

    display = lab_df[["test_name", "value", "unit", "reference_low", "reference_high"]].copy()
    display["Status"] = display.apply(
        lambda r: "Normal"
        if r["reference_low"] <= r["value"] <= r["reference_high"]
        else "Abnormal",
        axis=1,
    )
    display["Value"] = display["value"].map("{:.2f}".format)
    display["Ref Low"] = display["reference_low"].map("{:.1f}".format)
    display["Ref High"] = display["reference_high"].map("{:.1f}".format)
    display.columns = ["Test", "value_raw", "Unit", "ref_low_raw", "ref_high_raw", "Status", "Value", "Ref Low", "Ref High"]
    display = display[["Test", "Value", "Unit", "Ref Low", "Ref High", "Status"]]

    st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Risk gauge chart
# ---------------------------------------------------------------------------
def render_risk_gauge(risk_score: float, risk_level: str):
    """Plotly gauge with color zones: green/yellow/orange/red."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            title={"text": f"Overall Risk Score  -  {risk_level}", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 25], "color": "#2ecc71"},
                    {"range": [25, 50], "color": "#f1c40f"},
                    {"range": [50, 75], "color": "#e67e22"},
                    {"range": [75, 100], "color": "#e74c3c"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": risk_score,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=40, r=40))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Symptoms display
# ---------------------------------------------------------------------------
def render_symptoms_display(symptoms_dict: dict):
    """Grid of symptoms with active ones highlighted."""
    st.subheader("Symptoms Checklist")
    exclude = {"patient_id", "timestamp", "primary_diagnosis"}
    items = {k: v for k, v in symptoms_dict.items() if k not in exclude}
    cols = st.columns(5)
    for idx, (symptom, active) in enumerate(items.items()):
        label = symptom.replace("_", " ").title()
        active = int(active) == 1
        with cols[idx % 5]:
            if active:
                st.markdown(
                    f"<div style='background:#e74c3c;color:white;padding:6px 10px;"
                    f"border-radius:6px;margin:3px 0;text-align:center;font-weight:600'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background:rgba(128,128,128,0.15);color:inherit;padding:6px 10px;"
                    f"border-radius:6px;margin:3px 0;text-align:center;opacity:0.7'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Clinical note display
# ---------------------------------------------------------------------------
def render_clinical_note(note_text: str, note_type: str):
    """Formatted text display with note type badge."""
    badge_colors = {
        "admission": "#3498db",
        "progress": "#2ecc71",
        "discharge": "#9b59b6",
        "consultation": "#e67e22",
    }
    color = badge_colors.get(note_type.lower(), "#95a5a6")
    st.markdown(
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:0.85rem;font-weight:600'>"
        f"{note_type.upper()}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:rgba(128,128,128,0.1);padding:14px;border-left:4px solid {color};"
        f"border-radius:4px;margin:8px 0 16px 0'>{note_text}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Modality contribution bars
# ---------------------------------------------------------------------------
def render_modality_scores(scores_dict: dict):
    """Horizontal bars for each modality's contribution score."""
    modalities = list(scores_dict.keys())
    values = list(scores_dict.values())
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    fig = go.Figure(
        go.Bar(
            y=modalities,
            x=values,
            orientation="h",
            marker_color=colors[: len(modalities)],
            text=[f"{v:.1f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Per-Modality Risk Scores",
        xaxis_title="Risk Score",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
        height=280,
        margin=dict(t=40, b=30, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)
