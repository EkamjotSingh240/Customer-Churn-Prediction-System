import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.styling import inject_css, plotly_dark_template, render_sidebar
from utils.db import load_data
from utils.model_utils import (
    get_global_importance, compute_shap_for_row, predict, risk_band, REQUIRED_COLUMNS,
)

st.set_page_config(page_title="ChurnScope | Explainability", page_icon="🧠", layout="wide")
inject_css()
plotly_dark_template()
render_sidebar()

st.markdown('<div class="hero-title">Model Explainability</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">See which features drive the model globally, and why it flagged '
    'a specific customer.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- GLOBAL IMPORTANCE
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🌐 Global feature importance")
try:
    importance = get_global_importance().head(15).sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=importance.values, y=importance.index, orientation="h",
        marker_color="#8B7CFF",
    ))
    fig.update_layout(title="Top 15 features by model importance", height=500,
                       xaxis_title="Importance", margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This reflects how much each (post-encoding) feature contributes to the trained "
        "XGBoost model's decisions overall — consistent with the notebook's feature-importance "
        "and correlation findings (Tenure and Complain dominate)."
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Couldn't compute global importance: {exc}")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- PER-CUSTOMER SHAP
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🔬 Explain a single prediction")
st.caption("Pick a real customer from the loaded dataset to see exactly what pushed their score up or down.")

df, source, source_error = load_data()
customer_id = st.selectbox("Customer ID", df["CustomerID"].sort_values().tolist())
row = df[df["CustomerID"] == customer_id]

if st.button("Explain this customer"):
    try:
        threshold = st.session_state.get("churn_threshold", 0.5)
        pred_row = predict(row, threshold=threshold).iloc[0]
        prob = float(pred_row["churn_probability"])
        band, color = risk_band(prob)

        sv, base_value, names, values = compute_shap_for_row(row)
        contrib = pd.DataFrame({"feature": names, "shap_value": sv, "value": values})
        contrib["abs"] = contrib["shap_value"].abs()
        contrib = contrib.sort_values("abs", ascending=False).head(12).sort_values("shap_value")

        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.metric("Predicted churn probability", f"{prob*100:.1f}%")
            chip_class = {"High Risk": "chip-bad", "Medium Risk": "chip-warn", "Low Risk": "chip-good"}[band]
            st.markdown(f'<span class="metric-chip {chip_class}">{band}</span>', unsafe_allow_html=True)
            st.markdown("")
            st.caption(f"Base rate (model's average prediction): {base_value:.2f} log-odds")
            st.dataframe(row[REQUIRED_COLUMNS].T.rename(columns={row.index[0]: "value"}),
                         use_container_width=True, height=300)

        with c2:
            colors = ["#FF5C7A" if v > 0 else "#4ADE80" for v in contrib["shap_value"]]
            fig = go.Figure(go.Bar(
                x=contrib["shap_value"], y=contrib["feature"], orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}" for v in contrib["shap_value"]], textposition="outside",
            ))
            fig.update_layout(
                title="Top feature contributions (SHAP) — red pushes toward churn, green toward staying",
                height=480, xaxis_title="Impact on churn log-odds",
            )
            st.plotly_chart(fig, use_container_width=True)
    except ImportError as exc:
        st.warning(str(exc), icon="⚠️")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't explain this customer: {exc}")
st.markdown('</div>', unsafe_allow_html=True)
