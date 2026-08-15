import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styling import inject_css, plotly_dark_template, render_sidebar
from utils.db import load_data
from utils.model_utils import predict, risk_band, REQUIRED_COLUMNS

st.set_page_config(page_title="ChurnScope | Customer Lookup", page_icon="🔎", layout="wide")
inject_css()
plotly_dark_template()
render_sidebar()

st.markdown('<div class="hero-title">Customer Lookup</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Search a real customer from the database and see their full '
    'profile, churn score, and how they compare to the rest of the customer base.</div>',
    unsafe_allow_html=True,
)

df, source, source_error = load_data()

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
search_col1, search_col2 = st.columns([1, 3])
with search_col1:
    customer_id = st.number_input(
        "Customer ID", min_value=int(df["CustomerID"].min()), max_value=int(df["CustomerID"].max()),
        value=int(df["CustomerID"].min()), step=1,
    )
with search_col2:
    st.selectbox(
        "...or pick from the list", df["CustomerID"].sort_values().tolist(),
        key="picked_id",
    )
    if st.button("Use picked ID"):
        customer_id = st.session_state["picked_id"]
st.markdown('</div>', unsafe_allow_html=True)

match = df[df["CustomerID"] == customer_id]

if match.empty:
    st.warning(f"No customer found with ID {customer_id}.")
else:
    row = match.iloc[0]
    threshold = st.session_state.get("churn_threshold", 0.5)
    scored = predict(match, threshold=threshold).iloc[0]
    prob = float(scored["churn_probability"])
    band, color = risk_band(prob)

    top1, top2, top3 = st.columns([1, 1, 1.4])
    with top1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"### Customer #{int(row['CustomerID'])}")
        st.markdown(f"**{row['Gender']} · {row['MaritalStatus']}**")
        st.markdown(f"City tier {int(row['CityTier'])} · {row['PreferredLoginDevice']}")
        actual = "Churned" if int(row["Churn"]) == 1 else "Retained"
        chip = "chip-bad" if actual == "Churned" else "chip-good"
        st.markdown(f'<span class="metric-chip {chip}">Historical outcome: {actual}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with top2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("Predicted churn probability", f"{prob*100:.1f}%")
        chip_class = {"High Risk": "chip-bad", "Medium Risk": "chip-warn", "Low Risk": "chip-good"}[band]
        st.markdown(f'<span class="metric-chip {chip_class}">{band}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with top3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "bgcolor": "rgba(0,0,0,0)",
                   "steps": [{"range": [0, 40], "color": "rgba(74,222,128,0.18)"},
                             {"range": [40, 70], "color": "rgba(255,180,84,0.18)"},
                             {"range": [70, 100], "color": "rgba(255,92,122,0.18)"}]},
        ))
        fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Full profile")
    st.dataframe(match[REQUIRED_COLUMNS].T.rename(columns={match.index[0]: "value"}), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📍 How this customer compares to the population")
    compare_cols = ["Tenure", "DaySinceLastOrder", "CashbackAmount", "SatisfactionScore",
                     "NumberOfDeviceRegistered", "WarehouseToHome"]
    sel = st.selectbox("Compare on", compare_cols)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[sel], nbinsx=25, marker_color="#8B7CFF", opacity=0.75, name="All customers"))
    fig.add_vline(x=row[sel], line_color="#22D3EE", line_width=3,
                  annotation_text=f"This customer: {row[sel]}", annotation_position="top")
    percentile = (df[sel] < row[sel]).mean() * 100
    fig.update_layout(title=f"{sel} distribution — this customer is at the {percentile:.0f}th percentile",
                       height=380, xaxis_title=sel, yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
