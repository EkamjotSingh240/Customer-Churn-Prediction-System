import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.styling import inject_css, plotly_dark_template, render_sidebar
from utils.model_utils import get_model_status
from utils.db import load_data

st.set_page_config(
    page_title="ChurnScope | Home",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
plotly_dark_template()
render_sidebar()

# ---------------------------------------------------------------- HERO
st.markdown('<div class="hero-title">ChurnScope</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Predicting customer churn for an e-commerce platform '
    'using tenure, engagement, and order behavior signals.</div>',
    unsafe_allow_html=True,
)

df, source, source_error = load_data()
model_ok, model_name, model_size = get_model_status()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers on record", f"{len(df):,}")
c2.metric("Historical churn rate", f"{df['Churn'].mean()*100:.1f}%")
c3.metric("Features tracked", f"{df.shape[1]-2}")
c4.metric("Model test ROC-AUC", "0.997")

st.markdown("")

# ---------------------------------------------------------------- ABOUT DATASET / MODEL
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📦 About the dataset")
    st.markdown(
        f"""
The data comes from an e-commerce platform's customer master table, stored in
MySQL as **`ecommerce_cleaned`** and loaded live by this app (currently serving
from **{"MySQL" if source == "mysql" else "bundled CSV fallback"}**).

- **{len(df):,} customers**, **20 columns** (17 predictive features + ID + target)
- Target: **`Churn`** — 1 if the customer churned, 0 if retained
- Cleaned from the raw export: outlier rows removed, missing values imputed
  (group-wise median by churn status, or mode for categoricals), and
  inconsistent category labels standardized (e.g. `CC` → `Credit Card`)

**Feature groups:**
| Group | Columns |
|---|---|
| Engagement | Tenure, HourSpendOnApp, NumberOfDeviceRegistered, PreferredLoginDevice |
| Orders | OrderCount, CouponUsed, DaySinceLastOrder, OrderAmountHikeFromlastYear, PreferredOrderCat |
| Satisfaction | SatisfactionScore, Complain |
| Account / Logistics | CityTier, WarehouseToHome, NumberOfAddress |
| Value | CashbackAmount, PreferredPaymentMode |
| Demographics | Gender, MaritalStatus |
        """
    )
    if source == "csv_fallback" and source_error:
        st.warning(f"MySQL not reachable ({source_error}) — showing the bundled CSV snapshot instead.", icon="⚠️")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 About the model")
    st.markdown(
        """
**XGBoost Classifier** (early-stopped), trained on an 80/20 split after
comparing against Logistic Regression and Random Forest baselines.
        """
    )
    metric_cols = st.columns(2)
    metrics = [("Accuracy", "98.3%"), ("Precision", "95.3%"), ("Recall", "94.0%"),
               ("F1-score", "94.6%"), ("ROC-AUC", "99.7%"), ("Trees", "263")]
    for i, (label, val) in enumerate(metrics):
        metric_cols[i % 2].metric(label, val)

    st.markdown("**Top churn drivers (feature importance):**")
    st.markdown(
        """
1. **Tenure** — newer customers (0-1 months) churn far more than long-tenured ones
2. **Complain** — a logged complaint nearly triples churn risk (31.7% vs 10.9%)
3. **PreferredOrderCat** — Mobile Phone shoppers churn more; Laptop & Accessory less
4. **SatisfactionScore**, **CityTier**, **NumberOfAddress**
5. **MaritalStatus** — Single customers churn over 2x more than Married
        """
    )
    if model_ok:
        st.markdown(
            f'<span class="metric-chip chip-good">● Model loaded — {model_name}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="metric-chip chip-bad">● Model error — {model_name}</span>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- QUICK CHURN SNAPSHOT
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("⚡ Quick churn snapshot")
snap1, snap2 = st.columns(2)

with snap1:
    counts = df["Churn"].value_counts().rename({0: "Retained", 1: "Churned"})
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.55,
        marker=dict(colors=["#4ADE80", "#FF5C7A"]),
        textinfo="label+percent",
    ))
    fig.update_layout(title="Retained vs Churned customers", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with snap2:
    top_driver = (
        df.groupby("Complain", observed=True)["Churn"].mean().mul(100).rename({0: "No complaint", 1: "Complained"})
    )
    fig2 = go.Figure(go.Bar(
        x=top_driver.index.astype(str), y=top_driver.values,
        marker_color=["#22D3EE", "#FF5C7A"],
        text=[f"{v:.1f}%" for v in top_driver.values], textposition="outside",
    ))
    fig2.update_layout(title="Churn rate: complaint vs no complaint", height=340,
                        yaxis_title="Churn rate (%)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🧭 Get started")
st.markdown(
    """
Use the navigation in the sidebar to:
- **🔮 Predict Churn** — score a single customer via a form, or upload a CSV to score a batch
- **📊 EDA Dashboard** — explore the same univariate, bivariate, and correlation charts used in the project notebooks, interactively
    """
)
st.markdown('</div>', unsafe_allow_html=True)
