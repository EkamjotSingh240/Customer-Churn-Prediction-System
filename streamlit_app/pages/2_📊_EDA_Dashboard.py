import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

from utils.styling import inject_css, plotly_dark_template, render_sidebar
from utils.db import load_data

st.set_page_config(page_title="ChurnScope | Dashboard", page_icon="📊", layout="wide")
inject_css()
plotly_dark_template()
render_sidebar()

st.markdown('<div class="hero-title">EDA Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">The same univariate, bivariate, and correlation analysis from the project '
    'notebooks — rebuilt as interactive charts.</div>',
    unsafe_allow_html=True,
)

df, source, source_error = load_data()

NUMERIC_COLS = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear",
                 "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount", "NumberOfAddress"]
CATEGORICAL_COLS = ["Gender", "MaritalStatus", "PreferredLoginDevice", "PreferredPaymentMode",
                     "PreferredOrderCat", "CityTier", "SatisfactionScore", "NumberOfDeviceRegistered",
                     "Complain"]

# ---------------------------------------------------------------- SIDEBAR FILTERS
with st.sidebar:
    st.caption("DASHBOARD FILTERS")
    f_gender = st.multiselect("Gender", sorted(df["Gender"].unique()), default=list(df["Gender"].unique()))
    f_marital = st.multiselect("Marital status", sorted(df["MaritalStatus"].unique()), default=list(df["MaritalStatus"].unique()))
    f_city = st.multiselect("City tier", sorted(df["CityTier"].unique()), default=list(df["CityTier"].unique()))
    st.markdown("---")

fdf = df[
    df["Gender"].isin(f_gender) & df["MaritalStatus"].isin(f_marital) & df["CityTier"].isin(f_city)
].copy()
fdf["ChurnLabel"] = fdf["Churn"].map({0: "Retained", 1: "Churned"})

if fdf.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.caption(f"Showing **{len(fdf):,}** of {len(df):,} customers after filters"
           + (f"  •  ⚠️ data source: CSV fallback ({source_error})" if source == "csv_fallback" else "  •  data source: MySQL"))

tab_overview, tab_univariate, tab_numeric, tab_categorical, tab_corr, tab_deep = st.tabs(
    ["🎯 Target Overview", "📈 Univariate", "🔢 Numeric vs Churn", "🏷️ Categorical vs Churn",
     "🔗 Correlations", "🔍 Deep Dive"]
)

# ============================================================== TARGET OVERVIEW
with tab_overview:
    c1, c2 = st.columns(2)
    counts = fdf["ChurnLabel"].value_counts()
    with c1:
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map={"Retained": "#4ADE80", "Churned": "#FF5C7A"},
                     labels={"x": "Churn", "y": "Customers"})
        fig.update_layout(title="Churn class counts", showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(values=counts.values, names=counts.index, hole=0.45,
                     color=counts.index, color_discrete_map={"Retained": "#4ADE80", "Churned": "#FF5C7A"})
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(title="Churn class proportion", height=380)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== UNIVARIATE
with tab_univariate:
    st.markdown("##### Distribution explorer")
    u1, u2 = st.columns(2)
    with u1:
        num_col = st.selectbox("Numeric column", NUMERIC_COLS, key="uni_num")
        hist_data = [fdf[num_col].dropna().values]
        fig = ff.create_distplot(hist_data, [num_col], colors=["#8B7CFF"], show_rug=False)
        fig.update_layout(title=f"{num_col} — distribution with density curve", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Median: **{fdf[num_col].median():.1f}**  •  Mean: **{fdf[num_col].mean():.1f}**")
    with u2:
        cat_col = st.selectbox("Categorical column", CATEGORICAL_COLS, key="uni_cat")
        vc = fdf[cat_col].astype(str).value_counts()
        fig = px.bar(x=vc.index, y=vc.values, color_discrete_sequence=["#22D3EE"],
                     labels={"x": cat_col, "y": "Customers"})
        fig.update_layout(title=f"{cat_col} — value counts", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== NUMERIC VS CHURN
with tab_numeric:
    n1, n2 = st.columns([1, 3])
    with n1:
        sel_num = st.selectbox("Numeric feature", NUMERIC_COLS, key="numvschurn")
        chart_type = st.radio("Chart type", ["Box plot", "Violin plot", "KDE (density)"], key="numvschurn_type")
    with n2:
        if chart_type == "Box plot":
            fig = px.box(fdf, x="ChurnLabel", y=sel_num, color="ChurnLabel",
                         color_discrete_map={"Retained": "#4ADE80", "Churned": "#FF5C7A"})
        elif chart_type == "Violin plot":
            fig = px.violin(fdf, x="ChurnLabel", y=sel_num, color="ChurnLabel", box=True, points=False,
                             color_discrete_map={"Retained": "#4ADE80", "Churned": "#FF5C7A"})
        else:
            fig = ff.create_distplot(
                [fdf[fdf.Churn == 0][sel_num].dropna(), fdf[fdf.Churn == 1][sel_num].dropna()],
                ["Retained", "Churned"], colors=["#4ADE80", "#FF5C7A"], show_rug=False,
            )
        fig.update_layout(title=f"{sel_num} vs Churn ({chart_type})", height=440)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== CATEGORICAL VS CHURN
with tab_categorical:
    c1, c2 = st.columns([1, 3])
    with c1:
        sel_cat = st.selectbox("Categorical feature", CATEGORICAL_COLS, key="catvschurn")
        cat_chart = st.radio("Chart type", ["Churn rate (%)", "Grouped counts", "Crosstab heatmap (%)"], key="catvschurn_type")
    with c2:
        if cat_chart == "Churn rate (%)":
            rate = fdf.groupby(sel_cat, observed=True)["Churn"].mean().mul(100).sort_values(ascending=False)
            fig = px.bar(x=rate.index.astype(str), y=rate.values, color_discrete_sequence=["#8B7CFF"],
                         text=[f"{v:.1f}%" for v in rate.values],
                         labels={"x": sel_cat, "y": "Churn rate (%)"})
            fig.update_traces(textposition="outside")
            fig.update_layout(title=f"Churn rate by {sel_cat}", height=440)
        elif cat_chart == "Grouped counts":
            fig = px.histogram(fdf, x=sel_cat, color="ChurnLabel", barmode="group",
                                color_discrete_map={"Retained": "#4ADE80", "Churned": "#FF5C7A"})
            fig.update_layout(title=f"{sel_cat} — counts by churn", height=440)
        else:
            crosstab = pd.crosstab(fdf[sel_cat], fdf["ChurnLabel"], normalize="index") * 100
            fig = px.imshow(crosstab, text_auto=".1f", color_continuous_scale="YlOrRd", aspect="auto")
            fig.update_layout(title=f"{sel_cat} vs Churn (row %)", height=440)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== CORRELATIONS
with tab_corr:
    corr_cols = NUMERIC_COLS + ["Churn"]
    corr_df = fdf[corr_cols].copy()
    corr_matrix = corr_df.corr()

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(title="Correlation heatmap", height=480)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        corr_with_churn = corr_matrix["Churn"].drop("Churn").sort_values(key=np.abs, ascending=True)
        colors = ["#FF5C7A" if v < 0 else "#4ADE80" for v in corr_with_churn.values]
        fig2 = go.Figure(go.Bar(x=corr_with_churn.values, y=corr_with_churn.index, orientation="h",
                                 marker_color=colors,
                                 text=[f"{v:.2f}" for v in corr_with_churn.values], textposition="outside"))
        fig2.update_layout(title="Feature correlation with Churn (ranked)", height=480,
                            xaxis_title="Correlation")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Satisfaction × Complaint — churn rate heatmap")
    pivot = fdf.pivot_table(values="Churn", index="SatisfactionScore", columns="Complain",
                             aggfunc="mean", observed=True) * 100
    pivot.columns = [("No complaint" if c == 0 else "Complained") for c in pivot.columns]
    fig3 = px.imshow(pivot, text_auto=".1f", color_continuous_scale="YlOrRd", aspect="auto")
    fig3.update_layout(title="Churn rate (%): Satisfaction Score × Complaint", height=420)
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================== DEEP DIVE
with tab_deep:
    st.markdown("##### Pairwise relationships — key drivers")
    key_cols = ["Tenure", "DaySinceLastOrder", "WarehouseToHome", "CashbackAmount"]
    fig = px.scatter_matrix(fdf, dimensions=key_cols, color="ChurnLabel",
                             color_discrete_map={"Retained": "#4ADE80", "Churned": "#FF5C7A"},
                             opacity=0.55)
    fig.update_layout(title="Key features pairplot (Tenure, DaySinceLastOrder, WarehouseToHome, CashbackAmount)",
                       height=650)
    fig.update_traces(diagonal_visible=False, marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Tenure × Complaint interaction")
    d1, d2 = st.columns(2)
    with d1:
        fig = px.box(fdf, x="Complain", y="Tenure", color="ChurnLabel",
                     color_discrete_map={"Retained": "#4ADE80", "Churned": "#FF5C7A"})
        fig.update_layout(title="Tenure by Complaint, split by Churn", height=420,
                           xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["No complaint", "Complained"]))
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        bucketed = fdf.copy()
        bucketed["TenureBucket"] = pd.cut(bucketed["Tenure"], bins=[-1, 3, 6, 12, 24, 100],
                                           labels=["0-3", "4-6", "7-12", "13-24", "25+"])
        pv = bucketed.pivot_table(values="Churn", index="TenureBucket", columns="Complain",
                                   aggfunc="mean", observed=True) * 100
        pv.columns = [("No complaint" if c == 0 else "Complained") for c in pv.columns]
        fig = px.bar(pv, barmode="group", color_discrete_sequence=["#22D3EE", "#FF5C7A"])
        fig.update_layout(title="Churn rate (%) by Tenure bucket, split by Complaint", height=420,
                           xaxis_title="Tenure (months)", yaxis_title="Churn rate (%)")
        st.plotly_chart(fig, use_container_width=True)
