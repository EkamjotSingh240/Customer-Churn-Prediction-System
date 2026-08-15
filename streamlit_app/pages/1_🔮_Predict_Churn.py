import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styling import inject_css, plotly_dark_template, render_sidebar
from utils.model_utils import predict, REQUIRED_COLUMNS, CATEGORY_OPTIONS, risk_band

st.set_page_config(page_title="ChurnScope | Predict", page_icon="🔮", layout="wide")
inject_css()
plotly_dark_template()
render_sidebar()

st.markdown('<div class="hero-title">Predict Churn</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Score one customer with a form, or score a whole batch by uploading a CSV.</div>',
    unsafe_allow_html=True,
)

st.caption(f"Using decision threshold: **{st.session_state.get('churn_threshold', 0.5):.2f}** "
           f"(change this on the ⚙️ Settings page)")

tab_single, tab_batch = st.tabs(["👤 Single Customer", "📁 Batch (CSV Upload)"])

# ============================================================== SINGLE CUSTOMER
with tab_single:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### Customer profile")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tenure = st.number_input("Tenure (months)", 0, 61, 6)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        num_devices = st.slider("Devices registered", 1, 6, 3)
        num_address = st.number_input("Number of addresses", 1, 22, 2)

    with c2:
        login_device = st.selectbox("Preferred login device", CATEGORY_OPTIONS["PreferredLoginDevice"])
        payment_mode = st.selectbox("Preferred payment mode", CATEGORY_OPTIONS["PreferredPaymentMode"])
        gender = st.selectbox("Gender", CATEGORY_OPTIONS["Gender"])
        marital_status = st.selectbox("Marital status", CATEGORY_OPTIONS["MaritalStatus"])

    with c3:
        order_cat = st.selectbox("Preferred order category", CATEGORY_OPTIONS["PreferredOrderCat"])
        satisfaction = st.slider("Satisfaction score", 1, 5, 3)
        complain = st.radio("Logged a complaint recently?", ["No", "Yes"], horizontal=True)
        hours_on_app = st.slider("Avg. hours spent on app", 0, 12, 3)

    with c4:
        warehouse_dist = st.number_input("Warehouse to home distance (km)", 0, 130, 15)
        order_count = st.number_input("Order count (last month)", 0, 20, 3)
        coupon_used = st.number_input("Coupons used (last month)", 0, 20, 1)
        days_since_order = st.number_input("Days since last order", 0, 60, 5)

    cashback = st.slider("Avg. cashback amount (₹)", 0.0, 350.0, 150.0, step=5.0)

    predict_btn = st.button("🔮 Predict churn risk", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        customer = {
            "Tenure": tenure, "PreferredLoginDevice": login_device, "CityTier": city_tier,
            "WarehouseToHome": warehouse_dist, "PreferredPaymentMode": payment_mode, "Gender": gender,
            "HourSpendOnApp": hours_on_app, "NumberOfDeviceRegistered": num_devices,
            "PreferredOrderCat": order_cat, "SatisfactionScore": satisfaction,
            "MaritalStatus": marital_status, "NumberOfAddress": num_address,
            "Complain": 1 if complain == "Yes" else 0, "CouponUsed": coupon_used,
            "OrderCount": order_count, "DaySinceLastOrder": days_since_order,
            "CashbackAmount": cashback,
        }
        try:
            threshold = st.session_state.get("churn_threshold", 0.5)
            result = predict(customer, threshold=threshold).iloc[0]
            prob = float(result["churn_probability"])
            pred = int(result["churn_prediction"])
            band, color = risk_band(prob)

            res_col1, res_col2 = st.columns([1, 1.3])
            with res_col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 42}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#E9ECF1"},
                        "bar": {"color": color},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0, 40], "color": "rgba(74,222,128,0.18)"},
                            {"range": [40, 70], "color": "rgba(255,180,84,0.18)"},
                            {"range": [70, 100], "color": "rgba(255,92,122,0.18)"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": prob * 100},
                    },
                    title={"text": "Churn probability"},
                ))
                fig.update_layout(height=320, margin=dict(t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with res_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                verdict = "⚠️ Likely to churn" if pred == 1 else "✅ Likely to stay"
                st.markdown(f"### {verdict}")
                chip_class = {"High Risk": "chip-bad", "Medium Risk": "chip-warn", "Low Risk": "chip-good"}[band]
                st.markdown(f'<span class="metric-chip {chip_class}">{band}</span>', unsafe_allow_html=True)
                st.markdown("")
                st.markdown("**Suggested action:**")
                if band == "High Risk":
                    st.markdown(
                        "- Prioritize this customer for proactive outreach\n"
                        "- If a complaint is open, resolve it thoroughly, not just quickly\n"
                        "- Consider a personalized retention offer (cashback / coupon)"
                    )
                elif band == "Medium Risk":
                    st.markdown(
                        "- Monitor over the next few orders\n"
                        "- A small engagement nudge (offer, check-in) may help\n"
                        "- Watch tenure and complaint status closely"
                    )
                else:
                    st.markdown(
                        "- No action needed right now\n"
                        "- Keep up standard engagement / loyalty touchpoints"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")

# ============================================================== BATCH
with tab_batch:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### Upload a CSV of customers to score in bulk")
    st.caption(
        "Required columns: " + ", ".join(REQUIRED_COLUMNS)
    )

    template_df = pd.DataFrame([{
        "Tenure": 4, "PreferredLoginDevice": "Mobile Phone", "CityTier": 3, "WarehouseToHome": 6,
        "PreferredPaymentMode": "Debit Card", "Gender": "Female", "HourSpendOnApp": 3,
        "NumberOfDeviceRegistered": 3, "PreferredOrderCat": "Laptop & Accessory", "SatisfactionScore": 2,
        "MaritalStatus": "Single", "NumberOfAddress": 9, "Complain": 1, "CouponUsed": 1,
        "OrderCount": 1, "DaySinceLastOrder": 5, "CashbackAmount": 159.93,
    }])
    st.download_button(
        "⬇️ Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = set(REQUIRED_COLUMNS) - set(batch_df.columns)
            if missing:
                st.error(f"Uploaded file is missing required columns: {sorted(missing)}")
            else:
                threshold = st.session_state.get("churn_threshold", 0.5)
                results = predict(batch_df, threshold=threshold)
                results["Risk Band"] = results["churn_probability"].apply(lambda p: risk_band(p)[0])

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Customers scored", f"{len(results):,}")
                m2.metric("Predicted to churn", f"{int(results['churn_prediction'].sum()):,}")
                m3.metric("Avg. churn probability", f"{results['churn_probability'].mean()*100:.1f}%")

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    band_counts = results["Risk Band"].value_counts().reindex(
                        ["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
                    fig = go.Figure(go.Bar(
                        x=band_counts.index, y=band_counts.values,
                        marker_color=["#4ADE80", "#FFB454", "#FF5C7A"],
                        text=band_counts.values.astype(int), textposition="outside",
                    ))
                    fig.update_layout(title="Customers by risk band", height=320)
                    st.plotly_chart(fig, use_container_width=True)
                with chart_col2:
                    fig2 = go.Figure(go.Histogram(
                        x=results["churn_probability"], nbinsx=20, marker_color="#8B7CFF",
                    ))
                    fig2.update_layout(title="Churn probability distribution", height=320,
                                        xaxis_title="Churn probability", yaxis_title="Customers")
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("##### Scored customers")
                st.dataframe(
                    results.sort_values("churn_probability", ascending=False),
                    use_container_width=True, height=380,
                )

                st.download_button(
                    "⬇️ Download scored results (CSV)",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process file: {exc}")
