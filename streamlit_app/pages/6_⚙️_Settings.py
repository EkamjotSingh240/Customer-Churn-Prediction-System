import importlib.metadata as importlib_metadata

import streamlit as st

from utils.styling import inject_css, plotly_dark_template, render_sidebar
from utils.db import load_data, get_connection_status, _get_db_config
from utils.model_utils import get_model_status, MODEL_PATH

st.set_page_config(page_title="ChurnScope | Settings", page_icon="⚙️", layout="wide")
inject_css()
plotly_dark_template()
render_sidebar()

st.markdown('<div class="hero-title">Settings</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">App-wide options: decision threshold, data source, and cache.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- DECISION THRESHOLD
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🎚️ Decision threshold")
st.caption(
    "The probability above which a customer is classified as 'will churn'. Used by the "
    "Predict and Customer Lookup pages. Lower it to catch more at-risk "
    "customers (more false alarms); raise it to be more conservative."
)
current = st.session_state.get("churn_threshold", 0.5)
new_threshold = st.slider("Churn classification threshold", 0.05, 0.95, current, 0.05)
if new_threshold != current:
    st.session_state["churn_threshold"] = new_threshold
    st.success(f"Threshold updated to {new_threshold:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- DATA SOURCE
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🗄️ Data source")

cfg = _get_db_config()
db_ok, db_info = get_connection_status()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**MySQL configuration**")
    if cfg is None:
        st.warning("No `.streamlit/secrets.toml` found — copy `secrets.toml.example` and fill in your credentials.")
    else:
        st.markdown(
            f"""
- Host: `{cfg['host']}:{cfg['port']}`
- Database: `{cfg['database']}`
- Table: `{cfg['table']}`
- User: `{cfg['user']}`
- Password: `{'•' * len(cfg['password']) if cfg['password'] else '(empty)'}`
            """
        )
    status_chip = "chip-good" if db_ok else "chip-bad"
    status_text = "Connected" if db_ok else f"Unreachable — {db_info}"
    st.markdown(f'<span class="metric-chip {status_chip}">{status_text}</span>', unsafe_allow_html=True)

with col2:
    st.markdown("**Currently loaded dataset**")
    df, source, source_error = load_data()
    st.markdown(
        f"""
- Source in use: **{"MySQL" if source == "mysql" else "CSV fallback"}**
- Rows loaded: **{len(df):,}**
- Columns: **{df.shape[1]}**
        """
    )
    if source_error:
        st.caption(f"Fallback reason: {source_error}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- MODEL INFO
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🤖 Model file")
model_ok, model_info, size_kb = get_model_status()
st.markdown(f"- Path: `{MODEL_PATH}`")
if model_ok:
    st.markdown(f"- Status: **loaded** ({model_info}, {size_kb:.0f} KB)")
else:
    st.markdown(f"- Status: **error** — {model_info}")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- ENVIRONMENT
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🧩 Environment")
pkgs = ["streamlit", "pandas", "numpy", "scikit-learn", "xgboost", "plotly", "mysql-connector-python", "sqlalchemy", "shap"]
rows = []
for pkg in pkgs:
    try:
        rows.append((pkg, importlib_metadata.version(pkg)))
    except importlib_metadata.PackageNotFoundError:
        rows.append((pkg, "not installed"))
st.table({"package": [p for p, _ in rows], "version": [v for _, v in rows]})
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- CACHE
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🧹 Cache")
st.caption("Data and model are cached for performance. Clear the cache after updating the database or replacing the model file.")
if st.button("Clear cache & reload"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
