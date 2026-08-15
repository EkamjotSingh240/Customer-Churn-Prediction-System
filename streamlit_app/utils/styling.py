"""
styling.py
Custom CSS for the dark theme "glass card" look, a matching Plotly template,
and the shared sidebar block (model status, DB status, branding) that every
page renders so navigation + status stay consistent across the app.
"""

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

ACCENT = "#8B7CFF"
ACCENT_2 = "#22D3EE"
BG = "#0B0E14"
CARD_BG = "#141A24"
GOOD = "#4ADE80"
WARN = "#FFB454"
BAD = "#FF5C7A"

CUSTOM_CSS = f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .stApp {{
        background: radial-gradient(circle at 15% 0%, #161B2C 0%, {BG} 45%);
    }}

    h1, h2, h3 {{
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    .hero-title {{
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }}
    .hero-subtitle {{
        color: #9AA4B2;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }}

    .glass-card {{
        background: linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }}

    .metric-chip {{
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }}
    .chip-good {{ background: rgba(74,222,128,0.15); color: {GOOD}; }}
    .chip-warn {{ background: rgba(255,180,84,0.15); color: {WARN}; }}
    .chip-bad  {{ background: rgba(255,92,122,0.15); color: {BAD}; }}
    .chip-accent {{ background: rgba(139,124,255,0.18); color: {ACCENT}; }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.9rem 1rem 0.6rem 1rem;
    }}

    .stButton > button {{
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
        color: #0B0E14;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(139,124,255,0.35);
    }}

    section[data-testid="stSidebar"] {{
        background: #0D111A;
        border-right: 1px solid rgba(255,255,255,0.06);
    }}

    .sidebar-status-row {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }}
    .dot {{
        height: 9px; width: 9px; border-radius: 50%; flex-shrink: 0;
        box-shadow: 0 0 6px currentColor;
    }}
    .dot-good {{ background: {GOOD}; color: {GOOD}; }}
    .dot-bad {{ background: {BAD}; color: {BAD}; }}

    hr {{ border-color: rgba(255,255,255,0.08); }}
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def plotly_dark_template():
    """Register + return a Plotly template matching the app's dark theme."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E9ECF1", family="sans-serif"),
        colorway=[ACCENT, ACCENT_2, "#FF9F7A", "#4ADE80", "#FFB454", "#FF5C7A", "#C084FC", "#60A5FA"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, l=10, r=10, b=10),
    )
    pio.templates["churn_dark"] = template
    pio.templates.default = "churn_dark"
    return template


def render_sidebar():
    """Shared sidebar: branding + live model/data status. Call on every page."""
    from utils.model_utils import get_model_status
    from utils.db import get_connection_status

    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.2rem;">
                <div style="font-size:1.6rem;">🛒</div>
                <div style="font-weight:800;font-size:1.15rem;
                    background:linear-gradient(90deg,{ACCENT},{ACCENT_2});
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    ChurnScope
                </div>
            </div>
            <div style="color:#8891A3;font-size:0.8rem;margin-bottom:0.9rem;">
                E-Commerce Customer Churn Intelligence
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("SYSTEM STATUS")

        model_ok, model_info, size_kb = get_model_status()
        dot = "dot-good" if model_ok else "dot-bad"
        label = f"Model loaded ({model_info}, {size_kb:.0f} KB)" if model_ok and size_kb else f"Model error: {model_info}"
        st.markdown(
            f'<div class="sidebar-status-row"><span class="dot {dot}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

        db_ok, db_info = get_connection_status()
        dot = "dot-good" if db_ok else "dot-bad"
        label = f"MySQL connected ({db_info})" if db_ok else f"MySQL unavailable — using CSV fallback"
        st.markdown(
            f'<div class="sidebar-status-row"><span class="dot {dot}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
