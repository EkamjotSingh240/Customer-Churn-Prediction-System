"""
db.py
Loads the customer dataset from the MySQL `ecommerce_cleaned` table.

Connection settings are read from Streamlit secrets (.streamlit/secrets.toml,
see secrets.toml.example). If the database can't be reached, the app falls
back to the CSV snapshot in the project's shared sql/ folder (a sibling of
streamlit_app/, not a copy inside it) so the demo still works, and shows a
clear warning in the sidebar rather than failing silently.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# streamlit_app/utils/db.py -> streamlit_app -> project root -> sql/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FALLBACK_CSV = PROJECT_ROOT / "sql" / "ecommerce_cleaned.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "Churn", "Tenure", "PreferredLoginDevice", "CityTier",
    "WarehouseToHome", "PreferredPaymentMode", "Gender", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "PreferredOrderCat", "SatisfactionScore",
    "MaritalStatus", "NumberOfAddress", "Complain", "OrderAmountHikeFromlastYear",
    "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]


def _get_db_config():
    """Read MySQL settings from st.secrets. Returns None if not configured."""
    if "mysql" not in st.secrets:
        return None
    cfg = st.secrets["mysql"]
    return {
        "host": cfg.get("host", "localhost"),
        "port": int(cfg.get("port", 3306)),
        "user": cfg.get("user", "root"),
        "password": cfg.get("password", ""),
        "database": cfg.get("database", "ecommerce_db"),
        "table": cfg.get("table", "ecommerce_cleaned"),
    }


@st.cache_resource(show_spinner=False)
def _get_engine(host, port, user, password, database):
    """Cached SQLAlchemy engine so we don't reconnect on every rerun."""
    from sqlalchemy import create_engine
    url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(show_spinner="Loading customer data from MySQL...", ttl=600)
def load_data():
    """
    Returns (df, source, error) where:
      - df: pandas DataFrame of the ecommerce_cleaned table
      - source: "mysql" or "csv_fallback"
      - error: None, or a short string describing why MySQL wasn't used
    """
    cfg = _get_db_config()

    if cfg is not None:
        try:
            engine = _get_engine(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["database"])
            query = f"SELECT * FROM {cfg['table']}"
            df = pd.read_sql(query, engine)
            df["Churn"] = df["Churn"].astype(int)
            df["Complain"] = df["Complain"].astype(int)
            return df, "mysql", None
        except Exception as exc:  # noqa: BLE001 - surface any connector error to the UI
            error_msg = f"{type(exc).__name__}: {exc}"
            if FALLBACK_CSV.exists():
                df = pd.read_csv(FALLBACK_CSV)
                return df, "csv_fallback", error_msg
            raise
    else:
        if FALLBACK_CSV.exists():
            df = pd.read_csv(FALLBACK_CSV)
            return df, "csv_fallback", "No MySQL credentials found in .streamlit/secrets.toml"
        raise FileNotFoundError(
            "No MySQL secrets configured and no fallback CSV found. "
            "Add .streamlit/secrets.toml (see secrets.toml.example)."
        )


def get_connection_status():
    """Lightweight check used by the sidebar - does not load full data."""
    cfg = _get_db_config()
    if cfg is None:
        return False, "No secrets.toml found"
    try:
        engine = _get_engine(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["database"])
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True, f"{cfg['database']}.{cfg['table']}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"
