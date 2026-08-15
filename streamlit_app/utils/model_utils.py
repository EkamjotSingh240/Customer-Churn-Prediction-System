"""
model_utils.py
Loads the trained XGBoost churn pipeline (preprocessing + model bundled
together) and exposes helpers for single-customer and batch prediction.
Mirrors the logic in src/predict_churn.py from the original project.

The model file lives in the project's shared models/ folder (a sibling of
streamlit_app/, not a copy inside it) - see MODEL_PATH below.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# streamlit_app/utils/model_utils.py -> streamlit_app -> project root -> models/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_churn_model_pipeline.pkl"

REQUIRED_COLUMNS = [
    "Tenure", "PreferredLoginDevice", "CityTier", "WarehouseToHome",
    "PreferredPaymentMode", "Gender", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "PreferredOrderCat", "SatisfactionScore",
    "MaritalStatus", "NumberOfAddress", "Complain", "CouponUsed",
    "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]

# Options for categorical inputs, taken from the cleaned dataset's known values
CATEGORY_OPTIONS = {
    "PreferredLoginDevice": ["Mobile Phone", "Computer"],
    "PreferredPaymentMode": ["Debit Card", "Credit Card", "UPI", "Cash on Delivery", "E wallet"],
    "Gender": ["Male", "Female"],
    "PreferredOrderCat": ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
}


@st.cache_resource(show_spinner="Loading churn model...")
def load_model():
    """Load and cache the trained pipeline. Raises if the file is missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def get_model_status():
    """Non-raising status check for the sidebar."""
    if not MODEL_PATH.exists():
        return False, "File not found", None
    try:
        pipeline = load_model()
        size_kb = MODEL_PATH.stat().st_size / 1024
        model_name = type(pipeline.named_steps.get("model", pipeline)).__name__ if hasattr(pipeline, "named_steps") else type(pipeline).__name__
        return True, model_name, size_kb
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", None


def predict(customer_data, threshold=0.5):
    """
    Predict churn for one or more customers.

    Parameters
    ----------
    customer_data : dict | list[dict] | pd.DataFrame
    threshold : float

    Returns
    -------
    pd.DataFrame with columns: churn_prediction, churn_probability
    (plus any original columns passed in, preserved for batch display)
    """
    pipeline = load_model()

    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    elif isinstance(customer_data, list):
        df = pd.DataFrame(customer_data)
    elif isinstance(customer_data, pd.DataFrame):
        df = customer_data.copy()
    else:
        raise TypeError("customer_data must be a dict, list of dicts, or DataFrame")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    model_input = df[REQUIRED_COLUMNS]
    probabilities = pipeline.predict_proba(model_input)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = df.copy()
    result["churn_probability"] = probabilities.round(4)
    result["churn_prediction"] = predictions
    return result


def risk_band(probability):
    """Map a churn probability to a human-readable risk band + color."""
    if probability >= 0.7:
        return "High Risk", "#FF5C7A"
    elif probability >= 0.4:
        return "Medium Risk", "#FFB454"
    else:
        return "Low Risk", "#4ADE80"


# ---------------------------------------------------------------- Explainability helpers

def _get_estimator(pipeline):
    """Return the final classifier step of the pipeline (assumed to be the last step)."""
    if hasattr(pipeline, "steps"):
        return pipeline.steps[-1][1]
    return pipeline


def _get_preprocessor(pipeline):
    """Return everything before the final step (assumed to be preprocessing)."""
    if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
        return pipeline[:-1]
    return None


@st.cache_data(show_spinner=False)
def get_feature_names(_pipeline=None):
    """Feature names after preprocessing, for importance/SHAP charts. Cached by call site."""
    pipeline = _pipeline or load_model()
    pre = _get_preprocessor(pipeline)
    if pre is not None and hasattr(pre, "get_feature_names_out"):
        try:
            return list(pre.get_feature_names_out())
        except Exception:  # noqa: BLE001
            pass
    return REQUIRED_COLUMNS


@st.cache_data(show_spinner=False)
def get_global_importance():
    """Model's built-in feature_importances_, mapped to readable feature names."""
    pipeline = load_model()
    estimator = _get_estimator(pipeline)
    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError("Underlying model has no feature_importances_ attribute.")
    names = get_feature_names(pipeline)
    importances = estimator.feature_importances_
    n = min(len(names), len(importances))
    series = pd.Series(importances[:n], index=names[:n]).sort_values(ascending=False)
    return series


@st.cache_resource(show_spinner="Preparing explainability engine...")
def _get_shap_explainer():
    import shap
    pipeline = load_model()
    estimator = _get_estimator(pipeline)
    return shap.TreeExplainer(estimator)


def compute_shap_for_row(row_df):
    """
    Compute SHAP values for a single-row DataFrame (raw, unprocessed columns).
    Returns (shap_values: np.ndarray, base_value: float, feature_names: list, feature_values: list)
    Raises ImportError with a friendly message if the `shap` package isn't installed.
    """
    try:
        explainer = _get_shap_explainer()
    except ImportError as exc:
        raise ImportError(
            "The 'shap' package isn't installed. Add `shap` to requirements.txt "
            "and `pip install shap` to enable this page."
        ) from exc

    pipeline = load_model()
    pre = _get_preprocessor(pipeline)
    raw_row = row_df[REQUIRED_COLUMNS]

    if pre is not None:
        X_trans = pre.transform(raw_row)
    else:
        X_trans = raw_row.values
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    shap_values = explainer.shap_values(X_trans)
    base_value = explainer.expected_value

    # Handle both list-per-class and single-array SHAP outputs
    if isinstance(shap_values, list):
        sv = shap_values[-1][0]
        bv = base_value[-1] if isinstance(base_value, (list,)) else base_value
    else:
        sv = shap_values[0]
        bv = base_value[-1] if hasattr(base_value, "__len__") and len(base_value) > 1 else base_value

    names = get_feature_names(pipeline)
    n = min(len(names), len(sv))
    return sv[:n], float(bv) if not hasattr(bv, "__len__") else float(bv[0]), names[:n], list(X_trans[0][:n])
