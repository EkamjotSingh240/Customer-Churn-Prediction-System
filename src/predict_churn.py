"""
predict_churn.py

Loads the trained churn model pipeline and predicts churn probability
for new customer records.

Usage:
    from predict_churn import predict_churn

    new_customer = {
        'Tenure': 2,
        'PreferredLoginDevice': 'Mobile Phone',
        'CityTier': 1,
        'WarehouseToHome': 15,
        'PreferredPaymentMode': 'Credit Card',
        'Gender': 'Male',
        'HourSpendOnApp': 3,
        'NumberOfDeviceRegistered': 4,
        'PreferredOrderCat': 'Mobile Phone',
        'SatisfactionScore': 3,
        'MaritalStatus': 'Single',
        'NumberOfAddress': 5,
        'Complain': 1,
        'CouponUsed': 1,
        'OrderCount': 2,
        'DaySinceLastOrder': 3,
        'CashbackAmount': 150.0
    }

    result = predict_churn(new_customer)
    print(result)
    # {'churn_prediction': 1, 'churn_probability': 0.87}

Can also be run directly from the command line for a quick demo:
    python predict_churn.py
"""

import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'final_churn_model_pipeline.pkl'

# Columns the pipeline expects, in the same form used during training
# (CustomerID and OrderAmountHikeFromlastYear were dropped before training)
REQUIRED_COLUMNS = [
    'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
    'PreferredPaymentMode', 'Gender', 'HourSpendOnApp',
    'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore',
    'MaritalStatus', 'NumberOfAddress', 'Complain', 'CouponUsed',
    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
]

_pipeline = None


def _load_pipeline():
    """Load the trained pipeline once and cache it for reuse."""
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Make sure 'models/final_churn_model_pipeline.pkl' exists at the "
                "project root, as a sibling folder to 'src/' (where this script lives)."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_churn(customer_data, threshold=0.5):
    """
    Predict churn for one or more customers.

    Parameters
    ----------
    customer_data : dict or list of dict or pandas.DataFrame
        Raw customer feature values. Must contain all columns in
        REQUIRED_COLUMNS. A single dict is treated as one customer.
    threshold : float, default 0.5
        Probability cutoff above which a customer is classified as churn (1).

    Returns
    -------
    dict (if single customer) or list of dict (if multiple customers)
        Each result contains 'churn_prediction' (0/1) and
        'churn_probability' (float, probability of churn).
    """
    pipeline = _load_pipeline()

    # Normalize input into a DataFrame
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
        single_input = True
    elif isinstance(customer_data, list):
        df = pd.DataFrame(customer_data)
        single_input = False
    elif isinstance(customer_data, pd.DataFrame):
        df = customer_data.copy()
        single_input = False
    else:
        raise TypeError(
            "customer_data must be a dict, list of dicts, or pandas DataFrame"
        )

    # Validate required columns are present
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df = df[REQUIRED_COLUMNS]

    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = [
        {'churn_prediction': int(pred), 'churn_probability': round(float(prob), 4)}
        for pred, prob in zip(predictions, probabilities)
    ]

    return results[0] if single_input else results


if __name__ == '__main__':
    # Quick demo with a sample customer
    sample_customer = {
        'Tenure': 2,
        'PreferredLoginDevice': 'Mobile Phone',
        'CityTier': 1,
        'WarehouseToHome': 15,
        'PreferredPaymentMode': 'Credit Card',
        'Gender': 'Male',
        'HourSpendOnApp': 3,
        'NumberOfDeviceRegistered': 4,
        'PreferredOrderCat': 'Mobile Phone',
        'SatisfactionScore': 3,
        'MaritalStatus': 'Single',
        'NumberOfAddress': 5,
        'Complain': 1,
        'CouponUsed': 1,
        'OrderCount': 2,
        'DaySinceLastOrder': 3,
        'CashbackAmount': 150.0
    }

    result = predict_churn(sample_customer)
    print("Sample prediction:")
    print(f"  Churn prediction:   {result['churn_prediction']} "
          f"({'Will churn' if result['churn_prediction'] == 1 else 'Will stay'})")
    print(f"  Churn probability:  {result['churn_probability']:.2%}")