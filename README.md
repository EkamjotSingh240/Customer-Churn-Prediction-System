# E-Commerce Customer Churn Prediction

A complete data science project: cleaning a raw e-commerce customer dataset, exploring churn patterns, and building a predictive model to flag at-risk customers for retention efforts.

**Final model**: XGBoost (early stopping) — 98.3% accuracy, 94.6% F1, 99.7% ROC-AUC on held-out test data.

See [`reports/final_project_report.md`](reports/final_project_report.md) for the full write-up of findings, model performance, and business recommendations.

## Project Structure

```
├── notebooks/
│   ├── 01_data_cleaning.ipynb             # Raw data -> cleaned dataset (stored in MySQL)
│   ├── 02_eda.ipynb                       # Univariate, bivariate, multivariate analysis
│   ├── 03_feature_engineering_and_model_selection.ipynb     # Logistic Regression, Random Forest, XGBoost tuning
│   ├── 04_final_model_training.ipynb      # Final model training, evaluation
│   └── 05_interpretation_and_report.ipynb # Feature importance
│
├── models/
│   └── final_churn_model_pipeline.pkl  # Trained pipeline (preprocessing + model)
│
├── reports/
│   ├── final_project_report.md         # Full findings, results, and recommendations
│   └── figures/                        # All saved charts from EDA and modeling
│
├── src/
│   └── predict_churn.py                # Standalone script to predict churn on new data
│
├── dataset/
│   └── E_Commerce_Dataset.xlsx         # Original raw dataset
|
|
├── sql/
│   └── ecommerce_database.sql
│
├── config.py                           # MySQL connection settings (not committed)
├── requirements.txt
├── LICENCE
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `config.py` file in the project root with your MySQL credentials:
   ```python
   DB_CONFIG = {
       'host': 'localhost',
       'user': 'your_username',
       'password': 'your_password',
       'database': 'your_database_name'
   }
   ```

3. Run the notebooks in order (01 -> 04) to reproduce the full pipeline, or skip straight to using the saved model (see below).

## Using the Trained Model

To predict churn on new customer data without running any notebooks:

```python
from src.predict_churn import predict_churn

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
```

## Key Findings (Summary)

- **Tenure** and **Complain** are the two strongest churn drivers, confirmed by both exploratory analysis and the trained model's feature importance.
- Customers are most at risk of churning in their **first 1-2 months**.
- A logged **complaint nearly triples churn risk**, regardless of the customer's satisfaction score.
- Full details and business recommendations: [`reports/final_project_report.md`](reports/final_project_report.md)

## Tech Stack

Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn, MySQL