# E-Commerce Customer Churn Prediction

A complete data science project: cleaning a raw e-commerce customer dataset, exploring churn
patterns, building a predictive model to flag at-risk customers for retention efforts, and
serving it all through **ChurnScope** — a dark-themed, interactive Streamlit app.

**Final model**: XGBoost (early stopping) — 98.3% accuracy, 94.6% F1, 99.7% ROC-AUC on held-out
test data.

See [`reports/final_project_report.md`](reports/final_project_report.md) for the full write-up
of findings, model performance, and business recommendations.

## Key Findings (Summary)

- **Tenure** and **Complain** are the two strongest churn drivers, confirmed by both exploratory
  analysis and the trained model's feature importance.
- Customers are most at risk of churning in their **first 1-2 months**.
- A logged **complaint nearly triples churn risk**, regardless of the customer's satisfaction score.
- Full details and business recommendations: [`reports/final_project_report.md`](reports/final_project_report.md)

## Tech Stack

Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn, MySQL, Streamlit, Plotly, SHAP

---

## Project Structure

```
├── notebooks/
│   ├── 01_data_cleaning.ipynb             # Raw data -> cleaned dataset (stored in MySQL)
│   ├── 02_eda.ipynb                       # Univariate, bivariate, multivariate analysis
│   ├── 03_feature_engineering_and_model_selection.ipynb  # Logistic Regression, Random Forest, XGBoost tuning
│   ├── 04_final_model_training.ipynb      # Final model training, evaluation
│   ├── 05_interpretation_and_report.ipynb # Feature importance
|   └── config.py                              # MySQL connection settings (not committed)
│
├── models/
│   └── final_churn_model_pipeline.pkl     # Trained pipeline (preprocessing + model)
│
├── reports/
│   ├── final_project_report.md            # Full findings, results, and recommendations
│   └── figures/                           # All saved charts from EDA and modeling
│
├── src/
│   └── predict_churn.py                   # Standalone script to predict churn on new data
│
├── dataset/
│   └── E_Commerce_Dataset.xlsx            # Original raw dataset
│
├── sql/
│   ├── ecommerce_database.sql             # Schema for the `ecommerce_cleaned` MySQL table
│   └── ecommerce_cleaned.csv              # CSV snapshot — used by the app as a MySQL fallback
│
├── streamlit_app/                         # ChurnScope — the interactive web app (see below)
│   ├── app.py                             # Reads the model from ../models/ and the CSV
│   ├── pages/                             #   fallback from ../sql/ — no local copies
│   ├── utils/
│   └── .streamlit/
│
├── requirements.txt
├── LICENCE
└── README.md
```

## Setup — Notebooks & Standalone Script

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

3. Run the notebooks in order (01 -> 04) to reproduce the full pipeline, or skip straight to
   using the saved model (see below).

### Using the Trained Model Directly

To predict churn on new customer data without running any notebooks or the app:

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

---

## ChurnScope — The Streamlit App

A dark-themed, interactive front end built on top of this project: predicts churn with the
trained XGBoost pipeline and explores the dataset with the same charts used in the EDA
notebook — loaded live from MySQL.

### Pages
- **Home** (`app.py`) — dataset & model overview, headline metrics, quick churn snapshot
- **Predict Churn** — score a single customer via a form, or a batch via CSV upload
- **EDA Dashboard** — interactive univariate / bivariate / correlation charts, with sidebar filters
- **Model Explainability** — global feature importance + per-customer SHAP breakdown (requires the `shap` package)
- **Customer Lookup** — search a real customer by ID and see their profile, score, and population percentile
- **Settings** — decision threshold control, data source status, model info, cache clearing

### Setup

1. From the `streamlit_app/` folder, install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   `scikit-learn` is pinned to `1.6.1` to match the version the model pipeline was
   trained/pickled with — install a different version and you may hit an
   `InconsistentVersionWarning` or an unpickling error. Match your `xgboost`
   version to whatever was used during training if you retrain the model.

2. Point the app at your MySQL database. Copy the secrets template and fill it in:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Then edit `.streamlit/secrets.toml` with your host/user/password/database.
   The app queries the **`ecommerce_cleaned`** table (schema from
   `sql/ecommerce_database.sql` — run that script against your MySQL server
   first, and load `ecommerce_cleaned.csv` into it, if you haven't already).

   If MySQL isn't reachable, the app automatically falls back to the CSV
   snapshot at `../sql/ecommerce_cleaned.csv` (project root, not a copy inside
   `streamlit_app/`) so it still runs — the sidebar shows which source is active.

3. Run the app:
   ```bash
   streamlit run app.py
   ```

**Important:** `streamlit_app/` must stay inside the project root, as a sibling
of `models/` and `sql/` — it reads `../models/final_churn_model_pipeline.pkl`
and `../sql/ecommerce_cleaned.csv` directly rather than keeping its own copies.
Moving `streamlit_app/` elsewhere will break those paths.

### App layout
```
streamlit_app/
├── app.py                      # Home page
├── pages/
│   ├── 1_🔮_Predict_Churn.py
│   ├── 2_📊_EDA_Dashboard.py
│   ├── 3_🧠_Model_Explainability.py
│   ├── 5_🔎_Customer_Lookup.py
│   └── 6_⚙️_Settings.py
├── utils/
│   ├── db.py                   # MySQL loader; CSV fallback reads ../sql/ecommerce_cleaned.csv
│   ├── model_utils.py          # loads ../models/final_churn_model_pipeline.pkl; prediction + SHAP
│   └── styling.py              # dark theme CSS, Plotly template, shared sidebar
├── .streamlit/
│   ├── config.toml             # dark theme
│   └── secrets.toml.example
└── requirements.txt
```
(No `models/` or `data/` folders here — the app reads those files straight from the
project root's `models/` and `sql/` folders shown in the top-level structure above.)

### Notes
- The sidebar shows live status dots for both the **model** (loaded / error) and the
  **database connection** (MySQL / CSV fallback) on every page.
- The batch prediction page validates uploaded CSVs against the exact column set the model
  pipeline expects, and offers a downloadable template.
- The decision threshold (default 0.5, same as `predict_churn.py`) is adjustable on the
  Settings page and applies consistently across Predict and Customer Lookup.
