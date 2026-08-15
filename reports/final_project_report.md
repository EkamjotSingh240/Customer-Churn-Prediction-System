# E-Commerce Customer Churn — Final Project Report

## 1. Project Overview
- **Objective**: Identify drivers of customer churn and build a predictive model to flag at-risk customers for retention efforts.
- **Dataset**: 5,630 customers, 20 features, cleaned to 5,628 rows after outlier removal.
- **Churn rate**: 16.8% (940 churned / 4,688 retained).

## 2. Data Cleaning Summary
- Removed 2 outlier rows (`WarehouseToHome` = 126/127 — implausible delivery distance, likely data entry error).
- Imputed all missing values: group-wise median (by Churn) for `Tenure`, `WarehouseToHome`, `DaySinceLastOrder`; global median for `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`; mode for `HourSpendOnApp`.
- Standardized inconsistent categorical labels (`CC`/`Credit Card`, `COD`/`Cash on Delivery`, `Phone`/`Mobile Phone`).
- Verified zero nulls and zero true duplicate customers after cleaning.

## 3. Key EDA Findings

### Strongest predictors of churn
| Feature | Finding |
|---|---|
| **Tenure** | Strongest signal overall (correlation -0.37). Churned customers cluster near 0-1 months; retained customers spread much wider (median ~9-10 months). |
| **Complain** | Customers who complained churn at 31.7% vs. 10.9% for those who didn't — nearly 3x. |
| **NumberOfDeviceRegistered** | Clear upward trend: 9.4% churn (1-2 devices) rising to 34.6% (6 devices). |
| **PreferredOrderCat** | Mobile Phone category churns at 27.4%, far above Grocery (4.9%). |
| **MaritalStatus** | Single customers churn at 26.7%, more than double Married (11.5%). |
| **DaySinceLastOrder** | Moderate negative correlation (-0.17) with churn. |
| **CashbackAmount** | Mild negative correlation (-0.15) — lower cashback associates with higher churn. |

### Weak/no predictors
`HourSpendOnApp`, `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`, `Gender`, `PreferredLoginDevice` showed minimal separation between churned and retained customers.

### Most notable finding
Satisfaction score does **not** protect against churn when a complaint is involved — among customers who complained, churn rate climbed with satisfaction score (up to 38.8% at score 4), suggesting the complaint itself (or its resolution) matters more than the stated satisfaction level.

## 4. Modeling Summary

### Models tried
| Model | Test F1 | Test ROC-AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.618 | 0.902 | Baseline; high recall (0.847), low precision (0.486) |
| Random Forest (constrained) | 0.830 | 0.974 | Improved precision substantially over baseline |
| **XGBoost (early stopping)** | **0.946** | **0.997** | **Final model** — best balance of performance and controlled overfitting |

### Final model
**XGBoost Classifier, tuned via early stopping** (263 trees, max_depth=6, learning_rate=0.1, `scale_pos_weight` for class imbalance).

Overfitting was rigorously checked: train/validation gap was reduced through early stopping (F1 gap 0.056, smallest of all variants tested), and performance was confirmed stable on genuinely unseen test rows after ruling out train/test row-overlap effects.

### Final test-set performance
| Metric | Score |
|---|---|
| Accuracy | 0.983 |
| Precision | 0.953 |
| Recall | 0.940 |
| F1-score | 0.946 |
| ROC-AUC | 0.997 |

Out of all actual churners in the test set, the model correctly identified the large majority while keeping false alarms low — a strong, deployable-quality result.

## 5. Feature Importance (Model-Confirmed Drivers)
Top features by importance in the final XGBoost model:
1. **Tenure** — confirms EDA's strongest finding
2. **Complain** — confirms EDA's strongest categorical finding
3. **PreferredOrderCat (Laptop & Accessory)** — a model-only finding; likely a strong "low churn risk" signal given this category's large base and low churn rate
4. **SatisfactionScore**
5. **CityTier**
6. **NumberOfAddress**
7. **PreferredOrderCat (Mobile Phone)** — confirms EDA
8. **MaritalStatus (Single)** — confirms EDA
9. **DaySinceLastOrder** — confirms EDA
10. **NumberOfDeviceRegistered** — confirms EDA

The model's top two features align exactly with EDA's strongest findings, validating the exploratory analysis. Additional features surfaced by the model (SatisfactionScore, CityTier, NumberOfAddress) likely reflect interaction effects not fully visible in univariate/bivariate EDA charts.

## 6. Business Recommendations
1. **Prioritize new customers.** Tenure is the single strongest churn driver — customers are most at risk in their first 1-2 months. Consider a structured onboarding or early-engagement program targeting this window.
2. **Resolve complaints effectively, not just quickly.** Complaining nearly triples churn risk regardless of the customer's stated satisfaction score. Investing in complaint resolution quality (not just closure speed) may have outsized retention impact.
3. **Watch multi-device customers.** Churn risk rises steadily with the number of registered devices — this may indicate a specific customer segment (e.g., account sharing, shopping around) worth investigating further.
4. **Segment retention offers by order category and marital status.** Mobile Phone category shoppers and Single customers show meaningfully higher churn rates and may respond well to targeted retention campaigns.
5. **Deploy the churn model for proactive outreach.** With 94% recall and 95% precision on unseen data, the model can reliably flag at-risk customers for early intervention (e.g., personalized offers, check-in outreach) before they churn.