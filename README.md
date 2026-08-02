# Capstone Project: Telecom Customer Churn Prediction

Predicting which customers are likely to churn, and explaining *why*,
so a retention team can act before the customer leaves.

## 1. Problem Statement

Subscription telecom revenue depends on retaining customers. Acquiring
a new customer typically costs far more than retaining an existing one,
so the ability to identify **which customers are at risk of churning,
and why**, lets a retention team intervene early (targeted offers,
proactive outreach, contract incentives) instead of reacting after the
customer has already cancelled.

This project builds an end-to-end, production-style pipeline that:

1. Loads raw customer account data
2. Cleans known data-quality issues
3. Explores the data to understand churn drivers
4. Engineers business-meaningful features
5. Trains and compares multiple classification models
6. Evaluates the best model on held-out data with imbalance-aware metrics
7. Explains individual and global predictions with SHAP
8. Translates findings into concrete business recommendations

## 2. Repository Structure

```
capstone/
├── README.md                      <- you are here
├── requirements.txt                <- Python dependencies
├── data/
│   └── telco_customer_churn.csv    <- input dataset
├── notebook/
│   └── Capstone_Project.ipynb      <- the full, executed pipeline (primary deliverable)
├── scripts/
│   └── generate_data.py            <- reproducible synthetic-data generator (see note below)
├── report_assets/                  <- chart images used in the written report
├── Capstone_Final_Report.docx      <- written report: problem, approach, findings, recommendations
├── Model_Metrics.txt               <- final model spec, hyperparameters, and performance numbers
└── outputs/                        <- (optional) any additional exported artifacts
```

## 3. Data

**Note on the dataset:** `data/telco_customer_churn.csv` is a **synthetically
generated** dataset (`scripts/generate_data.py`) that mirrors the schema,
scale, churn rate (~22-23%), and feature relationships of the widely used
IBM/Kaggle "Telco Customer Churn" dataset. It was generated this way so
the project is fully self-contained and reproducible without an external
download, while still containing realistic, non-trivial signal and
realistic data-quality issues (missing values, stray whitespace,
duplicate rows) for the cleaning stage to address. The entire pipeline
(cleaning, EDA, feature engineering, modelling, evaluation, explainability)
is written generically and will run unchanged against the real IBM Telco
Customer Churn CSV if you substitute it in `data/` with matching column
names.

**Columns include:** demographics (`gender`, `SeniorCitizen`, `Partner`,
`Dependents`), account info (`tenure`, `Contract`, `PaperlessBilling`,
`PaymentMethod`), subscribed services (`PhoneService`, `MultipleLines`,
`InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
`TechSupport`, `StreamingTV`, `StreamingMovies`), billing
(`MonthlyCharges`, `TotalCharges`), and the target `Churn` (`Yes`/`No`).

## 4. How to Run

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the synthetic dataset
python scripts/generate_data.py

# 4. Launch the notebook
jupyter notebook notebook/Capstone_Project.ipynb
# Run all cells top-to-bottom (Cell -> Run All)
```

The notebook is self-contained: it loads `data/telco_customer_churn.csv`,
runs the full pipeline, and displays every plot/table inline. It has
already been executed end-to-end and saved with outputs, so you can also
just read it without re-running.

## 5. Methodology Summary

| Stage | What was done |
|---|---|
| **Data Load** | Load raw CSV, initial shape/dtype/target-balance check |
| **Data Cleaning** | De-duplication, categorical text normalization, `TotalCharges` type coercion + principled missing-value imputation (0 for brand-new accounts), numeric sanity checks |
| **EDA** | Target distribution, numeric distributions by churn, churn rate by categorical drivers (contract, internet service, payment method, add-ons), correlation analysis |
| **Feature Engineering** | `NumAddOnServices`, `AvgMonthlySpend`, `TenureBucket`, `HasStreaming`, `IsElectronicCheck`, `IsMonthToMonth`; `ColumnTransformer` pipeline (scaling + one-hot encoding) to prevent leakage |
| **Modelling** | Logistic Regression, Random Forest, XGBoost compared via 5-fold stratified CV (ROC-AUC); best model tuned with `RandomizedSearchCV` |
| **Evaluation** | Held-out test-set accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC and precision-recall curves |
| **Explainability** | SHAP global summary (beeswarm + mean-\|impact\| bar) and a local force-plot explanation for an individual high-risk customer |

## 6. Key Results (see `Model_Metrics.txt` for full detail)

- Final model: **Logistic Regression** (selected by cross-validated ROC-AUC among the three candidates), tuned via `RandomizedSearchCV`.
- Test-set ROC-AUC: **0.729**
- Test-set Recall on churn class: **0.66** (catches ~2 out of 3 customers who actually churn)
- Strongest churn drivers (consistent across EDA, model coefficients, and SHAP): **contract type (month-to-month)**, **tenure**, **internet service type / payment method**, and **number of add-on services**.

## 7. Deliverables Checklist

- [x] Jupyter notebook with full pipeline, executed with outputs (`notebook/Capstone_Project.ipynb`)
- [x] Reproducible data-generation script (`scripts/generate_data.py`)
- [x] `requirements.txt`
- [x] Written report — problem, approach, findings, further research, recommendations (`Capstone_Final_Report.docx`)
- [x] Model metrics file — features, hyperparameters, performance (`Model_Metrics.txt`)
- [x] This README

## 8. Further Research & Recommendations

See Sections 6 and 7 of `Capstone_Final_Report.docx` for the full
discussion, including three concrete, prioritized recommendations for
the client's retention team.
