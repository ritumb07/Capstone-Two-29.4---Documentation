"""
Generates a synthetic telecom customer churn dataset with realistic,
non-trivial relationships between features and the churn target.
This mimics the structure of the well-known IBM Telco Customer Churn
dataset but is fully synthetic (no external download required).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 7000

# ---- Demographics ----
gender = rng.choice(["Male", "Female"], size=N)
senior_citizen = rng.choice([0, 1], size=N, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], size=N, p=[0.48, 0.52])
dependents = np.where(
    partner == "Yes",
    rng.choice(["Yes", "No"], size=N, p=[0.55, 0.45]),
    rng.choice(["Yes", "No"], size=N, p=[0.12, 0.88]),
)

# ---- Account / tenure ----
tenure = rng.integers(0, 73, size=N)  # months
contract = rng.choice(
    ["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.24, 0.21]
)
paperless_billing = rng.choice(["Yes", "No"], size=N, p=[0.59, 0.41])
payment_method = rng.choice(
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    size=N,
    p=[0.34, 0.23, 0.22, 0.21],
)

# ---- Services ----
phone_service = rng.choice(["Yes", "No"], size=N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "Yes",
    rng.choice(["Yes", "No"], size=N, p=[0.42, 0.58]),
    "No phone service",
)
internet_service = rng.choice(
    ["DSL", "Fiber optic", "No"], size=N, p=[0.34, 0.44, 0.22]
)


def dependent_service(has_internet, p_yes=0.5):
    out = []
    for h in has_internet:
        if h == "No":
            out.append("No internet service")
        else:
            out.append(rng.choice(["Yes", "No"], p=[p_yes, 1 - p_yes]))
    return np.array(out)


online_security = dependent_service(internet_service, 0.38)
online_backup = dependent_service(internet_service, 0.44)
device_protection = dependent_service(internet_service, 0.44)
tech_support = dependent_service(internet_service, 0.38)
streaming_tv = dependent_service(internet_service, 0.49)
streaming_movies = dependent_service(internet_service, 0.49)

# ---- Charges ----
base = 18.0
service_add = (
    (phone_service == "Yes") * 5
    + (multiple_lines == "Yes") * 5
    + (internet_service == "DSL") * 22
    + (internet_service == "Fiber optic") * 42
    + (online_security == "Yes") * 5
    + (online_backup == "Yes") * 5
    + (device_protection == "Yes") * 5
    + (tech_support == "Yes") * 5
    + (streaming_tv == "Yes") * 8
    + (streaming_movies == "Yes") * 8
)
noise = rng.normal(0, 3, size=N)
monthly_charges = np.clip(base + service_add + noise, 18, 130).round(2)
total_charges = np.clip(
    monthly_charges * tenure + rng.normal(0, 20, size=N), 0, None
).round(2)
total_charges = np.where(tenure == 0, monthly_charges.round(2), total_charges)

# ---- Latent churn probability (realistic, multi-factor) ----
logit = (
    -1.6
    + 1.15 * (contract == "Month-to-month")
    + 0.35 * (contract == "One year")
    + 0.0 * (contract == "Two year")
    + 0.55 * (internet_service == "Fiber optic")
    - 0.55 * (internet_service == "No")
    + 0.45 * (payment_method == "Electronic check")
    + 0.30 * (paperless_billing == "Yes")
    - 0.40 * (online_security == "Yes")
    - 0.35 * (tech_support == "Yes")
    - 0.02 * tenure
    + 0.012 * (monthly_charges - 65)
    + 0.30 * senior_citizen
    - 0.25 * (partner == "Yes")
    - 0.15 * (dependents == "Yes")
    + rng.normal(0, 0.55, size=N)  # unexplained noise
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = (rng.uniform(size=N) < prob_churn).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

customer_id = [f"CUST-{100000 + i}" for i in range(N)]

df = pd.DataFrame(
    {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_label,
    }
)

# ---- Inject realistic messiness for the cleaning step ----
df["TotalCharges"] = df["TotalCharges"].astype(object)

# 1. Missing TotalCharges for a handful of brand-new customers (as in the real dataset)
missing_idx = rng.choice(df.index, size=25, replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan

# 2. A few stray whitespace strings (common real-world artifact)
ws_idx = rng.choice(df.index.difference(missing_idx), size=10, replace=False)
df.loc[ws_idx, "TotalCharges"] = " "

# 3. Inconsistent capitalization / stray spaces in a categorical column
messy_idx = rng.choice(df.index, size=40, replace=False)
df.loc[messy_idx, "gender"] = df.loc[messy_idx, "gender"].str.lower() + " "

# 4. Duplicate rows
dup_rows = df.sample(15, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# 5. Shuffle
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("/home/claude/capstone/data/telco_customer_churn.csv", index=False)
print("Saved:", df.shape)
print(df["Churn"].value_counts(normalize=True))
