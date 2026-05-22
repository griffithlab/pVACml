"""Imputation quality check (MCAR simulation) using manuscript snapshot pickles."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  # registers IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Data: manuscript/data/imputation_analysis/ (resolved from __file__, clone-agnostic)
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR.parent / "data" / "imputation_analysis"

with open(DATA_DIR / "data_to_impute.pkl", "rb") as f:
    data_to_impute = pickle.load(f)

with open(DATA_DIR / "features_to_test.pkl", "rb") as f:
    features_to_test = pickle.load(f)

# Normalize to an ordered list of column names present in the frame
_feature_names = pd.Series(features_to_test).astype(str).tolist()
features_to_test = [c for c in _feature_names if c in data_to_impute.columns]

# Step 1: Identify Complete Cases

# Keep only rows with no missing values as your "ground truth"
df_complete = data_to_impute.dropna()
print(f"Complete cases: {len(df_complete)}")

df_na = data_to_impute[data_to_impute.isnull().any(axis=1)]


# Step 2: Define Which Features to Evaluate
# Select the features that originally had missingness
print(f"Features to test: {features_to_test}")


# Step 3: Artificially Introduce Missingnes
def introduce_mcar(df, missing_rate=0.15, random_state=918):
    """
    Randomly mask values in complete-case data to simulate MCAR missingness.
    Returns the masked dataframe and a boolean mask of where values were removed.
    """
    np.random.seed(random_state)
    df_masked = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns) # creates a same-shape boolean table to record which values were removed.

    for col in features_to_test:
        n_to_mask = int(len(df) * missing_rate) # computes how many rows to bland out
        mask_idx = np.random.choice(df.index, size=n_to_mask, replace=False) # picks that many unique row indices at random
        df_masked.loc[mask_idx, col] = np.nan # replaces selected rows with NaN
        mask.loc[mask_idx, col] = True # marks those exact cells as artificially removed

    return df_masked, mask

EVAL_SEED = 918
df_masked, mask = introduce_mcar(df_complete, missing_rate=0.15, random_state=EVAL_SEED)


# Step 4: Impute the Masked Data
# IterativeImputer
imputer_eval = IterativeImputer(max_iter=10, random_state=EVAL_SEED)
df_imputed_array = imputer_eval.fit_transform(df_masked)
df_imputed = pd.DataFrame(df_imputed_array, columns=df_masked.columns, index=df_masked.index)

# Median imputer on the SAME df_masked
simple_imputer = SimpleImputer(strategy="median")
df_simple_array = simple_imputer.fit_transform(df_masked)
df_simple = pd.DataFrame(df_simple_array, columns=df_masked.columns, index=df_masked.index)

print(f"Masked dataset: {df_masked.shape[0]} rows, {df_masked.isnull().sum().sum()} total masked values")



# Step 5: Evaluate Imputation Quality
rows = []

for col in features_to_test:
    col_mask = mask[col]
    true_vals = df_complete.loc[col_mask, col]
    imputed_vals_iter = df_imputed.loc[col_mask, col]
    imputed_vals_median = df_simple.loc[col_mask, col]

    feature_std = df_complete[col].std()

    # IterativeImputer metrics
    rmse_iter = np.sqrt(mean_squared_error(true_vals, imputed_vals_iter))
    nrmse_iter = rmse_iter / feature_std if feature_std > 0 else np.nan
    r2_iter = r2_score(true_vals, imputed_vals_iter)

    # Median imputer metrics
    rmse_median = np.sqrt(mean_squared_error(true_vals, imputed_vals_median))
    nrmse_median = rmse_median / feature_std if feature_std > 0 else np.nan
    r2_median = r2_score(true_vals, imputed_vals_median)

    rows.append({
        'Feature': col,
        'NRMSE_iter': round(nrmse_iter, 3),
        'R2_iter': round(r2_iter, 3),
        'NRMSE_median': round(nrmse_median, 3),
        'R2_median': round(r2_median, 3)
    })

nrmse_table_replace = pd.DataFrame(rows).sort_values('NRMSE_iter').reset_index(drop=True)
print(nrmse_table_replace.to_string())

# Step 6 make final data table with relevant columns:

# Missingness summary for the same features (rebuilt from shipped data; no extra pickle)
missing_summary_replace = (
    data_to_impute[features_to_test]
    .isna()
    .sum()
    .rename("n_missing")
    .to_frame()
    .reset_index()
    .rename(columns={"index": "Feature"})
)
missing_summary_replace["n_rows"] = len(data_to_impute)
missing_summary_replace["percent_missing"] = (
    missing_summary_replace["n_missing"] / missing_summary_replace["n_rows"] * 100
)
missing_summary_replace = missing_summary_replace.sort_values(
    "percent_missing", ascending=False
)
missing_summary_replace["percent_missing"] = missing_summary_replace["percent_missing"].round(3)
missing_summary_replace = missing_summary_replace[
    missing_summary_replace["percent_missing"] != 0
].reset_index(drop=True)

# Merge with missing_summary to get n_missing and n_rows, then select final columns, and sort by n_missing ascending
nrmse_final_replace = (
    nrmse_table_replace
    .rename(columns={'NRMSE_iter': 'NRMSE', 'R2_iter': 'R2'})
    .merge(missing_summary_replace[['Feature', 'n_missing', 'n_rows']], 
           on='Feature', how='left')
)
# Calculate % missing, rename columns, select final columns, and sort
nrmse_final_replace['% missing'] = nrmse_final_replace['n_missing'] / nrmse_final_replace['n_rows']
nrmse_final_replace = (
    nrmse_final_replace
    .rename(columns={'n_missing': 'N missing', 'n_rows': 'N rows'})
    [['Feature', 'N missing', 'N rows', '% missing', 'NRMSE', 'R2']]
    .sort_values('N missing', ascending=False)
    .reset_index(drop=True)
)
# Round all columns to 3 decimal places
nrmse_final_replace = nrmse_final_replace.round(3)

print(nrmse_final_replace.to_string())