# %% Import dependencies
import pickle
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder


# %% Paths (for reproducibility when used as a standalone script)
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data"
# Imputer + label encoders for downstream train/predict
model_dir = BASE_DIR / "model" / "pvactools7.0_model"
model_dir.mkdir(parents=True, exist_ok=True)
numpy_version = "numpy126"  # compatible with pVACtools / NumPy 1.26 stack

# Line-buffer stdout so sklearn stderr (e.g. ConvergenceWarning) does not appear above our banner
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (OSError, ValueError):
        pass

print()
print("Impute missing values (fit encoders + IterativeImputer)", flush=True)
print("-" * 60, flush=True)
print(f"Working directory: {BASE_DIR}", flush=True)

# %% Load pre-imputation table (still contains missing numeric values)
input_csv = data_dir / "merged_df_pre_imputation.csv"
print(f"\nLoading input CSV:\n  {input_csv}", flush=True)
pre_imputed_data = pd.read_csv(input_csv)
na_before = int(pre_imputed_data.isnull().sum().sum())
print(
    f"  Rows: {len(pre_imputed_data):,}  |  Columns: {pre_imputed_data.shape[1]}  |  "
    f"Missing cells: {na_before:,}",
    flush=True,
)

# %% Exclude columns from imputation and label encode categorical columns
# Exclude columns from imputation (ID columns and Evaluation column to prevent data leakage)
exclude_columns = ["ID", "patient_id", "Evaluation"]
columns_to_impute = pre_imputed_data.columns.difference(exclude_columns)

excluded_data = pre_imputed_data[exclude_columns].copy()
# .copy() avoids pandas SettingWithCopyWarning when assigning encoded columns below
data_to_impute = pre_imputed_data[columns_to_impute].copy()

print(
    f"\nColumns held out of the imputer (re-attached after): {exclude_columns}",
    flush=True,
)
print(f"Columns sent to the imputer: {len(columns_to_impute)}", flush=True)

# %% Label encode categorical columns
categorical_columns = data_to_impute.select_dtypes(include=["category", "object"]).columns
cat_list = list(categorical_columns)
if cat_list:
    print(
        f"\nLabel-encoding {len(cat_list)} categorical column(s): {', '.join(cat_list)}",
        flush=True,
    )
else:
    print("\nNo object/category columns to label-encode.", flush=True)

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data_to_impute[col] = le.fit_transform(data_to_impute[col])
    label_encoders[col] = le

encoders_path = model_dir / f"label_encoders_{numpy_version}.pkl"
with open(encoders_path, "wb") as f:
    pickle.dump(label_encoders, f)
print(f"Saved label encoders:\n  {encoders_path}", flush=True)

# %% Impute missing values
print(
    "\nFitting IterativeImputer. "
    "A sklearn ConvergenceWarning here is common and does not mean the run failed.",
    flush=True,
)
imputer = IterativeImputer(random_state=918)
imputed_data = imputer.fit_transform(data_to_impute)

imputer_path = model_dir / f"trained_imputer_{numpy_version}.joblib"
joblib.dump(imputer, imputer_path)
print(f"Saved trained imputer:\n  {imputer_path}", flush=True)

imputed_data = pd.DataFrame(imputed_data, columns=columns_to_impute)

post_imputed_data = pd.concat(
    [excluded_data.reset_index(drop=True), imputed_data.reset_index(drop=True)],
    axis=1,
)

na_after = int(post_imputed_data.isnull().sum().sum())
print(f"\nMissing cells after imputation: {na_after}", flush=True)

output_csv = data_dir / "merged_df_post_imputation.csv"
post_imputed_data.to_csv(output_csv, index=False)
print(f"Saved imputed table:\n  {output_csv}", flush=True)
print("-" * 60, flush=True)
print("Finished successfully.\n", flush=True)

# %%
