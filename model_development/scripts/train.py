# %% Import dependencies
import numpy as np
import pandas as pd
from pathlib import Path

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib


# %% Paths — this file lives in model_development/scripts/; data and model/ are siblings of scripts/
_MODEL_DEV_ROOT = Path(__file__).resolve().parent.parent
data_dir = _MODEL_DEV_ROOT / "data"
# Grid-search checkpoints and best-hparam pickles
temporary_artifacts_dir = _MODEL_DEV_ROOT / "model" / "temporary_model_artifacts"
# Final RF bundle for prediction
model_dir = _MODEL_DEV_ROOT / "model" / "pvactools7.0_model"
temporary_artifacts_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
numpy_version = "numpy126"  # compatible with packages in pVACtools 7.0
# %% Read imputed data
post_imputed_data = pd.read_csv(data_dir / "merged_df_post_imputation.csv")

# %% Label encode categorical columns
# Filter out "Review" and replace "Pending" with "Reject"
post_imputed_data = post_imputed_data[post_imputed_data['Evaluation'] != 'Review']  # Remove rows with "Review"
post_imputed_data['Evaluation'] = post_imputed_data['Evaluation'].replace('Pending', 'Reject')  # Replace "Pending" with "Reject"

# Encode "Accept" as 1 and "Reject" as 0
post_imputed_data['Evaluation'] = post_imputed_data['Evaluation'].map({'Accept': 1, 'Reject': 0})

# %% Split data into training and testing sets
# Split data into training and testing sets 
train_data, test_data = train_test_split(post_imputed_data, test_size=0.25, random_state=42)

# Separate features and target
drop_cols = [col for col in ['Evaluation', 'ID', 'patient_id'] if col in train_data.columns]
X_train = train_data.drop(columns=drop_cols)
y_train = train_data['Evaluation']
X_test = test_data.drop(columns=drop_cols)
y_test = test_data['Evaluation']

print(f"   Features: {X_train.shape[1]} columns")
print(f"   Training features shape: {X_train.shape}")
print(f"   Testing features shape: {X_test.shape}")
# Save train/test splits with label encoders
#train_data.to_csv(data_dir / "rf_train.csv", index=False)
#test_data.to_csv(data_dir / "rf_test.csv", index=False)

# %% Hyperparameter tuning using GridSearchCV
print("\n" + "=" * 80)
print("Hyperparameter tuning with GridSearchCV...")
print("   This may take several hours...")
    
# Setup RF model with out-of-bag support (same parameters as original)
rf = BalancedRandomForestClassifier(
    oob_score=True,
    n_estimators=100,  # Number of trees
    random_state=918,   # Set random seed for reproducibility
    n_jobs=-1,         # use all available cores
    replacement=True,
    bootstrap=True     # Set bootstrap=True for OOB estimation
)

# Define search space for max_features (mtry) and n_estimators (ntree)
param_grid = {
    'max_features': list(range(1, 72)),  # mtry from 1 to 72 since there are 72 features in total
    'n_estimators': list(range(1, 5001, 50))  # ntree from 1 to 5000 with 50 interval
}

# Perform grid search with accuracy scoring
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Fit the model with encoded data
print("   Starting grid search...")
grid_search.fit(X_train, y_train)

# Extract and print best parameters
best_params = grid_search.best_params_
best_mtry = best_params["max_features"]
best_ntree = best_params["n_estimators"]
print(f"   Best max_features (mtry): {best_mtry}")
print(f"   Best n_estimators (ntree): {best_ntree}")

# Save the best parameters if want to use them later without re-running the grid search
joblib.dump(best_mtry, temporary_artifacts_dir / "best_mtry_rf_numpy126.pkl")
joblib.dump(best_ntree, temporary_artifacts_dir / "best_ntree_rf_numpy126.pkl")

# %% Train model with best parameters
print("\n" + "=" * 80)
print("TRAINING BALANCED RANDOM FOREST WITH BEST PARAMETERS")
print("=" * 80)
rf_downsample = BalancedRandomForestClassifier(
    n_estimators=best_ntree,
    max_features=best_mtry,
    oob_score=True,
    random_state=918,
    n_jobs=-1,
    replacement=True,
    bootstrap=True
)
rf_downsample.fit(X_train, y_train)

# Save the trained model
_model_filename = f"rf_downsample_model_{numpy_version}.pkl"
_model_path = model_dir / _model_filename
joblib.dump(rf_downsample, _model_path)
print(f"   Model saved to: {_model_path}")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"Environment: neoantigen_ml_{numpy_version}")
print(f"NumPy version: {np.__version__}")
print(f"Best parameters: mtry={best_mtry}, ntree={best_ntree}, saved in {temporary_artifacts_dir}")
print(f"Model saved to: {_model_path}")