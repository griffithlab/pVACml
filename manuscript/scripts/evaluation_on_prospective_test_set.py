"""Evaluate the RF model on the prospective test set (figures + tables)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Data: manuscript/data/training_testing_data/ (resolved from __file__, clone-agnostic)
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR.parent / "data" / "training_testing_data"


def _load_prospective_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    for col in ("Evaluation", "python_Evaluation_pred"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# read the prospective test set
prospective_test_set = _load_prospective_csv("prospective_test_set.csv")



## Confusion matrix
python_conf_matrix = confusion_matrix(prospective_test_set['Evaluation'], prospective_test_set['python_Evaluation_pred'])
disp_python = ConfusionMatrixDisplay(
    confusion_matrix=python_conf_matrix,
    display_labels=prospective_test_set['Evaluation'].unique()
)
fig, ax = plt.subplots()
disp_python.plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix (Prospective Test Set)', fontsize=14)
#plt.savefig(outdir / "rf_ds_external_eval_conf_matrix.png", dpi=800, bbox_inches='tight')
plt.show()
# Calculate and print accuracy
python_accuracy = np.trace(python_conf_matrix) / np.sum(python_conf_matrix)
print(f"Accuracy (Python Model): {python_accuracy:.3f}")




## ROC and PR curves for "Accept" and "Reject" binary classification
# Filter the data to only include "Accept" and "Reject" rows
filtered_df = prospective_test_set[prospective_test_set['Evaluation'].isin(['Accept', 'Reject'])]
filtered_df = filtered_df[filtered_df['python_Evaluation_pred'].isin(['Accept', 'Reject'])]

# Convert to binary classification (Accept = 1, Reject = 0)
y_true = (filtered_df['Evaluation'] == 'Accept').astype(int) # labels of 0s and 1s
y_pred_proba = filtered_df['python_Accept_pred_prob'].values # model prediction probabilities of the Accept class

# Create ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = roc_auc_score(y_true, y_pred_proba)

# Create PR curve
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
avg_precision = average_precision_score(y_true, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
#plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Accept vs Reject)')
plt.legend()
#plt.grid(True, alpha=0.3)

# Plot PR curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (AUPRC = {avg_precision:.3f})', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Accept vs Reject)')
plt.legend()

plt.tight_layout()
#plt.savefig(outdir / "roc_pr_curves_accept_reject.png", dpi=800, bbox_inches='tight')
plt.show()


# Calculate per-class sensitivity, specificity, and F1 for Python model
# Get class labels
classes = prospective_test_set['Evaluation'].cat.categories.tolist()
print(classes)

# Calculate metrics for each class
metrics_list = []

for i, class_name in enumerate(classes):
    # True Positives: diagonal element
    tp = python_conf_matrix[i, i]
    
    # False Negatives: sum of row i minus diagonal
    fn = np.sum(python_conf_matrix[i, :]) - tp
    
    # False Positives: sum of column i minus diagonal
    fp = np.sum(python_conf_matrix[:, i]) - tp
    
    # True Negatives: all other correct predictions
    tn = np.sum(python_conf_matrix) - tp - fn - fp
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    metrics_list.append({
        'Class': class_name,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1': f1,
        'Precision': precision,
        'TP': tp,
        'FN': fn,
        'FP': fp,
        'TN': tn
    })

# Create DataFrame
python_metrics_df = pd.DataFrame(metrics_list)

# Round to 3 decimal places for display
python_metrics_df[['Sensitivity', 'Specificity', 'F1', 'Precision']] = python_metrics_df[['Sensitivity', 'Specificity', 'F1', 'Precision']].round(3)

print("Per-class Metrics for Entire Prospective Test Set:")
print("=" * 80)
print(python_metrics_df.to_string(index=False))
print("\n")






### Performance assessment WITHOUT the 4 threshold calibration samples
## read the prospective test set without the 4 threshold calibration samples
prospective_test_set_no_cal = _load_prospective_csv(
    "prospective_test_set_no_calibration.csv"
)


## Confusion matrix
python_conf_matrix_no_cal = confusion_matrix(prospective_test_set_no_cal['Evaluation'], prospective_test_set_no_cal['python_Evaluation_pred'])
disp_python_no_cal = ConfusionMatrixDisplay(confusion_matrix=python_conf_matrix_no_cal, display_labels=prospective_test_set_no_cal['Evaluation'].unique())
fig, ax = plt.subplots()
disp_python_no_cal.plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix (Prospective Test Set) WITHOUT the 4 threshold calibration samples', fontsize=14)
#plt.savefig(outdir / "rf_ds_external_eval_conf_matrix_no_cal.png", dpi=800, bbox_inches='tight')
plt.show()
# Calculate and print accuracy
python_accuracy_no_cal = np.trace(python_conf_matrix_no_cal) / np.sum(python_conf_matrix_no_cal)
print(f"Accuracy WITHOUT the 4 threshold calibration samples: {python_accuracy_no_cal:.3f}")



## ROC and PR curves for "Accept" and "Reject" binary classification WITHOUT the 4 threshold calibration samples
# Filter the data to only include "Accept" and "Reject" rows
filtered_df = prospective_test_set_no_cal[prospective_test_set_no_cal['Evaluation'].isin(['Accept', 'Reject'])]
filtered_df = filtered_df[filtered_df['python_Evaluation_pred'].isin(['Accept', 'Reject'])]

print(f"Original data shape: {prospective_test_set_no_cal.shape}")
print(f"Filtered data shape (Accept/Reject only): {filtered_df.shape}")

# Convert to binary classification (Accept = 1, Reject = 0)
y_true = (filtered_df['Evaluation'] == 'Accept').astype(int) # labels of 0s and 1s
y_pred_proba = filtered_df['python_Accept_pred_prob'].values # model prediction probabilities of the Accept class

# Create ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = roc_auc_score(y_true, y_pred_proba)

# Create PR curve
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
avg_precision = average_precision_score(y_true, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Accept vs Reject) WITHOUT the 4 threshold calibration samples')
plt.legend()
#plt.grid(True, alpha=0.3)
#plt.savefig(outdir / "roc_curve_accept_reject_no_cal.png", dpi=800, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR curve (AUPRC = {avg_precision:.3f})', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Accept vs Reject) WITHOUT the 4 threshold calibration samples')
plt.legend()
#plt.grid(True, alpha=0.3)
#plt.savefig(outdir / "pr_curve_accept_reject_no_cal.png", dpi=800, bbox_inches='tight')
plt.show()


## Per-class metrics WITHOUT the 4 threshold calibration samples
# Calculate per-class sensitivity, specificity, and F1 for Python model
# Get class labels
classes = prospective_test_set_no_cal['Evaluation'].cat.categories.tolist()
print(classes)

# Calculate metrics for each class
metrics_list = []

for i, class_name in enumerate(classes):
    # True Positives: diagonal element
    tp = python_conf_matrix_no_cal[i, i]
    
    # False Negatives: sum of row i minus diagonal
    fn = np.sum(python_conf_matrix_no_cal[i, :]) - tp
    
    # False Positives: sum of column i minus diagonal
    fp = np.sum(python_conf_matrix_no_cal[:, i]) - tp
    
    # True Negatives: all other correct predictions
    tn = np.sum(python_conf_matrix_no_cal) - tp - fn - fp
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    metrics_list.append({
        'Class': class_name,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1': f1,
        'Precision': precision,
        'TP': tp,
        'FN': fn,
        'FP': fp,
        'TN': tn
    })

# Create DataFrame
python_metrics_df = pd.DataFrame(metrics_list)

# Round to 3 decimal places for display
python_metrics_df[['Sensitivity', 'Specificity', 'F1', 'Precision']] = python_metrics_df[['Sensitivity', 'Specificity', 'F1', 'Precision']].round(3)

print("Per-class Metrics WITHOUT the 4 threshold calibration samples:")
print("=" * 80)
print(python_metrics_df.to_string(index=False))
print("\n")