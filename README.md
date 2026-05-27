# pVACml

**Machine learning models for neoantigen candidate classification in personalized cancer vaccine design**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![pVACtools](https://img.shields.io/badge/integrated%20into-pVACtools%20v7-teal)](https://pvactools.readthedocs.io/en/7.0.0_docs/)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)]()

---

## Overview

pVACml houses the machine learning models and analysis code developed to support automated neoantigen candidate pre-classification within the [pVACtools](https://pvactools.readthedocs.io/en/7.0.0_docs/) suite. The model is trained on expert Immunogenomics Tumor Board (ITB) decisions from real-world personalized cancer vaccine clinical trials and classifies neoantigen peptide candidates as **Accept**, **Review**, or **Reject** based on a combination of genomic, expression, and MHC binding features.

The repository is organized into **two independent tracks**. They use **different dependency files** and **different model bundles**:

| Track | Purpose | Code location | Dependencies |
|--------|---------|----------------|--------------|
| **Manuscript** | Reproduce figures, analyses, manuscript model training workflows, and a **demonstration** prediction on a new case | [`manuscript/`](manuscript/) | [`manuscript/requirements.txt`](manuscript/requirements.txt) |
| **pVACtools 7.0 (compatible)** | Retrain / refresh the pipeline model and artifacts intended for **pVACtools v7** integration | Repository **root** scripts + [`model/pvactools7.0_model/`](model/pvactools7.0_model/) | [`requirements.txt`](requirements.txt) (repository root) |

**Important distinctions**

- The **manuscript** track reflects the paper’s analyses and bundled manuscript model. It is **not** the same artifact bundle that ships inside pVACtools 7.0.
- The **pVACtools 7.0** track is the one meant for **future retraining** and for the files that are **copied into the pVACtools codebase** for the v7 ML pipeline. The staging folder is **`model/pvactools7.0_model/`**, which corresponds to [`pvactools/supporting_files/ml_model_artifacts/`](https://github.com/griffithlab/pVACtools/tree/master/pvactools/supporting_files/ml_model_artifacts) in [griffithlab/pVACtools](https://github.com/griffithlab/pVACtools) (see [`model/pvactools7.0_model/README.md`](model/pvactools7.0_model/README.md)).

> End users running pVACseq with ML enabled should follow [pVACtools documentation](https://pvactools.readthedocs.io) (e.g. `pvacseq add_ml_predictions`). This README focuses on **developers** reproducing the paper or refreshing the v7 model from this repo.

---

## Repository structure

```
ITB_Automation_ML_Predictor/
├── manuscript/                      # Publication reproducibility (NOT the v7-shipped bundle)
│   ├── requirements.txt             # Python deps for all manuscript/scripts/*.py
│   ├── manuscript_model/            # Artifacts used by manuscript/scripts/predict.py demo
│   ├── data/
│   │   ├── predict_new_case_data/   # Demo inputs for manuscript prediction script
│   │   ├── training_testing_data/
│   │   ├── imputation_analysis/
│   │   ├── review_time_analysis_data/
│   │   └── …
│   └── scripts/
│       ├── ml_randomforest_model.py
│       ├── ml_logistic_model.py
│       ├── predict.py               # Manuscript: demo prediction on a new case
│       ├── evaluation_on_prospective_test_set.py
│       ├── imputation_analysis.py
│       └── review_time_analysis.py
│
├── model/
│   └── pvactools7.0_model/          # Staging = pVACtools pvactools/supporting_files/ml_model_artifacts
│       ├── README.md                # Maps this folder → upstream GitHub path
│       ├── rf_downsample_model_*.pkl
│       ├── trained_imputer_*.joblib
│       └── label_encoders_*.pkl
│
├── data/                            # Root-level training / imputation tables (pVACtools track)
├── impute_missing.py                # pVACtools track — step 1: fit imputer + encoders
├── train.py                         # pVACtools track — step 2: grid search + train RF
├── predict.py                       # pVACtools track — step 3: score a new case
├── requirements.txt                 # Python deps for root impute / train / predict scripts
└── README.md
```

---

## Track 1: Manuscript (`manuscript/`)

Use this when reproducing **figures, statistical analyses, manuscript RF/logistic workflows**, and the **manuscript** walkthrough of prediction on a new case.

### Environment

```bash
pip install -r manuscript/requirements.txt
```

Use **only** [`manuscript/requirements.txt`](manuscript/requirements.txt) for scripts under [`manuscript/scripts/`](manuscript/scripts/). That environment is aligned with the paper’s tooling (e.g. matplotlib, seaborn, broader analysis stack) and is **separate** from the minimal root `requirements.txt`.

### Demo prediction (manuscript)

The script [`manuscript/scripts/predict.py`](manuscript/scripts/predict.py) merges three pVACseq-style TSVs for one sample, applies the **manuscript** imputer/encoders/model, and writes an aggregated TSV.

Place inputs under **`manuscript/data/predict_new_case_data/`**:

- `<sample>.MHC_I.all_epitopes.aggregated.tsv`
- `<sample>.MHC_I.all_epitopes.tsv`
- `<sample>.MHC_II.all_epitopes.aggregated.tsv`

From the repository root:

```bash
python manuscript/scripts/predict.py
```

Paths, artifact directory, artifact version string, output directory, and thresholds are controlled by the constants at the bottom of `manuscript/scripts/predict.py` (see that file for defaults).

### Other manuscript analyses

```bash
cd manuscript/scripts
python <script_name>.py
```

Examples include prospective test evaluation, review-time analysis, and imputation comparisons. Each script may assume paths under `manuscript/data/`.

---

## Track 2: pVACtools 7.0–compatible pipeline (repository root)

Use this when **retraining** or regenerating artifacts for the **pVACtools v7**–compatible pipeline. Scripts are intended to be run **in this order**:

1. [`impute_missing.py`](impute_missing.py) — load pre-imputation table, fit label encoders + `IterativeImputer`, write imputed table and encoder/imputer artifacts.
2. [`train.py`](train.py) — train `BalancedRandomForestClassifier` (e.g. grid search), write best hyperparameters and the RF pickle.
3. [`predict.py`](predict.py) — merge class I/II inputs for a sample, apply saved imputer/encoders/model, write ML prediction TSV.

### Environment

```bash
pip install -r requirements.txt
```

Use the repository root [`requirements.txt`](requirements.txt) (NumPy / scikit-learn / imbalanced-learn pins aligned with the `neoantigen_ml_numpy126`–style stack used for these scripts). Do **not** mix this file with `manuscript/requirements.txt` unless you understand the version differences.

### Artifacts shipped to pVACtools

The directory **`model/pvactools7.0_model/`** holds the model bundle **intended to be copied into the pVACtools repository** for the v7 ML path. In **pVACtools**, the same files live under:

**[`pvactools/supporting_files/ml_model_artifacts/`](https://github.com/griffithlab/pVACtools/tree/master/pvactools/supporting_files/ml_model_artifacts)**  
([`griffithlab/pVACtools`](https://github.com/griffithlab/pVACtools) on GitHub)

| Here (pVACml) | In pVACtools |
|---------------|----------------|
| `model/pvactools7.0_model/` | `pvactools/supporting_files/ml_model_artifacts/` |

Git cannot keep two folders in different repositories synchronized by itself; the link above is the **canonical upstream location**. Use copy/PR (or an internal sync script) to publish updates. See [`model/pvactools7.0_model/README.md`](model/pvactools7.0_model/README.md).

After retraining, ensure filenames and layout match what the pVACtools integration expects before copying.

---

## Dataset

The training dataset comprises **1,943 expert-labeled neoantigen peptide candidates** spanning **33 patients** and **8 cancer types** from three clinical trials at Washington University School of Medicine:

| Trial | Cancer type | Patients |
|---|---|---|
| NCT05111353 | Pancreatic cancer | 14 |
| NCT03606967 | Metastatic TNBC | 11 |
| NCT05741242 | Basket trial (multiple types) | 8 |

Each record includes ITB labels (Accept / Reject / Review) and up to 72 features covering MHC binding predictions, RNA expression, tumor variant allele frequency, transcript support, driver gene status, etc.

> **Note on data availability:** Clinical genomic data from individual patients cannot be shared publicly due to IRB restrictions. Aggregate feature matrices used for model training are provided in [`manuscript/data/`](manuscript/data/) in de-identified form.

---

## Integration with pVACtools

The production model is integrated into **pVACtools v7**. End-user commands (for example):

```bash
pvacseq run ... --run-ml-predictions
```

```bash
pvacseq add_ml_predictions \
  input.tsv \
  output_dir/ \
  --accept-threshold 0.55 \
  --reject-threshold 0.30
```

Predictions are displayed in **pVACview** alongside binding affinity, expression, and variant-level features. Predicted labels are pre-populated but fully editable during ITB review.

**Developers:** when updating the bundled model in pVACtools from this repository, copy from **`model/pvactools7.0_model/`** into **[`pvactools/supporting_files/ml_model_artifacts/`](https://github.com/griffithlab/pVACtools/tree/master/pvactools/supporting_files/ml_model_artifacts)** after completing the root **impute → train** (and validate **predict**) workflow with [`requirements.txt`](requirements.txt).

---

## Citation

If you use pVACml or the associated model in your work, please cite:

```
Yao J, Singhal K, Kiwala S, et al.
Automating immunogenomic tumor board decision-making for neoantigen cancer vaccine design.
[Journal] (2025). DOI: [pending]
```

---

## Related resources

- [pVACtools documentation](https://pvactools.readthedocs.io/en/7.0.0_docs/)
- [pVACtools ML model artifacts on GitHub](https://github.com/griffithlab/pVACtools/tree/master/pvactools/supporting_files/ml_model_artifacts) (`ml_model_artifacts` — matches `model/pvactools7.0_model/` here)
- [pVACview interface](https://pvactools.readthedocs.io/en/latest/pvacview.html)
- [ImmunoNX pipeline](https://github.com/griffithlab/ImmunoNX_protocol)

---

## Questions and contributions

For questions about the model or codebase, please [open an issue](https://github.com/jyao36/pVACml/issues). For questions related to the pVACtools integration, see the [pVACtools GitHub](https://github.com/griffithlab/pVACtools).
