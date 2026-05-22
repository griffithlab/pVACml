# pVACml

**Machine learning models for neoantigen candidate classification in personalized cancer vaccine design**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![pVACtools](https://img.shields.io/badge/integrated%20into-pVACtools%20v7-teal)](https://pvactools.readthedocs.io/en/7.0.0_docs/)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)]()

---

## Overview

pVACml houses the machine learning models and analysis code developed to support automated neoantigen candidate pre-classification within the [pVACtools](https://pvactools.readthedocs.io/en/7.0.0_docs/) suite. The model is trained on expert Immunogenomics Tumor Board (ITB) decisions from real-world personalized cancer vaccine clinical trials and classifies neoantigen peptide candidates as **Accept**, **Review**, or **Reject** based on a combination of genomic, expression, and MHC binding features.

This repository serves two purposes:

1. **Manuscript reproducibility** — code, data, and notebooks used to produce all results, figures, and supplementary analyses in the associated publication
2. **Model versioning** — versioned model files for both the manuscript model and the model currently deployed in pVACtools v7

> The model is integrated into pVACtools as `pvacseq add_ml_predictions`. See the [pVACtools documentation](https://pvactools.readthedocs.io) for usage instructions.

---

## Repository Structure

```
pVACml/
├── manuscript/                  # Code and data for the manuscript
│   ├── data/                    
│   │   ├── imputation_analysis         
│   │   ├── predict_new_case_data     
│   │   ├── review_time_analysis_data
│   │   └── training_testing_data
│   └── scripts/               # Scripts for reproducing all analysis in the manuscript
│       ├── ml_logistic_model.py
│       ├── ml_randomforest_model.py
│       ├── predict.py
│       ├── evaluation_on_prospective_test_set.py
│       ├── review_time_analysis.py
│       └── imputation_analysis.py
│
├── models/                      # Versioned model files
│   ├── v1_manuscript/           # Model described in the manuscript
│   │   ├── rf_downsample_model_numpy126.pkl            # Trained Random Forest (down-sampling)
│   │   ├── trained_imputer_numpy126.joblib             # IterativeImputer instance
│   │   └── label_encoders_numpy126.pkl                 # Label encoder for preprosessing data
│   └── v2_pvactools7.0/          # Model deployed in pVACtools v7
│       ├── rf_downsample_model.pkl
│       ├── trained_imputer.joblib
│       └── label_encoders.pkl
│
├── requirements.txt
└── README.md
```

---

## Model Versions

| Version | Description | Imputation strategy | Integrated in |
|---|---|---|---|
| `v1_manuscript` | Model described in the associated publication | Median pre-fill for NetMHC/SMM/SMMPMBEC, then IterativeImputer | — |
| `v2_pvactools` | Model deployed in pVACtools v7 | IterativeImputer on all features, all samples prior to label filtering | pVACtools v7.0 |

See [`models/`](models/) for model files and [`manuscript/scripts/`](manuscript/scripts/) to reproduce training and evaluation in the manuscript.

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

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```


### Run predictions on new pVACseq-style outputs

The manuscript script `manuscript/scripts/predict.py` serves as a quick demonstration. It merges **three** pVACseq-style TSVs for a single sample (same file stem), applies the bundled imputer and label encoders, runs the trained random forest, and writes one aggregated TSV with ML predictions.

Place inputs under `manuscript/data/predict_new_case_data/` using this naming pattern:

- `<sample>.MHC_I.all_epitopes.aggregated.tsv`
- `<sample>.MHC_I.all_epitopes.tsv`
- `<sample>.MHC_II.all_epitopes.aggregated.tsv`

From the **repository root**:

```bash
python manuscript/scripts/predict.py
```

Configuration (sample name, artifact directory, artifact bundle id `numpy126`, output folder, accept/reject thresholds) is set in the `SAMPLE_NAME`, `ARTIFACTS_DIR`, `ARTIFACTS_VERSION`, `OUTPUT_DIR`, `ML_THRESHOLD_ACCEPT`, and `ML_THRESHOLD_REJECT` constants at the bottom of `manuscript/scripts/predict.py`. By default, artifacts are read from `models/v1_manuscript/` and predictions are written to `results/predictions/<sample>.MHC_I.all_epitopes.aggregated.ML_predict.tsv`.

> NOTE: Thresholds are customizable. The defaults (Accept > 0.55, Review 0.30–0.55, Reject < 0.30) were calibrated on 4 patients from the prospective test set.

### Reproduce manuscript analyses

```bash
cd manuscript/scripts
python script
```

---

## Integration with pVACtools

This model is natively integrated into **pVACtools v7**. To apply predictions during a pVACseq run:

```bash
pvacseq run ... --run-ml-predictions
```

Or as a standalone post-processing step on existing pVACseq results:

```bash
pvacseq add_ml_predictions \
  input.tsv \
  output_dir/ \
  --accept-threshold 0.55 \
  --reject-threshold 0.30
```

Predictions are displayed in **pVACview** alongside binding affinity, expression, and variant-level features. Predicted labels are pre-populated but fully editable during ITB review.

---

## Citation

If you use pVACml or the associated model in your work, please cite:

```
Yao J, Singhal K, Kiwala S, et al.
Automating immunogenomic tumor board decision-making for neoantigen cancer vaccine design.
[Journal] (2025). DOI: [pending]
```

---

## Related Resources

- [pVACtools documentation](https://pvactools.readthedocs.io/en/7.0.0_docs/)
- [pVACview interface](https://pvactools.readthedocs.io/en/latest/pvacview.html)
- [ImmunoNX pipeline](https://github.com/griffithlab/ImmunoNX_protocol)

---

## Questions and Contributions

For questions about the model or codebase, please [open an issue](https://github.com/jyao36/pVACml/issues). For questions related to the pVACtools integration, see the [pVACtools GitHub](https://github.com/griffithlab/pVACtools).