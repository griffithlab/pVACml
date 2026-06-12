# pVACtools 7.0 ML artifacts (staging)

Files in this directory are the **same bundle** that lives in the **pVACtools** repository under:

**[griffithlab/pVACtools — `pvactools/supporting_files/ml_model_artifacts`](https://github.com/griffithlab/pVACtools/tree/master/pvactools/supporting_files/ml_model_artifacts)**

When you refresh the model from **pVACml**, copy the updated pickles/joblib from here into that folder in a **pVACtools** checkout (or open a PR that updates those paths). Filenames should stay aligned with what pVACtools loads (for example the `*_numpy126.*` artifacts shown on GitHub).

There is no automatic cross-repo link in Git; this README documents the intended **one-to-one path mapping**:

| This repo (`ITB_Automation_ML_Predictor`) | pVACtools (`griffithlab/pVACtools`) |
|------------------------------------------|--------------------------------------|
| `model/pvactools7.0_model/` | `pvactools/supporting_files/ml_model_artifacts/` |
