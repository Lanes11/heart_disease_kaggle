# Heart Disease Prediction — Model Ensemble

A weighted ensemble of five machine-learning models that predicts the **probability of heart
disease** from routine clinical measurements. Built for a Kaggle-style binary classification
competition (metric: ROC AUC).

**Leaderboard result:** `0.95522` private / `0.95373` public.

---

## Overview

Given 13 clinical features per patient, the task is to predict a binary target
(`Presence` / `Absence` of heart disease) and output a probability for each test row. The
solution:

1. Cleans and explores the data (EDA).
2. Engineers two extra families of features (binned + per-digit).
3. Trains five diverse models inside a shared **out-of-fold (OOF) cross-validation framework**.
4. Tunes each model's hyperparameters with **Optuna**.
5. Blends the models with **Optuna-optimized weights** into a final ensemble.
6. Predicts the test set and writes `submission.csv`.

The whole pipeline lives in a single notebook: [`main.ipynb`](main.ipynb).

---

## Dataset

| File | Rows | Columns |
|---|---|---|
| `datasets/train.csv` | 630,000 | 14 (13 features + target) |
| `datasets/test.csv` | 270,000 | 13 (features only) |

- **Target:** `Heart Disease` — encoded `Presence → 1`, `Absence → 0`. Class balance ≈ 55% / 45%.
- **Continuous features:** `Age`, `BP`, `Cholesterol`, `Max HR`, `ST depression`.
- **Categorical features:** `Sex`, `Chest pain type`, `FBS over 120`, `EKG results`,
  `Exercise angina`, `Slope of ST`, `Number of vessels fluro`, `Thallium`.

The strongest signals (Spearman correlation with the target) come from `Thallium`,
`Chest pain type`, `Number of vessels fluro`, `Exercise angina`, `Max HR`, `ST depression`
and `Slope of ST`.

---

## Pipeline

```
Load data
  → Data audit (drop duplicates, handle inf/NaN, check class balance)
  → EDA (density / count / box plots, Spearman correlation heatmap)
  → Feature engineering (Bin + Digit features → 4 feature sets)
  → Baseline OOF framework (5-fold stratified CV)
  → Model comparison (5 models × 4 feature sets)  → model_comparison.csv
  → Hyperparameter tuning (Optuna, 50 trials per model)
  → Fit final models with best params            → models/*.joblib
  → Ensemble weight search (Optuna, 5000 trials)  → models/ensemble_*.joblib
  → Predict test set                              → submission.csv
```

### Feature engineering

Two helper functions derive extra features from the continuous columns:

- **`add_bin_features`** — discretizes each continuous column into 5 equal-width bins
  (stored as pandas `category` dtype for native categorical handling in trees).
- **`add_digit_features`** — extracts the individual decimal digits (units / tens / hundreds,
  plus the first decimal of `ST depression`), which can expose recording artifacts in the data.

These produce four feature sets that are all benchmarked: `Base`, `Base + Bin`, `Base + Digit`,
`Base + Bin + Digit`.

### Models & architecture

All models share a small reusable framework built around an abstract base class:

- **`BaseOOFModel`** (abstract) — owns the 5-fold stratified CV loop, out-of-fold prediction
  assembly, AUC scoring, and `.joblib` save/load. Subclasses implement only per-fold
  `_fit_fold` / `_predict_fold`.
- **Backends:**
  - `LogisticRegressionOOFModel` — `StandardScaler` + `OneHotEncoder` sklearn pipeline.
  - `XGBoostOOFModel` — sklearn pipeline with passthrough numerics + one-hot categoricals,
    early stopping via an encoded eval set.
  - `LightGBMOOFModel` — native categorical splits via pandas `category` dtype + early stopping.
  - `CatBoostOOFModel` — native categorical handling via `Pool`.
  - `RealMLPOOFModel` — a tabular neural network (`pytabkit`) with a per-fold validation set.
- **`_BACKEND_REGISTRY` + `load_oof_model(path)`** — reload any saved model from disk, choosing
  the right subclass from the `backend` tag stored in the payload.

Training uses the **GPU** (the notebook was run on an NVIDIA RTX 3070 with CUDA); XGBoost is
moved back to CPU before pickling so saved models load anywhere.

Cross-validation is a shared `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, so
every model's OOF AUC is directly comparable.

---

## Results

Best OOF AUC per model after tuning (from `model_comparison.csv`):

| Model | Best feature set | OOF AUC  |
|---|---|----------|
| CatBoost | Base + Bin + Digit | 0.955545 |
| XGBoost | Base | 0.95553  |
| LightGBM | Base + Digit | 0.95542  |
| RealMLP | Base + Bin + Digit | 0.95493  |
| Logistic Regression | Base + Bin + Digit | 0.95332  |

Final ensemble weights (Optuna, maximizing OOF AUC):

| Model | Weight |
|---|---|
| CatBoost | 0.5292 |
| XGBoost | 0.4702 |
| LightGBM | 0.0003 |
| RealMLP | 0.0002 |
| Logistic Regression | 0.0000 |

- **Ensemble OOF AUC:** `0.955588` (a small but real gain over the best single model, `0.955545`).
- **Leaderboard:** `0.95522` private / `0.95373` public.

The ensemble is effectively a CatBoost + XGBoost blend; the other models add negligible weight.

---

## Project structure

```
heart_disease_kaggle/
├── main.ipynb              # Full pipeline: EDA → features → models → ensemble → submission
├── datasets/
│   ├── train.csv           # 630k labeled rows
│   └── test.csv            # 270k rows to predict
├── models/                 # Saved artifacts
│   ├── best_params.json    # Tuned hyperparameters per model
│   ├── <Model> (...)_oof_<auc>.joblib   # Per-model fitted CV folds
│   └── ensemble_oof_<auc>.joblib        # Ensemble weights + member references
├── model_comparison.csv    # OOF AUC for every model × feature set
├── submission.csv          # Final test-set predictions
├── requirements.txt        # Pinned dependencies
└── README.md
```

---

## Setup & usage

The models train on GPU. `requirements.txt` pins CUDA builds of PyTorch
(`torch==2.12.1+cu132`) and expects GPU-enabled XGBoost / LightGBM / CatBoost. To run on CPU,
change `device`/`task_type`/`device_type` in the parameter dictionaries in `main.ipynb`.

```bash
# From the project root
python -m venv .venv
source .venv/Scripts/activate      # Windows (bash); use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Then open and run the notebook top to bottom:

```bash
jupyter notebook main.ipynb
```

Running all cells reproduces the model comparison (`model_comparison.csv`), retrains and saves
the five models plus the ensemble under `models/`, and writes the final `submission.csv`
(270,000 predicted probabilities).

---

## Tech stack

pandas · numpy · scikit-learn · XGBoost · LightGBM · CatBoost · pytabkit (RealMLP) · Optuna ·
matplotlib · seaborn · joblib
