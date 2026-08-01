# Heart Disease Prediction — Model Ensemble

A weighted ensemble of three machine-learning gradient boosting models that predicts the **probability of heart
disease**. Built for
a [Kaggle binary classification competition]((https://www.kaggle.com/competitions/playground-series-s6e2)) (metric: ROC
AUC).

**Leaderboard result:** `0.95521` private / `0.95375` public.

---

## Overview

Given 13 clinical features per patient, the task is to predict a binary target
(`Presence` / `Absence` of heart disease) and output a probability for each test row. The
solution:

1. Cleans and explores the data (EDA).
2. Engineers two extra families of features (binned + digit).
3. Trains three diverse models.
4. Tunes each model's hyperparameters with **Optuna**.
5. Blends the models with **Optuna-optimized weights** into a final ensemble.
6. Predicts the test set and writes `submission.csv`.

The whole pipeline lives in a single notebook: [`main.ipynb`](main.ipynb).

---

## Dataset

| File                 | Rows    | Columns                   |
|----------------------|---------|---------------------------|
| `datasets/train.csv` | 630,000 | 14 (13 features + target) |
| `datasets/test.csv`  | 270,000 | 13 (features only)        |

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
  → Load data
  → Data audit (drop duplicates, handle inf/NaN, check class balance)
  → EDA (density / count / box plots, Spearman correlation heatmap)
  → Feature engineering (Bin + Digit features → 4 feature sets)
  → Baseline OOF models (5-fold stratified CV)
  → Model comparison (3 models × 4 feature sets) → model_comparison.csv, best_params.json
  → Hyperparameter tuning (Optuna, 50 trials per model)
  → Fit final models with best params            → models/*.joblib
  → Ensemble weight search (Optuna, 5000 trials) → best_w.json
  → Predict test set                             → submission.csv
```

### Feature engineering

Two helper functions derive extra features from the continuous columns:

- **`add_bin_features`** — discretizes each continuous column into 5 equal-width bins.
- **`add_digit_features`** — extracts the individual decimal digits (units / tens / hundreds,
  plus the first decimal of `ST depression`).

These produce four feature sets that are all benchmarked: `Base`, `Base + Bin`, `Base + Digit`,
`Base + Bin + Digit`.

### Models

All models share a small reusable framework built around a single higher-order training loop:

`train_oof(X, y, cv, fit_fold, predict_fold, name)` — owns the shared cross-validation loop, out-of-fold prediction
assembly, and per-fold + overall AUC scoring. Each backend plugs in only two callbacks: `fit_fold` (train one fold,
return an artifact) and `predict_fold` (score a fold from that artifact).

Backends (each supplies its own fit_fold / predict_fold):

- `train_xgboost` — sklearn pipeline with passthrough numerics + one-hot categoricals; early stopping via a separately
  encoded validation eval_set.
- `train_lightgbm` — native categorical splits via pandas category dtype, early
  stopping on AUC, with the best iteration stored in the artifact.
- `train_catboost` — native categorical handling via Pool and use_best_model=True.

`predict_test(fold_models, predict_fold, X_test)` — scores the test set with every fold model and averages the
predictions (bagged inference), so no single fold model has to be picked.

Training uses the GPU (the notebook was run on an NVIDIA RTX 3070 with CUDA); XGBoost is moved back to CPU before
pickling so saved models load anywhere.

Cross-validation is a shared `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, so every model's OOF AUC is
directly comparable.

---

## Results

Best OOF AUC per model after tuning (from `model_comparison.csv`):

| Model    | Best feature set   | OOF AUC |
|----------|--------------------|---------|
| CatBoost | Base + Digit       | 0.9555  |
| XGBoost  | Base + Bin + Digit | 0.9552  |
| LightGBM | Base + Bin + Digit | 0.9552  |

Final ensemble weights (Optuna, maximizing OOF AUC):

| Model    | Weight |
|----------|--------|
| CatBoost | 0.5407 |
| XGBoost  | 0.4591 |
| LightGBM | 0.0001 |

- **Ensemble OOF AUC:** `0.955608` (a small gain over the best single model, `0.9555`).

The ensemble is effectively a CatBoost + XGBoost blend; the other models add negligible weight.

---

## Project structure

```
heart_disease_kaggle/
├── main.ipynb                # Full pipeline
├── datasets/
│   ├── train.csv             # 630k rows
│   └── test.csv              # 270k rows
├── models/                   # Saved artifacts
│   ├── best_params.json      # Best params for each models
│   ├── best_w.json           # Best weight for ensemble
│   └── <Model>_folds.joblib  # Per-model fitted CV folds
├── model_comparison.csv      # OOF AUC for every model × feature set
├── submission.csv            # Final test-set predictions
├── requirements.txt          # Pinned dependencies
└── README.md
```

---

## Setup & usage

The models train on GPU. `requirements.txt` pins CUDA builds of PyTorch
(`torch==2.12.1+cu132`) and expects GPU-enabled XGBoost / LightGBM / CatBoost.

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

---

## Tech stack

pandas · numpy · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · matplotlib · seaborn · joblib
