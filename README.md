# NIFTY IV Surface Prediction

> Imputing missing implied volatility (IV) values across strikes and timestamps for NIFTY 50 options, minimising MSE against hidden ground truth.

Hosted as a Kaggle competition (Finance Club IIT Roorkee Open Projects 2026) with a 30% / 70% public–private leaderboard split.

---

## Problem

An IV surface describes how implied volatility varies across strike prices and time for NIFTY 50 options. In practice it is rarely complete — missing values arise from illiquid strikes, sparse trading, bid–ask filtering, and microstructure noise. The goal is to fill in the missing IVs (CE/Call and PE/Put, across strikes) as accurately as possible.

---

## Approach

A structured pipeline across 7 notebooks, moving from interpolation baselines to a cascaded ML + interpolation ensemble.

| #   | Notebook                       | What it does                                                              |
| --- | ------------------------------ | ------------------------------------------------------------------------- |
| 1   | `01_eda.ipynb`                 | Missingness analysis, IV smile, ATM-IV over time, CE vs PE distributions  |
| 2   | `02_feature_engineering.ipynb` | Wide → long transform, leak-free feature construction                     |
| 3   | `03_baseline.ipynb`            | Interpolation & statistical baselines (B1–B9)                             |
| 4   | `04_ml_modelling.ipynb`        | XGBoost, LightGBM, CatBoost with time-series CV                           |
| 5   | `05_DL_MLP_.ipynb`             | PyTorch MLP regressor                                                     |
| 6   | `06_ensemble_I_MLP.ipynb`      | Cascaded interpolation + MLP predictor                                    |
| 7   | `07_submission.ipynb`          | Final 3-component pipeline                                                |

### Key design choices

- **Leak-free features:** all temporal features use only past values (`.shift(1)`).
- **Expiry days handled separately:** near-expiry IVs are dominated by time-decay and deep-OTM wing explosions, where ML is unreliable — so those rows are filled by interpolation, not the MLP.
- **Floor clip:** all IV predictions clipped to a minimum of `0.005`.

---

## Feature Engineering

The wide dataset is reshaped to long format (one row per timestamp × option type × strike).

| Group                      | Features                                                                         |
| -------------------------- | -------------------------------------------------------------------------------- |
| Time                       | `hour`, `minute`, `day_of_week`, `session_progress`, `days_to_expiry`            |
| Moneyness                  | `moneyness` (K/S), `log_moneyness`, `dist_from_atm`, `dist_from_atm_pct`         |
| Structure                  | `is_ce`, `strike`, `strike_rank`                                                 |
| Cross-sectional neighbours | `iv_neighbor_±1`, `iv_neighbor_±2`, `iv_neighbor_mean`, `wide_iv_neighbor_mean`  |
| Temporal                   | `iv_lag_1`, `iv_roll_mean_5`, `iv_roll_mean_10`, `iv_roll_std_5`                 |
| Cross-area                 | `mean_ce_iv`, `mean_pe_iv` per timestamp                                         |

The final MLP uses a 13-feature subset (see `FEATURE_COLS` in `07_submission.ipynb`).

---

## Results

All MSE values are from a masked holdout / 2-fold time-series CV (train on first N days, validate on next), as reported in the notebook outputs.

### Baselines 

| Method                          | MSE          |
| ------------------------------- | ------------ |
| Forward fill                    | 0.001769     |
| Rolling mean (5)                | 0.002985     |
| Neighbour mean                  | 0.000223     |
| FFill + neighbour blend         | 0.000772     |
| Linear interp (cross-sectional) | 0.000037     |
| Cubic spline                    | 0.000162     |
| Quadratic                       | 0.000071     |
| **PCHIP**                       | **0.000034** |
| Akima                           | 0.000044     |

PCHIP was the strongest baseline — it preserves monotonicity and avoids overshoot on the IV smile.

### ML & DL models 

| Model                   | CV MSE       |
| ----------------------- | ------------ |
| XGBoost (full features) | 0.000049     |
| XGBoost (pruned)        | 0.000045     |
| LightGBM                | 0.000049     |
| CatBoost                | 0.000060     |
| **MLP (PyTorch)**       | **0.000027** |

### Cascaded ensemble 

| Variant                        | CV MSE             |
| ------------------------------ | ------------------ |
| Cascaded (Linear interp + MLP) | 0.000032           |
| Cascaded (Akima + MLP)         | best fold 0.000013 |

The cascaded approach — interpolation on expiry rows, MLP elsewhere — consistently beat both standalone ML and standalone interpolation.

---

## Final Pipeline 

A three-component system:

1. **Expiry-day fill** — cross-sectional linear interpolation in log-moneyness space (slope-damped extrapolation for deep-OTM wings) blended with a PCHIP temporal pass.
2. **MLP (non-expiry days)** — 13-feature `IVSurfaceMLP`, `StandardScaler` normalisation, CUDA if available.
3. **Fallback** — forward/back-fill for any residual NaNs.

All outputs clipped to `≥ 0.005`; result written as a Kaggle-compatible CSV.

---

## Tech Stack

| Category      | Libraries                              |
| ------------- | -------------------------------------- |
| Data          | `pandas`, `numpy`                      |
| Visualisation | `matplotlib`, `seaborn`                |
| Interpolation | `scipy` (`interp1d`, PCHIP, Akima)     |
| ML            | `xgboost`, `lightgbm`, `catboost`      |
| Deep learning | `torch`                                |
| Utilities     | `scikit-learn`                         |

---

## Running

Notebooks are designed to run in order, `01 → 07`. The final submission is produced by `07_submission.ipynb`, which expects the raw competition CSV at `../data/raw/dataset.csv` and writes to `../outputs/2_submissions/`.