# IV Surface Reconstruct

A machine learning pipeline to reconstruct missing **Implied Volatility (IV)** values across a full NIFTY options surface. The project tackles a real-world data imputation problem where options data has gaps due to illiquid strikes, halted trading, or feed failures — and produces a clean, complete IV surface ready for downstream quant analysis.

Built for a Kaggle-style competition setting using NIFTY weekly expiry contracts (January 2026 expiry).

---

## Problem Statement

The implied volatility surface is a critical input for options pricing, hedging, and risk management. In practice, raw options data is never fully observed — deep out-of-the-money strikes trade infrequently, certain intervals have no quotes, and data vendors have gaps.

This project imputes those missing IV values across **975 rows and hundreds of option columns** to produce a complete surface, enabling:
- Accurate Greeks computation across all strikes
- Volatility smile / skew analysis
- Risk-neutral density estimation
- Arbitrage-free surface construction

---

## Approach

The pipeline uses a **two-stage hybrid strategy** that treats expiry and non-expiry days differently, since their IV dynamics are fundamentally distinct.

### Stage 1 — Expiry Day (Cross-Sectional + PCHIP)

On the expiry date, IV behaves erratically — time value collapses, illiquidity spikes, and standard time-series patterns break down. A purely data-driven model trained on non-expiry data would extrapolate poorly here. Instead:

- Missing strikes are filled using **cross-sectional interpolation** — leveraging the fact that the IV smile is smooth across strikes at any given moment
- **PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)** is used rather than linear interpolation, because it preserves local monotonicity and avoids overshooting — critical for a well-behaved volatility smile
- This fill is applied only to the final expiry date (75 rows out of 975)

### Stage 2 — Non-Expiry Days (MLP Neural Network)

For the remaining 900 non-expiry rows, a neural network learns the structural relationship between observable features and IV:

- A **Multi-Layer Perceptron (MLP)** is trained on all observed (non-missing) IV data points in long format
- Input features include: strike price, moneyness (strike / underlying), time-to-expiry, underlying price, log-moneyness, and time-of-day features
- The model is trained with a **90/10 split** used purely as an early-stopping signal — the full observed dataset is used for training, with no held-out test leakage
- Best epoch is selected based on validation loss; training runs for up to 200 epochs
- The trained model then predicts **4,885 missing IV cells** across non-expiry rows

### Stage 3 — Final Assembly & Safety Fills

- MLP predictions and PCHIP fills are merged back into a unified **wide-format DataFrame** aligned by datetime
- A vectorized merge strategy (no slow Python loops) ensures correct row alignment
- Any cells still missing after both fills receive a **forward-fill then backward-fill** as a last resort
- All IV values are clipped to a floor of `0.005` to prevent non-physical zero or negative volatilities

---

## Project Structure

```
iv_surface_reconstruct/
│
├── notebooks/                      # Jupyter notebooks with full pipeline
│   └── iv_surface_reconstruct.ipynb
│
├── eda_outputs/                    # EDA plots, heatmaps, and analysis artifacts
│
├── outputs/
│   ├── iv_filled_dataset.csv       # Full wide-format IV surface (filled)
│   └── submission.csv              # Kaggle-format submission file
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/ynyetname/iv_surface_reconstruct.git
cd iv_surface_reconstruct
```

### 2. Create and activate a conda environment
```bash
conda create -n iv_surface_reconstruct python=3.10 -y
conda activate iv_surface_reconstruct
```

### 3. Install dependencies
```bash
pip install numpy pandas scipy torch scikit-learn tqdm jupyter
```

---

## Reproducing the Results

Run the following commands end-to-end to reproduce the outputs from scratch:

```bash
# Step 1 — Clone and enter the repo
git clone https://github.com/ynyetname/iv_surface_reconstruct.git
cd iv_surface_reconstruct

# Step 2 — Create and activate environment
conda create -n iv_surface_reconstruct python=3.10 -y
conda activate iv_surface_reconstruct

# Step 3 — Install dependencies
pip install numpy pandas scipy torch scikit-learn tqdm jupyter

# Step 4 — Launch the notebook
jupyter notebook notebooks/iv_surface_reconstruct.ipynb
```

Once the notebook is open in your browser, go to **Kernel → Restart & Run All** to execute the complete pipeline from data loading to submission CSV generation.

> **Expected output:** The terminal will log progress through each stage — data loading, expiry fill, MLP training (with best epoch), prediction count, IV range, and final file paths.

---

## Output Format

### `iv_filled_dataset.csv`
Wide-format DataFrame with one row per datetime timestamp and one column per option contract. Every cell is filled — no NaNs.

| datetime | underlying_price | NIFTY27JAN2625900CE | NIFTY27JAN2625900PE | ... |
|---|---|---|---|---|
| 27-01-2026 09:15 | 23500.0 | 0.1823 | 0.2104 | ... |
| 27-01-2026 09:20 | 23512.5 | 0.1796 | 0.2089 | ... |

### `submission.csv`
Long-format Kaggle submission — one row per originally-missing cell, identified by a composite key of datetime and column name.

| id | value |
|---|---|
| 27-01-2026 09:15\|\|NIFTY27JAN2625900CE | 0.1823 |
| 27-01-2026 09:20\|\|NIFTY27JAN2626000PE | 0.2341 |

---

## Key Design Decisions

- **Expiry treated separately** — expiry-day IV dynamics (time decay, illiquidity) are fundamentally different from non-expiry; mixing them would degrade model quality on both
- **No data leakage** — the MLP is trained exclusively on observed (non-missing) IV values; missing cells are never seen during training
- **PCHIP over linear interpolation** — preserves monotonicity of the volatility smile and avoids unnatural overshoots between strikes
- **Vectorized merge over row loops** — predictions are written back to the wide DataFrame using datetime-aligned merges, making the assembly step fast even at scale
- **Duplicate deduplication** — raw data is deduplicated on `(datetime, symbol)` before pivoting to prevent pandas from silently creating `.1` duplicate column suffixes, which would cause KeyErrors at submission time
- **IV floor clipping** — all predictions are clipped to `≥ 0.005` to ensure the surface is arbitrage-free and numerically stable

---

## Author

**ynyetname** — [github.com/ynyetname](https://github.com/ynyetname)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
