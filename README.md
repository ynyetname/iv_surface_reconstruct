# IV Surface Reconstruct

A machine learning pipeline to reconstruct missing **Implied Volatility (IV)** values across a full options surface for NIFTY weekly expiry contracts. Uses a hybrid approach combining cross-sectional interpolation, PCHIP spline filling, and a PyTorch MLP for non-expiry cells.

---

## Problem Statement

Real-world options data has missing IV values due to illiquid strikes, halted trading, or data feed gaps. This project imputes those missing values to produce a **complete IV surface** — enabling downstream tasks like volatility modelling, Greeks computation, and risk analysis.

---

## Approach

The pipeline uses a two-stage hybrid strategy:

### Stage 1 — Expiry Day (Cross-Sectional + PCHIP)
- On the expiry date, IV behaves differently (rapid time-decay, illiquidity)
- Missing values are filled using **cross-sectional interpolation** across strikes
- PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) ensures smooth, monotone fills

### Stage 2 — Non-Expiry Days (MLP Neural Network)
- A **Multi-Layer Perceptron (MLP)** is trained on all observed non-expiry IV data
- Features include strike, moneyness, time-to-expiry, underlying price, and temporal features
- The trained model predicts missing IV cells across all non-expiry rows
- A 90/10 split is used for early-stopping signal during training

### Final Assembly
- MLP predictions and PCHIP fills are merged into a unified wide-format DataFrame
- Any remaining NaNs are handled via forward-fill / backward-fill as a safety net
- All IV values are clipped to a minimum of `0.005` to avoid non-physical negatives

---

## Project Structure

```
iv_surface_reconstruct/
│
├── notebooks/              # Jupyter notebooks with full pipeline
├── eda_outputs/            # EDA plots and analysis artifacts
├── outputs/
│   ├── iv_filled_dataset.csv   # Full wide-format IV surface (filled)
│   └── submission.csv          # Kaggle-format submission file
│
├── requirements.txt        # Python dependencies
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

### 2. Create and activate conda environment
```bash
conda create -n iv_surface_reconstruct python=3.10
conda activate iv_surface_reconstruct
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
torch>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## Usage

Open and run the main notebook in `notebooks/`:

```bash
jupyter notebook notebooks/iv_surface_reconstruct.ipynb
```

The pipeline will:
1. Load raw options data from `RAW_CSV`
2. Fill expiry day IVs using PCHIP interpolation
3. Train an MLP on observed non-expiry IVs
4. Predict and fill all missing non-expiry cells
5. Export `iv_filled_dataset.csv` and `submission.csv` to `outputs/`

---

## Output Format

### `iv_filled_dataset.csv`
Wide-format DataFrame with one row per datetime and one column per option contract:

| datetime | underlying_price | NIFTY27JAN2625900CE | NIFTY27JAN2625900PE | ... |
|---|---|---|---|---|
| 27-01-2026 09:15 | 23500.0 | 0.1823 | 0.2104 | ... |

### `submission.csv`
Long-format Kaggle submission file:

| id | value |
|---|---|
| 27-01-2026 09:15\|\|NIFTY27JAN2625900CE | 0.1823 |

---

## Key Design Decisions

- **No data leakage** — MLP is trained only on observed (non-missing) IV values
- **Expiry handled separately** — expiry-day dynamics differ significantly from non-expiry
- **Vectorized merge** — predictions are written back using datetime-aligned merges, not slow row loops
- **Duplicate safety** — raw data is deduplicated on `(datetime, symbol)` before pivoting to prevent pandas `.1` column suffix bugs

---

## Author

**ynyetname** — [github.com/ynyetname](https://github.com/ynyetname)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
