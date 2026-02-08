"""
Task 2(b): Mean-Variance Efficient Portfolio via Indicator Regression
=====================================================================
Regress Y = 1 on portfolio returns R_t (no intercept):
    1 = beta' R_t + epsilon

beta_hat ~ Sigma^{-1} mu  (tangency portfolio weights)

Data prep (consistent with Task 2a PCA prepare_for_pca):
  1. Drop columns with >30% missing
  2. Drop rows with any remaining NaN
  3. Split pre-2004 (train) / 2004+ (test)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading lsret.csv...")
lsret = pd.read_csv("lsret.csv")
lsret["date"] = pd.to_datetime(lsret["date"])
lsret = lsret.sort_values("date").reset_index(drop=True)

ret_cols = [c for c in lsret.columns if c != "date"]

print(f"  Raw: {lsret.shape[0]} months, {len(ret_cols)} portfolios")
print(f"  Date range: {lsret['date'].iloc[0].date()} to {lsret['date'].iloc[-1].date()}")
n_nan = lsret[ret_cols].isna().sum().sum()
print(f"  NaN cells: {n_nan:,} / {lsret[ret_cols].size:,} "
      f"({n_nan / lsret[ret_cols].size * 100:.1f}%)")

# Step 1: Drop columns with >30% missing (same as Task 2a PCA prepare_for_pca)
col_miss = lsret[ret_cols].isnull().mean()
keep_cols = col_miss[col_miss <= 0.3].index.tolist()
dropped_cols = [c for c in ret_cols if c not in keep_cols]
print(f"\n  Step 1 - Drop columns with >30% NaN:")
print(f"    Dropped: {len(dropped_cols)} columns")
print(f"    Retained: {len(keep_cols)} columns")

# Step 2: Drop rows with any remaining NaN
lsret_clean = lsret[["date"] + keep_cols].dropna().reset_index(drop=True)
n_dropped_rows = lsret.shape[0] - lsret_clean.shape[0]
print(f"\n  Step 2 - Drop rows with remaining NaN:")
print(f"    Dropped: {n_dropped_rows} rows")
print(f"    Final: {lsret_clean.shape[0]} months x {len(keep_cols)} portfolios")
print(f"    Date range: {lsret_clean['date'].iloc[0].date()} to "
      f"{lsret_clean['date'].iloc[-1].date()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
train_mask = lsret_clean["date"] < "2004-01-01"
test_mask  = lsret_clean["date"] >= "2004-01-01"

R_train = lsret_clean.loc[train_mask, keep_cols].values
R_test  = lsret_clean.loc[test_mask,  keep_cols].values
dates_train = lsret_clean.loc[train_mask, "date"].values
dates_test  = lsret_clean.loc[test_mask,  "date"].values

Y_train = np.ones(R_train.shape[0])

print(f"\n  Train: {R_train.shape[0]} months "
      f"({str(dates_train[0])[:10]} to {str(dates_train[-1])[:10]})")
print(f"  Test:  {R_test.shape[0]} months "
      f"({str(dates_test[0])[:10]} to {str(dates_test[-1])[:10]})")
print(f"  Portfolios: {R_train.shape[1]}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. HELPER
# ══════════════════════════════════════════════════════════════════════════════
def annualized_sharpe(mr):
    m, s = mr.mean(), mr.std(ddof=1)
    return (m / s) * np.sqrt(12) if s > 0 and not np.isnan(s) else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 4. RIDGE INDICATOR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RIDGE INDICATOR REGRESSION (no intercept)")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=5)
ridge_alphas = np.logspace(-4, 8, 150)

ridge_cv = RidgeCV(alphas=ridge_alphas, cv=tscv, fit_intercept=False)
ridge_cv.fit(R_train, Y_train)

w_ridge = ridge_cv.coef_
port_ridge_oos = R_test @ w_ridge

sr_ridge = annualized_sharpe(port_ridge_oos)
ann_mean_ridge = port_ridge_oos.mean() * 12
ann_vol_ridge  = port_ridge_oos.std() * np.sqrt(12)

print(f"  Best alpha: {ridge_cv.alpha_:.4e}")
print(f"  Non-zero weights: {np.sum(np.abs(w_ridge) > 1e-10)}/{len(keep_cols)}")
print(f"  Weight norm (L2): {np.linalg.norm(w_ridge):.6f}")
print(f"\n  OOS Portfolio:")
print(f"    Ann. Mean Return:  {ann_mean_ridge:+.4f}")
print(f"    Ann. Volatility:   {ann_vol_ridge:.4f}")
print(f"    Ann. Sharpe Ratio: {sr_ridge:+.3f}")

top_ridge = np.argsort(np.abs(w_ridge))[::-1][:10]
print(f"\n  Top 10 weights (by magnitude):")
for i, idx in enumerate(top_ridge, 1):
    print(f"    {i:2d}. {keep_cols[idx]:30s}  w={w_ridge[idx]:+.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LASSO INDICATOR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LASSO INDICATOR REGRESSION (no intercept)")
print("=" * 70)

lasso_alphas = np.logspace(-5, 1, 100)

lasso_cv = LassoCV(alphas=lasso_alphas, cv=tscv, fit_intercept=False, max_iter=50000)
lasso_cv.fit(R_train, Y_train)

w_lasso = lasso_cv.coef_
n_nonzero = np.sum(np.abs(w_lasso) > 1e-10)
port_lasso_oos = R_test @ w_lasso

sr_lasso = annualized_sharpe(port_lasso_oos)
ann_mean_lasso = port_lasso_oos.mean() * 12
ann_vol_lasso  = port_lasso_oos.std() * np.sqrt(12)

print(f"  Best alpha: {lasso_cv.alpha_:.4e}")
print(f"  Non-zero weights: {n_nonzero}/{len(keep_cols)}")
print(f"\n  OOS Portfolio:")
print(f"    Ann. Mean Return:  {ann_mean_lasso:+.4f}")
print(f"    Ann. Volatility:   {ann_vol_lasso:.4f}")
print(f"    Ann. Sharpe Ratio: {sr_lasso:+.3f}")

nonzero_idx = np.where(np.abs(w_lasso) > 1e-10)[0]
print(f"\n  Selected portfolios ({n_nonzero}):")
for idx in nonzero_idx[np.argsort(np.abs(w_lasso[nonzero_idx]))[::-1]]:
    print(f"    {keep_cols[idx]:30s}  w={w_lasso[idx]:+.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 1/N BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
w_equal = np.ones(len(keep_cols)) / len(keep_cols)
port_equal_oos = R_test @ w_equal
sr_equal = annualized_sharpe(port_equal_oos)
ann_mean_equal = port_equal_oos.mean() * 12
ann_vol_equal  = port_equal_oos.std() * np.sqrt(12)


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Cumulative OOS returns
ax = axes[0]
for port, label, color in [
    (port_ridge_oos, f"Ridge (SR={sr_ridge:+.2f})", "#3498db"),
    (port_lasso_oos, f"Lasso (SR={sr_lasso:+.2f})", "#e74c3c"),
    (port_equal_oos, f"1/N Equal (SR={sr_equal:+.2f})", "#95a5a6"),
]:
    cum = (1 + port).cumprod()
    ax.plot(pd.to_datetime(dates_test), cum, label=label, linewidth=1.5, color=color)

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Cumulative Return ($1 invested)", fontsize=11)
ax.set_title("OOS Cumulative Returns: Indicator Regression", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# Plot 2: Weight distribution
ax = axes[1]
w_ridge_sorted = np.sort(w_ridge)[::-1]
w_lasso_sorted = np.sort(w_lasso)[::-1]
x = np.arange(len(keep_cols))
ax.bar(x, w_ridge_sorted, alpha=0.5,
       label=f"Ridge ({np.sum(np.abs(w_ridge)>1e-10)} weights)",
       color="#3498db", width=1.0)
ax.bar(x, w_lasso_sorted, alpha=0.7,
       label=f"Lasso ({n_nonzero} weights)",
       color="#e74c3c", width=0.6)
ax.set_xlabel("Portfolio (sorted by weight)", fontsize=11)
ax.set_ylabel("Weight", fontsize=11)
ax.set_title("Portfolio Weights (sorted descending)", fontsize=12)
ax.legend(fontsize=10)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("task2b_indicator_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved task2b_indicator_regression.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: Indicator Regression OOS Results")
print("=" * 70)

summary = pd.DataFrame([
    {"Method": "Ridge", "Alpha": f"{ridge_cv.alpha_:.2e}",
     "Non-zero": f"{np.sum(np.abs(w_ridge)>1e-10)}/{len(keep_cols)}",
     "Ann.Mean": f"{ann_mean_ridge:+.4f}", "Ann.Vol": f"{ann_vol_ridge:.4f}",
     "Sharpe": f"{sr_ridge:+.3f}"},
    {"Method": "Lasso", "Alpha": f"{lasso_cv.alpha_:.2e}",
     "Non-zero": f"{n_nonzero}/{len(keep_cols)}",
     "Ann.Mean": f"{ann_mean_lasso:+.4f}", "Ann.Vol": f"{ann_vol_lasso:.4f}",
     "Sharpe": f"{sr_lasso:+.3f}"},
    {"Method": "1/N Equal", "Alpha": "-",
     "Non-zero": f"{len(keep_cols)}/{len(keep_cols)}",
     "Ann.Mean": f"{ann_mean_equal:+.4f}", "Ann.Vol": f"{ann_vol_equal:.4f}",
     "Sharpe": f"{sr_equal:+.3f}"},
])
print(summary.to_string(index=False))

print(f"\n  Data prep (consistent with Task 2a PCA):")
print(f"    Columns dropped (>30% NaN): {len(dropped_cols)}")
print(f"    Rows dropped (remaining NaN): {n_dropped_rows}")
print(f"    Final: {lsret_clean.shape[0]} months x {len(keep_cols)} portfolios")
print(f"    Train: {R_train.shape[0]} months, Test: {R_test.shape[0]} months")
print()
