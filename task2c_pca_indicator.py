"""
Task 2(c): Indicator Regression on PCA Latent Factors
=====================================================
Instead of regressing Y=1 on raw portfolio returns, first reduce
dimensionality via PCA, then regress Y=1 on the first K factor returns.

For each K: fit PCA on train, project train/test onto K factors,
run Ridge/Lasso indicator regression, compute OOS Sharpe.

Data prep identical to Task 2(b):
  1. Drop columns with >30% missing
  2. Drop rows with any remaining NaN
  3. Split pre-2004 (train) / 2004+ (test)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE DATA (same as Task 2b)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading lsret.csv...")
lsret = pd.read_csv("lsret.csv")
lsret["date"] = pd.to_datetime(lsret["date"])
lsret = lsret.sort_values("date").reset_index(drop=True)

ret_cols = [c for c in lsret.columns if c != "date"]

# Step 1: Drop columns with >30% missing
col_miss = lsret[ret_cols].isnull().mean()
keep_cols = col_miss[col_miss <= 0.3].index.tolist()
print(f"  Columns: {len(ret_cols)} -> {len(keep_cols)} (dropped {len(ret_cols)-len(keep_cols)} with >30% NaN)")

# Step 2: Drop rows with any remaining NaN
lsret_clean = lsret[["date"] + keep_cols].dropna().reset_index(drop=True)
print(f"  Rows: {lsret.shape[0]} -> {lsret_clean.shape[0]} (dropped {lsret.shape[0]-lsret_clean.shape[0]} with NaN)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
train_mask = lsret_clean["date"] < "2004-01-01"
test_mask  = lsret_clean["date"] >= "2004-01-01"

R_train = lsret_clean.loc[train_mask, keep_cols].values
R_test  = lsret_clean.loc[test_mask,  keep_cols].values
dates_test = lsret_clean.loc[test_mask, "date"].values

print(f"  Train: {R_train.shape[0]} months x {R_train.shape[1]} portfolios")
print(f"  Test:  {R_test.shape[0]} months x {R_test.shape[1]} portfolios")

P = R_train.shape[1]  # number of portfolios


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT PCA ON TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting PCA on training data...")
pca = PCA(n_components=P)
pca.fit(R_train)

# Factor returns: project RAW returns onto PC loadings (no demeaning).
# pca.transform() subtracts the mean, which kills the mean signal that
# the indicator regression needs. Use direct projection instead.
V = pca.components_  # (P, P) - each row is a loading vector
F_train_all = R_train @ V.T  # (T_train, P)
F_test_all  = R_test  @ V.T  # (T_test, P)

cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"  PC1 explains: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PCs for 80%: {np.searchsorted(cumvar, 0.80) + 1}")
print(f"  PCs for 90%: {np.searchsorted(cumvar, 0.90) + 1}")
print(f"  PCs for 95%: {np.searchsorted(cumvar, 0.95) + 1}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HELPER
# ══════════════════════════════════════════════════════════════════════════════
def annualized_sharpe(mr):
    m, s = mr.mean(), mr.std(ddof=1)
    return (m / s) * np.sqrt(12) if s > 0 and not np.isnan(s) else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOOP OVER FACTOR COUNTS
# ══════════════════════════════════════════════════════════════════════════════
K_values = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, P]
K_values = [k for k in K_values if k <= P]  # ensure within bounds

Y_train = np.ones(R_train.shape[0])
tscv = TimeSeriesSplit(n_splits=5)
ridge_alphas = np.logspace(-4, 8, 100)

results = []

print("\n" + "=" * 70)
print(f"{'K':>4s}  {'Ridge SR':>10s}  {'Lasso SR':>10s}  {'Ridge alpha':>12s}  "
      f"{'Lasso alpha':>12s}  {'Lasso nz':>9s}")
print("-" * 70)

for K in K_values:
    F_train = F_train_all[:, :K]
    F_test  = F_test_all[:, :K]

    # Ridge
    ridge_cv = RidgeCV(alphas=ridge_alphas, cv=tscv, fit_intercept=False)
    ridge_cv.fit(F_train, Y_train)
    port_ridge = F_test @ ridge_cv.coef_
    sr_ridge = annualized_sharpe(port_ridge)

    # Lasso
    lasso_cv = LassoCV(alphas=np.logspace(-6, 2, 100), cv=tscv,
                        fit_intercept=False, max_iter=50000)
    lasso_cv.fit(F_train, Y_train)
    port_lasso = F_test @ lasso_cv.coef_
    sr_lasso = annualized_sharpe(port_lasso)
    n_nz = np.sum(np.abs(lasso_cv.coef_) > 1e-10)

    results.append({
        "K": K,
        "sr_ridge": sr_ridge,
        "sr_lasso": sr_lasso,
        "ridge_alpha": ridge_cv.alpha_,
        "lasso_alpha": lasso_cv.alpha_,
        "lasso_nz": n_nz,
        "ridge_mean": port_ridge.mean() * 12,
        "ridge_vol": port_ridge.std() * np.sqrt(12),
        "lasso_mean": port_lasso.mean() * 12,
        "lasso_vol": port_lasso.std() * np.sqrt(12),
        "cumvar": cumvar[K - 1],
        "port_ridge": port_ridge,
        "port_lasso": port_lasso,
    })

    print(f"{K:4d}  {sr_ridge:+10.3f}  {sr_lasso:+10.3f}  {ridge_cv.alpha_:12.2e}  "
          f"{lasso_cv.alpha_:12.2e}  {n_nz:5d}/{K}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIND OPTIMAL K
# ══════════════════════════════════════════════════════════════════════════════
best_ridge_idx = np.argmax([r["sr_ridge"] for r in results])
best_lasso_idx = np.argmax([r["sr_lasso"] for r in results])
best_ridge_K = results[best_ridge_idx]["K"]
best_lasso_K = results[best_lasso_idx]["K"]

print(f"\n  Best Ridge: K={best_ridge_K}, SR={results[best_ridge_idx]['sr_ridge']:+.3f}")
print(f"  Best Lasso: K={best_lasso_K}, SR={results[best_lasso_idx]['sr_lasso']:+.3f}")

# Compare to Task 2(b) raw returns (K=P with no PCA, same as last entry)
print(f"\n  Task 2(b) baseline (raw, no PCA):")
print(f"    Ridge SR: +1.717, Lasso SR: +1.419")


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

Ks = [r["K"] for r in results]
sr_ridges = [r["sr_ridge"] for r in results]
sr_lassos = [r["sr_lasso"] for r in results]

# --- Plot 1: SR vs K ---
ax = axes[0]
ax.plot(Ks, sr_ridges, "o-", color="#3498db", linewidth=2, markersize=6, label="Ridge")
ax.plot(Ks, sr_lassos, "s-", color="#e74c3c", linewidth=2, markersize=6, label="Lasso")

# Mark optimal points
ax.axvline(x=best_ridge_K, color="#3498db", linestyle="--", alpha=0.4)
ax.axvline(x=best_lasso_K, color="#e74c3c", linestyle="--", alpha=0.4)
ax.scatter([best_ridge_K], [results[best_ridge_idx]["sr_ridge"]],
           color="#3498db", s=150, zorder=5, edgecolors="black", linewidth=1.5)
ax.scatter([best_lasso_K], [results[best_lasso_idx]["sr_lasso"]],
           color="#e74c3c", s=150, zorder=5, edgecolors="black", linewidth=1.5)

# Task 2(b) baselines
ax.axhline(y=1.717, color="#3498db", linestyle=":", alpha=0.5, label="Ridge raw (2b)")
ax.axhline(y=1.419, color="#e74c3c", linestyle=":", alpha=0.5, label="Lasso raw (2b)")

ax.set_xlabel("Number of PCA Factors (K)", fontsize=11)
ax.set_ylabel("OOS Annualized Sharpe Ratio", fontsize=11)
ax.set_title("Sharpe Ratio vs. Number of PCA Factors", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, P + 5)

# --- Plot 2: SR vs cumulative variance explained ---
ax = axes[1]
cumvars = [r["cumvar"] for r in results]
ax.plot(cumvars, sr_ridges, "o-", color="#3498db", linewidth=2, markersize=6, label="Ridge")
ax.plot(cumvars, sr_lassos, "s-", color="#e74c3c", linewidth=2, markersize=6, label="Lasso")

ax.set_xlabel("Cumulative Variance Explained", fontsize=11)
ax.set_ylabel("OOS Annualized Sharpe Ratio", fontsize=11)
ax.set_title("Sharpe Ratio vs. Variance Explained", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 3: Cumulative returns for best K vs raw ---
ax = axes[2]

# Best Lasso PCA
best_res = results[best_lasso_idx]
cum_best = (1 + best_res["port_lasso"]).cumprod()
ax.plot(pd.to_datetime(dates_test), cum_best,
        label=f"Lasso PCA K={best_lasso_K} (SR={best_res['sr_lasso']:+.2f})",
        linewidth=2, color="#e74c3c")

# Best Ridge PCA
best_res_r = results[best_ridge_idx]
cum_best_r = (1 + best_res_r["port_ridge"]).cumprod()
ax.plot(pd.to_datetime(dates_test), cum_best_r,
        label=f"Ridge PCA K={best_ridge_K} (SR={best_res_r['sr_ridge']:+.2f})",
        linewidth=2, color="#3498db")

# Raw baseline (K=P, last entry)
raw_res = results[-1]
cum_raw_lasso = (1 + raw_res["port_lasso"]).cumprod()
ax.plot(pd.to_datetime(dates_test), cum_raw_lasso,
        label=f"Lasso raw K={P} (SR={raw_res['sr_lasso']:+.2f})",
        linewidth=1.5, color="#e74c3c", linestyle="--", alpha=0.6)

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Cumulative Return ($1 invested)", fontsize=11)
ax.set_title("OOS Cumulative Returns: Best PCA vs Raw", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task2c_pca_indicator.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved task2c_pca_indicator.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DETAILED SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY: Indicator Regression on PCA Factors - OOS Results")
print("=" * 80)

print(f"\n{'K':>4s}  {'CumVar':>7s}  {'Ridge SR':>10s}  {'Ridge Mean':>12s}  "
      f"{'Ridge Vol':>11s}  {'Lasso SR':>10s}  {'Lasso Mean':>12s}  {'Lasso Vol':>11s}  {'Lasso nz':>8s}")
print("-" * 95)
for r in results:
    sr_l = f"{r['sr_lasso']:+.3f}" if not np.isnan(r['sr_lasso']) else "NaN"
    print(f"{r['K']:4d}  {r['cumvar']:7.1%}  {r['sr_ridge']:+10.3f}  "
          f"{r['ridge_mean']:+12.6f}  {r['ridge_vol']:11.6f}  "
          f"{sr_l:>10s}  {r['lasso_mean']:+12.6f}  {r['lasso_vol']:11.6f}  "
          f"{r['lasso_nz']:>4d}/{r['K']}")

print(f"\n  Optimal K:")
print(f"    Ridge: K={best_ridge_K} (SR={results[best_ridge_idx]['sr_ridge']:+.3f}, "
      f"CumVar={results[best_ridge_idx]['cumvar']:.1%})")
print(f"    Lasso: K={best_lasso_K} (SR={results[best_lasso_idx]['sr_lasso']:+.3f}, "
      f"CumVar={results[best_lasso_idx]['cumvar']:.1%})")
print()
