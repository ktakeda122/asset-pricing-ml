"""
Task 2(d): Compare Indicator Regression across Three Universes
==============================================================
Apply raw-return and PCA-based indicator regression to:
  1. lsret.csv  (pre-computed L/S portfolios)
  2. Large Cap  (rank-weighted L/S from largeml.pq)
  3. Small Cap  (rank-weighted L/S from smallml.pq)

For each universe:
  - Data prep: drop cols >30% NaN, drop rows with remaining NaN
  - Raw indicator regression (Ridge/Lasso)
  - PCA indicator regression across K factors
  - Report OOS Sharpe ratios
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
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def annualized_sharpe(mr):
    m, s = mr.mean(), mr.std(ddof=1)
    return (m / s) * np.sqrt(12) if s > 0 and not np.isnan(s) else np.nan


def yyyymm_to_datetime(yyyymm_series):
    """Convert yyyymm integers (e.g. 200401) to proper datetime objects."""
    return pd.to_datetime(
        yyyymm_series.astype(int).astype(str), format="%Y%m"
    )


def build_ls_returns(filepath, label):
    """Build rank-weighted L/S portfolio returns for each characteristic.
    Returns a DataFrame with a proper datetime 'date' column."""
    df = pd.read_parquet(filepath)
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    feature_cols = [c for c in df.columns if c not in ["permno", "yyyymm", "ret"]]
    df = df.dropna(subset=["ret"])

    all_ports = {}
    for char in feature_cols:
        sub = df[["yyyymm", char, "ret"]].dropna(subset=[char, "ret"])
        if len(sub) == 0:
            continue
        sub["rank"] = sub.groupby("yyyymm")[char].rank(method="average")
        sub["signal"] = sub.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())
        sub["abs_sum"] = sub.groupby("yyyymm")["signal"].transform(lambda x: x.abs().sum())
        sub["weight"] = sub["signal"] / sub["abs_sum"]
        port = (sub["weight"] * sub["ret"]).groupby(sub["yyyymm"]).sum()
        if len(port) >= 12:
            all_ports[char] = port

    ret_df = pd.DataFrame(all_ports)
    ret_df.index.name = "yyyymm"

    # Convert yyyymm index to a proper datetime 'date' column
    ret_df = ret_df.reset_index()
    ret_df["date"] = yyyymm_to_datetime(ret_df["yyyymm"])
    ret_df = ret_df.drop(columns=["yyyymm"]).set_index("date").reset_index()

    print(f"  {label}: {ret_df.shape[0]} months, {ret_df.shape[1]-1} portfolios")
    print(f"    Date range: {ret_df['date'].iloc[0].date()} to {ret_df['date'].iloc[-1].date()}")
    return ret_df


def clean_and_split(ret_df, name, split_date="2004-01-01"):
    """
    1. Drop columns with >30% NaN
    2. Drop rows with any remaining NaN
    3. Split at split_date
    Returns: R_train, R_test, dates_test, keep_cols
    """
    split_ts = pd.Timestamp(split_date)
    dates = ret_df["date"]
    data = ret_df.drop(columns=["date"])
    ret_cols = data.columns.tolist()

    # Drop columns with >30% NaN
    col_miss = data.isnull().mean()
    keep_cols = col_miss[col_miss <= 0.3].index.tolist()
    n_dropped_cols = len(ret_cols) - len(keep_cols)

    # Build clean frame with date
    clean = data[keep_cols].copy()
    clean["date"] = dates.values
    clean = clean.dropna(subset=keep_cols).reset_index(drop=True)

    print(f"  {name}: {len(ret_cols)}->{len(keep_cols)} cols (dropped {n_dropped_cols}), "
          f"{ret_df.shape[0]}->{clean.shape[0]} rows")
    print(f"    Clean date range: {clean['date'].iloc[0].date()} to "
          f"{clean['date'].iloc[-1].date()}")

    # Split
    train_mask = clean["date"] < split_ts
    test_mask  = clean["date"] >= split_ts

    R_train = clean.loc[train_mask, keep_cols].values
    R_test  = clean.loc[test_mask,  keep_cols].values
    dates_train = clean.loc[train_mask, "date"].values
    dates_test  = clean.loc[test_mask,  "date"].values

    print(f"    Train: {R_train.shape[0]} months "
          f"({str(dates_train[0])[:10]} to {str(dates_train[-1])[:10]})")
    print(f"    Test:  {R_test.shape[0]} months "
          f"({str(dates_test[0])[:10]} to {str(dates_test[-1])[:10]})")
    print(f"    Portfolios: {len(keep_cols)}")

    return R_train, R_test, dates_test, keep_cols


def run_indicator_regression(R_train, R_test, name, tscv):
    """Run raw indicator regression (Ridge + Lasso) and return results."""
    Y_train = np.ones(R_train.shape[0])
    P = R_train.shape[1]

    # Ridge
    ridge_cv = RidgeCV(alphas=np.logspace(-4, 8, 100), cv=tscv, fit_intercept=False)
    ridge_cv.fit(R_train, Y_train)
    port_ridge = R_test @ ridge_cv.coef_
    sr_ridge = annualized_sharpe(port_ridge)

    # Lasso
    lasso_cv = LassoCV(alphas=np.logspace(-5, 1, 100), cv=tscv,
                        fit_intercept=False, max_iter=50000)
    lasso_cv.fit(R_train, Y_train)
    port_lasso = R_test @ lasso_cv.coef_
    sr_lasso = annualized_sharpe(port_lasso)
    n_nz = np.sum(np.abs(lasso_cv.coef_) > 1e-10)

    # 1/N
    w_equal = np.ones(P) / P
    port_equal = R_test @ w_equal
    sr_equal = annualized_sharpe(port_equal)

    print(f"  {name} Raw: Ridge SR={sr_ridge:+.3f} (alpha={ridge_cv.alpha_:.2e}), "
          f"Lasso SR={sr_lasso:+.3f} ({n_nz}/{P} nz), "
          f"1/N SR={sr_equal:+.3f}")

    return {
        "sr_ridge": sr_ridge, "sr_lasso": sr_lasso, "sr_equal": sr_equal,
        "port_ridge": port_ridge, "port_lasso": port_lasso,
        "lasso_nz": n_nz, "n_ports": P,
        "ridge_alpha": ridge_cv.alpha_, "lasso_alpha": lasso_cv.alpha_,
    }


def run_pca_indicator(R_train, R_test, name, tscv, K_values=None):
    """Run PCA indicator regression across factor counts."""
    Y_train = np.ones(R_train.shape[0])
    P = R_train.shape[1]

    if K_values is None:
        K_values = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        K_values = [k for k in K_values if k <= P]
        if P not in K_values:
            K_values.append(P)

    # Fit PCA on train
    pca = PCA(n_components=P)
    pca.fit(R_train)
    V = pca.components_
    F_train_all = R_train @ V.T  # raw projection (no demeaning)
    F_test_all  = R_test  @ V.T
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    results = []
    for K in K_values:
        F_train = F_train_all[:, :K]
        F_test  = F_test_all[:, :K]

        # Ridge
        ridge_cv = RidgeCV(alphas=np.logspace(-4, 8, 100), cv=tscv, fit_intercept=False)
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
            "K": K, "cumvar": cumvar[K - 1],
            "sr_ridge": sr_ridge, "sr_lasso": sr_lasso, "lasso_nz": n_nz,
            "port_ridge": port_ridge, "port_lasso": port_lasso,
        })

    # Find optimal
    valid_ridge = [(i, r["sr_ridge"]) for i, r in enumerate(results)
                   if not np.isnan(r["sr_ridge"])]
    valid_lasso = [(i, r["sr_lasso"]) for i, r in enumerate(results)
                   if not np.isnan(r["sr_lasso"])]

    best_ridge_idx = max(valid_ridge, key=lambda x: x[1])[0] if valid_ridge else 0
    best_lasso_idx = max(valid_lasso, key=lambda x: x[1])[0] if valid_lasso else 0

    print(f"  {name} PCA: Best Ridge K={results[best_ridge_idx]['K']} "
          f"(SR={results[best_ridge_idx]['sr_ridge']:+.3f}), "
          f"Best Lasso K={results[best_lasso_idx]['K']} "
          f"(SR={results[best_lasso_idx]['sr_lasso']:+.3f})")

    return results, best_ridge_idx, best_lasso_idx


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL THREE DATASETS (all with proper datetime 'date' column)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

# lsret.csv - already has datetime-parseable date
print("\n1. lsret.csv (pre-computed L/S portfolios)")
lsret = pd.read_csv("lsret.csv")
lsret["date"] = pd.to_datetime(lsret["date"])
lsret = lsret.sort_values("date").reset_index(drop=True)
ret_cols_ls = [c for c in lsret.columns if c != "date"]
print(f"  lsret: {lsret.shape[0]} months, {len(ret_cols_ls)} portfolios")
print(f"    Date range: {lsret['date'].iloc[0].date()} to {lsret['date'].iloc[-1].date()}")

# Large Cap - yyyymm index converted to datetime inside build_ls_returns
print("\n2. Large Cap L/S portfolios")
lg_ret = build_ls_returns("largeml.pq", "Large Cap")

# Small Cap
print("\n3. Small Cap L/S portfolios")
sm_ret = build_ls_returns("smallml.pq", "Small Cap")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN, SPLIT, AND RUN FOR EACH DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DATA CLEANING, SPLITTING (pre-2004 / 2004+), & INDICATOR REGRESSION")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=5)
all_results = {}

for ds_name, raw_df in [("lsret", lsret), ("Large Cap", lg_ret), ("Small Cap", sm_ret)]:
    print(f"\n{'─' * 70}")
    print(f"  {ds_name}")
    print(f"{'─' * 70}")

    R_train, R_test, dates_test, keep_cols = clean_and_split(raw_df, ds_name)

    if R_train.shape[0] < 20 or R_test.shape[0] < 12:
        print(f"  SKIPPING {ds_name}: insufficient data "
              f"(train={R_train.shape[0]}, test={R_test.shape[0]})")
        continue

    # Raw indicator regression
    raw_res = run_indicator_regression(R_train, R_test, ds_name, tscv)

    # PCA indicator regression
    K_max = min(R_train.shape[0] - 1, R_train.shape[1])
    K_list = [k for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
              if k <= K_max]
    if K_max not in K_list:
        K_list.append(K_max)
    K_list = sorted(set(K_list))

    pca_res, best_ridge_i, best_lasso_i = run_pca_indicator(
        R_train, R_test, ds_name, tscv, K_values=K_list
    )

    all_results[ds_name] = {
        "raw": raw_res,
        "pca": pca_res,
        "best_ridge_i": best_ridge_i,
        "best_lasso_i": best_lasso_i,
        "dates_test": dates_test,
        "n_train": R_train.shape[0],
        "n_test": R_test.shape[0],
        "n_ports": R_train.shape[1],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: Indicator Regression OOS Sharpe Ratios")
print("=" * 70)

print(f"\n{'Universe':>12s}  {'Ports':>5s}  {'Train':>5s}  {'Test':>5s}  "
      f"{'Ridge Raw':>10s}  {'Lasso Raw':>10s}  {'1/N':>6s}  "
      f"{'Ridge PCA':>10s}  {'K_r':>4s}  {'Lasso PCA':>10s}  {'K_l':>4s}")
print("-" * 105)

for ds_name, res in all_results.items():
    raw = res["raw"]
    pca = res["pca"]
    br_i = res["best_ridge_i"]
    bl_i = res["best_lasso_i"]

    print(f"{ds_name:>12s}  {res['n_ports']:5d}  {res['n_train']:5d}  {res['n_test']:5d}  "
          f"{raw['sr_ridge']:+10.3f}  {raw['sr_lasso']:+10.3f}  {raw['sr_equal']:+6.2f}  "
          f"{pca[br_i]['sr_ridge']:+10.3f}  {pca[br_i]['K']:4d}  "
          f"{pca[bl_i]['sr_lasso']:+10.3f}  {pca[bl_i]['K']:4d}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
colors = {"lsret": "#3498db", "Large Cap": "#e67e22", "Small Cap": "#e74c3c"}
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Plot 1: SR vs K (Ridge) ---
ax = axes[0, 0]
for ds_name, res in all_results.items():
    pca = res["pca"]
    Ks = [r["K"] for r in pca]
    srs = [r["sr_ridge"] for r in pca]
    ax.plot(Ks, srs, "o-", color=colors[ds_name], linewidth=2, markersize=5, label=ds_name)
    # Mark raw baseline
    ax.axhline(y=res["raw"]["sr_ridge"], color=colors[ds_name],
               linestyle=":", alpha=0.5)
    # Mark optimal
    bi = res["best_ridge_i"]
    ax.scatter([pca[bi]["K"]], [pca[bi]["sr_ridge"]],
               color=colors[ds_name], s=120, zorder=5, edgecolors="black", linewidth=1.5)

ax.set_xlabel("Number of PCA Factors (K)", fontsize=11)
ax.set_ylabel("OOS Annualized Sharpe Ratio", fontsize=11)
ax.set_title("Ridge: Sharpe Ratio vs. PCA Factors", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Plot 2: SR vs K (Lasso) ---
ax = axes[0, 1]
for ds_name, res in all_results.items():
    pca = res["pca"]
    Ks = [r["K"] for r in pca]
    srs = [r["sr_lasso"] for r in pca]
    ax.plot(Ks, srs, "s-", color=colors[ds_name], linewidth=2, markersize=5, label=ds_name)
    # Mark raw baseline
    ax.axhline(y=res["raw"]["sr_lasso"], color=colors[ds_name],
               linestyle=":", alpha=0.5)
    # Mark optimal
    bi = res["best_lasso_i"]
    ax.scatter([pca[bi]["K"]], [pca[bi]["sr_lasso"]],
               color=colors[ds_name], s=120, zorder=5, edgecolors="black", linewidth=1.5)

ax.set_xlabel("Number of PCA Factors (K)", fontsize=11)
ax.set_ylabel("OOS Annualized Sharpe Ratio", fontsize=11)
ax.set_title("Lasso: Sharpe Ratio vs. PCA Factors", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Plot 3: Bar chart comparing all methods ---
ax = axes[1, 0]
ds_names = list(all_results.keys())
x = np.arange(len(ds_names))
w = 0.15

methods = []
for ds_name, res in all_results.items():
    bi_r = res["best_ridge_i"]
    bi_l = res["best_lasso_i"]
    methods.append({
        "Ridge Raw": res["raw"]["sr_ridge"],
        "Lasso Raw": res["raw"]["sr_lasso"],
        "Ridge PCA*": res["pca"][bi_r]["sr_ridge"],
        "Lasso PCA*": res["pca"][bi_l]["sr_lasso"],
        "1/N": res["raw"]["sr_equal"],
    })

bar_colors = ["#3498db", "#e74c3c", "#2980b9", "#c0392b", "#95a5a6"]
method_names = list(methods[0].keys())
for i, mname in enumerate(method_names):
    vals = [m[mname] for m in methods]
    ax.bar(x + i * w, vals, w, label=mname, color=bar_colors[i], alpha=0.85)

ax.set_xticks(x + 2 * w)
ax.set_xticklabels(ds_names, fontsize=11)
ax.set_ylabel("OOS Annualized Sharpe Ratio", fontsize=11)
ax.set_title("Comparison: All Methods x All Universes", fontsize=13)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0, color="black", linewidth=0.5)

# --- Plot 4: Cumulative returns (best PCA Lasso for each universe) ---
ax = axes[1, 1]
for ds_name, res in all_results.items():
    bi = res["best_lasso_i"]
    port = res["pca"][bi]["port_lasso"]
    K = res["pca"][bi]["K"]
    sr = res["pca"][bi]["sr_lasso"]
    cum = (1 + port).cumprod()
    dt = pd.to_datetime(res["dates_test"])  # already proper datetimes
    ax.plot(dt, cum, label=f"{ds_name} K={K} (SR={sr:+.2f})",
            color=colors[ds_name], linewidth=2)

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Cumulative Return ($1 invested)", fontsize=11)
ax.set_title("Best PCA Lasso: Cumulative OOS Returns (2004+)", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task2d_compare.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved task2d_compare.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DETAILED PCA RESULTS PER UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════
for ds_name, res in all_results.items():
    print(f"\n{'=' * 70}")
    print(f"PCA Detail: {ds_name}")
    print(f"{'=' * 70}")
    print(f"{'K':>4s}  {'CumVar':>7s}  {'Ridge SR':>10s}  {'Lasso SR':>10s}  {'Lasso nz':>9s}")
    print("-" * 45)
    for r in res["pca"]:
        sr_l = f"{r['sr_lasso']:+.3f}" if not np.isnan(r['sr_lasso']) else "NaN"
        print(f"{r['K']:4d}  {r['cumvar']:7.1%}  {r['sr_ridge']:+10.3f}  "
              f"{sr_l:>10s}  {r['lasso_nz']:5d}/{r['K']}")

print()
