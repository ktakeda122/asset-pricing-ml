"""
Task 2(a): PCA Analysis on Three Sets of Long-Short Portfolios
===============================================================
1. lsret.csv  - Pre-computed L/S portfolio returns
2. Large Cap  - Rank-weighted L/S from largeml.pq
3. Small Cap  - Rank-weighted L/S from smallml.pq

For each set: PCA, cumulative explained variance, factor Sharpe ratios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & BUILD ALL THREE RETURN MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def annualized_sharpe(mr):
    m, s = mr.mean(), mr.std(ddof=1)
    return (m / s) * np.sqrt(12) if s > 0 and not np.isnan(s) else np.nan


# ── 1a. lsret.csv ───────────────────────────────────────────────────────────
print("Loading lsret.csv...")
lsret = pd.read_csv("lsret.csv")
lsret["date"] = pd.to_datetime(lsret["date"])
# Convert date to yyyymm for alignment
lsret["yyyymm"] = lsret["date"].dt.year * 100 + lsret["date"].dt.month
lsret_cols = [c for c in lsret.columns if c not in ["date", "yyyymm"]]
print(f"  lsret: {lsret.shape[0]} months, {len(lsret_cols)} portfolios")


# ── 1b. Large Cap rank-weighted L/S portfolios ──────────────────────────────
def build_ls_returns(filepath, label):
    """Build rank-weighted L/S portfolio returns for each characteristic."""
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
    print(f"  {label}: {ret_df.shape[0]} months, {ret_df.shape[1]} portfolios")
    return ret_df


print("Building Large Cap L/S portfolios...")
lg_ret = build_ls_returns("largeml.pq", "Large Cap")

print("Building Small Cap L/S portfolios...")
sm_ret = build_ls_returns("smallml.pq", "Small Cap")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPARE RETURN MATRICES FOR PCA
# ══════════════════════════════════════════════════════════════════════════════

def prepare_for_pca(ret_df, name, yyyymm_col=None):
    """
    Prepare a return matrix for PCA:
    - Drop columns with >30% missing
    - Drop rows with any remaining NaN
    - Standardize
    Returns: (standardized matrix, column names, yyyymm index, scaler, raw matrix)
    """
    if yyyymm_col is not None:
        # For lsret which has yyyymm as a column
        idx = ret_df[yyyymm_col].values
        data = ret_df.drop(columns=["date", yyyymm_col], errors="ignore")
    else:
        idx = ret_df.index.values
        data = ret_df.copy()

    # Drop columns with too many NaN
    thresh = 0.3
    col_miss = data.isnull().mean()
    keep_cols = col_miss[col_miss <= thresh].index.tolist()
    data = data[keep_cols]

    # Drop rows with any remaining NaN
    valid_mask = data.notna().all(axis=1)
    data = data[valid_mask]
    idx = idx[valid_mask] if isinstance(idx, np.ndarray) else data.index.values

    cols = data.columns.tolist()
    raw = data.values

    # Standardize (demean + unit variance) for PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(raw)

    print(f"  {name}: {X.shape[0]} months x {X.shape[1]} portfolios (after cleaning)")
    return X, cols, idx, scaler, raw


print("\nPreparing data for PCA...")
X_ls, cols_ls, idx_ls, _, raw_ls = prepare_for_pca(lsret, "lsret", yyyymm_col="yyyymm")
X_lg, cols_lg, idx_lg, _, raw_lg = prepare_for_pca(lg_ret, "Large Cap")
X_sm, cols_sm, idx_sm, _, raw_sm = prepare_for_pca(sm_ret, "Small Cap")


# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN PCA ON EACH SET
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning PCA...")

datasets = [
    ("lsret (Pre-computed)", X_ls, cols_ls, idx_ls, raw_ls),
    ("Large Cap (Derived)",  X_lg, cols_lg, idx_lg, raw_lg),
    ("Small Cap (Derived)",  X_sm, cols_sm, idx_sm, raw_sm),
]

pca_results = {}

for name, X, cols, idx, raw in datasets:
    n_comp = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    factors = pca.fit_transform(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_80 = np.searchsorted(cumvar, 0.80) + 1
    n_90 = np.searchsorted(cumvar, 0.90) + 1
    n_95 = np.searchsorted(cumvar, 0.95) + 1

    # Sharpe ratios of the first 10 PC factors
    # PC factors from standardized data need to be projected back to return space
    # Factor return = portfolio of original returns weighted by PC loading
    # Simpler: use the raw (non-standardized) returns projected onto PC directions
    loadings = pca.components_  # (n_components, n_features)
    # Factor returns from raw data: raw @ loadings.T
    factor_rets_raw = raw @ loadings.T  # (T, n_components)

    sharpes = []
    for i in range(min(10, n_comp)):
        sr = annualized_sharpe(factor_rets_raw[:, i])
        sharpes.append(sr)

    pca_results[name] = {
        "pca": pca,
        "factors": factors,
        "cumvar": cumvar,
        "n_80": n_80,
        "n_90": n_90,
        "n_95": n_95,
        "sharpes": sharpes,
        "n_total": n_comp,
        "factor_rets": factor_rets_raw,
        "idx": idx,
    }

    print(f"\n  {name}:")
    print(f"    Total components: {n_comp}")
    print(f"    PC1 explains: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"    Components for 80%: {n_80}")
    print(f"    Components for 90%: {n_90}")
    print(f"    Components for 95%: {n_95}")
    print(f"    Top 5 PC Sharpe ratios: {['%.2f' % s for s in sharpes[:5]]}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION: Cumulative Explained Variance
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = {"lsret (Pre-computed)": "#3498db",
          "Large Cap (Derived)": "#e67e22",
          "Small Cap (Derived)": "#e74c3c"}

# Plot 1: Full cumulative variance
ax = axes[0]
for name, res in pca_results.items():
    n_plot = min(50, len(res["cumvar"]))
    ax.plot(range(1, n_plot + 1), res["cumvar"][:n_plot],
            label=f"{name} (90%: {res['n_90']} PCs)", color=colors[name], linewidth=2)
ax.axhline(y=0.80, color="gray", linestyle="--", alpha=0.7, label="80%")
ax.axhline(y=0.90, color="gray", linestyle=":", alpha=0.7, label="90%")
ax.set_xlabel("Number of Principal Components", fontsize=11)
ax.set_ylabel("Cumulative Explained Variance", fontsize=11)
ax.set_title("Cumulative Explained Variance Ratio", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)
ax.set_ylim(0, 1.02)

# Plot 2: Individual explained variance (scree plot) for first 20
ax = axes[1]
for name, res in pca_results.items():
    n_plot = min(20, len(res["pca"].explained_variance_ratio_))
    ax.bar(np.arange(1, n_plot + 1) + (list(colors.keys()).index(name) - 1) * 0.25,
           res["pca"].explained_variance_ratio_[:n_plot] * 100,
           width=0.25, label=name, color=colors[name], alpha=0.8)
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("Variance Explained (%)", fontsize=11)
ax.set_title("Scree Plot (First 20 PCs)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_xlim(0, 21)

plt.tight_layout()
plt.savefig("task2a_explained_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved task2a_explained_variance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FACTOR SHARPE RATIOS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

n_factors = 10
x = np.arange(1, n_factors + 1)
w = 0.25

for i, (name, res) in enumerate(pca_results.items()):
    srs = res["sharpes"][:n_factors]
    bars = ax.bar(x + (i - 1) * w, srs, w, label=name, color=colors[name], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([f"PC{i}" for i in x])
ax.set_xlabel("Principal Component Factor", fontsize=11)
ax.set_ylabel("Annualized Sharpe Ratio", fontsize=11)
ax.set_title("Sharpe Ratios of PC Factors Across Universes", fontsize=13)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("task2a_factor_sharpes.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved task2a_factor_sharpes.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: PCA ANALYSIS")
print("=" * 70)

print(f"\n{'Dataset':30s}  {'Months':>6s}  {'Ports':>5s}  {'80%':>4s}  {'90%':>4s}  {'95%':>4s}  {'PC1%':>6s}")
print("-" * 70)
for name, res in pca_results.items():
    pc1 = res["pca"].explained_variance_ratio_[0] * 100
    n_ports = res["pca"].n_features_in_
    n_months = res["factors"].shape[0]
    print(f"{name:30s}  {n_months:>6d}  {n_ports:>5d}  {res['n_80']:>4d}  "
          f"{res['n_90']:>4d}  {res['n_95']:>4d}  {pc1:>5.1f}%")

print(f"\n{'Dataset':30s}  {'PC1':>7s}  {'PC2':>7s}  {'PC3':>7s}  {'PC4':>7s}  {'PC5':>7s}")
print("-" * 70)
for name, res in pca_results.items():
    srs = res["sharpes"]
    print(f"{name:30s}  " + "  ".join(f"{s:+7.2f}" for s in srs[:5]))

# Top-loading characteristics for PC1
print("\n" + "=" * 70)
print("TOP LOADINGS ON PC1 (absolute value)")
print("=" * 70)
for dname, X, cols, idx, raw in datasets:
    res = pca_results[dname]
    pc1_loadings = res["pca"].components_[0]
    top_idx = np.argsort(np.abs(pc1_loadings))[::-1][:10]
    print(f"\n  {dname}:")
    for rank, j in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {cols[j]:30s}  loading={pc1_loadings[j]:+.4f}")
print()
