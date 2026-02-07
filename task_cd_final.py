"""
Tasks (c) & (d): Small-Cap ML Portfolios + Large vs. Small Comparison
======================================================================
1. Retrain all small-cap models, construct rank-weighted L/S portfolios
2. Re-compute univariate Sharpe ratios on the TEST period for both universes
3. Consolidated comparison plots and tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.kernel_approximation import RBFSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def annualized_sharpe(mr):
    m, s = mr.mean(), mr.std(ddof=1)
    return (m / s) * np.sqrt(12) if s > 0 and not np.isnan(s) else np.nan


def portfolio_from_signal(frame, signal_col):
    tmp = frame[["yyyymm", "ret", signal_col]].dropna()
    tmp["rank"] = tmp.groupby("yyyymm")[signal_col].rank(method="average")
    tmp["signal"] = tmp.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())
    tmp["abs_sum"] = tmp.groupby("yyyymm")["signal"].transform(lambda x: x.abs().sum())
    tmp["weight"] = tmp["signal"] / tmp["abs_sum"]
    return (tmp["weight"] * tmp["ret"]).groupby(tmp["yyyymm"]).sum()


def univariate_sharpes_test(df_test, feature_cols):
    """Compute rank-weighted L/S Sharpe for each characteristic on test data."""
    sharpes = {}
    for char in feature_cols:
        sub = df_test[["permno", "yyyymm", char, "ret"]].dropna(subset=[char, "ret"])
        if len(sub) == 0:
            continue
        sub["rank"] = sub.groupby("yyyymm")[char].rank(method="average")
        sub["signal"] = sub.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())
        sub["abs_sum"] = sub.groupby("yyyymm")["signal"].transform(lambda x: x.abs().sum())
        sub["weight"] = sub["signal"] / sub["abs_sum"]
        port = (sub["weight"] * sub["ret"]).groupby(sub["yyyymm"]).sum()
        if len(port) >= 12:
            sharpes[char] = annualized_sharpe(port)
    return sharpes


def prepare_and_model(filepath, label):
    """Full pipeline: load, prep, split, train all models, build portfolios."""
    print(f"\n{'#' * 70}")
    print(f"  PROCESSING: {label} ({filepath})")
    print(f"{'#' * 70}")

    # Load & prep
    df = pd.read_parquet(filepath)
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    feature_cols = [c for c in df.columns if c not in ["permno", "yyyymm", "ret"]]
    df = df.dropna(subset=["ret"]).reset_index(drop=True)

    # Split
    train_mask = df["yyyymm"] <= 194511
    val_mask   = (df["yyyymm"] > 194511) & (df["yyyymm"] <= 195711)
    test_mask  = df["yyyymm"] > 195711

    # Save RAW test data (before imputation) for univariate analysis
    test_raw = df.loc[test_mask].copy()

    # Impute (for ML models only)
    df[feature_cols] = df.groupby("yyyymm")[feature_cols].transform(
        lambda x: x.fillna(x.median()))
    df[feature_cols] = df[feature_cols].fillna(0)

    X_tr = df.loc[train_mask, feature_cols].values
    y_tr = df.loc[train_mask, "ret"].values
    X_va = df.loc[val_mask, feature_cols].values
    y_va = df.loc[val_mask, "ret"].values
    X_te = df.loc[test_mask, feature_cols].values
    y_te = df.loc[test_mask, "ret"].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    test_df = df.loc[test_mask, ["permno", "yyyymm", "ret"]].copy().reset_index(drop=True)
    test_full = test_raw  # Use raw data for univariate Sharpe calculation

    print(f"  Train: {X_tr_s.shape}  Val: {X_va_s.shape}  Test: {X_te_s.shape}")

    # ── Train all models & collect test predictions ──────────────────────────
    preds = {}

    # (i) OLS
    preds["OLS"] = LinearRegression().fit(X_tr_s, y_tr).predict(X_te_s)

    # (ii) Ridge - tune
    def r2z(yt, yp):
        return 1 - np.sum((yt - yp)**2) / np.sum(yt**2)

    best_r2, best_a = -np.inf, None
    for a in np.logspace(0, 6, 40):
        m = Ridge(alpha=a, max_iter=50000).fit(X_tr_s, y_tr)
        r2 = r2z(y_va, m.predict(X_va_s))
        if r2 > best_r2:
            best_r2, best_a = r2, a
    preds["Ridge"] = Ridge(alpha=best_a, max_iter=50000).fit(X_tr_s, y_tr).predict(X_te_s)
    print(f"  Ridge alpha={best_a:.2e}")

    # (ii) Lasso - tune
    best_r2, best_a = -np.inf, None
    for a in np.logspace(-5, 0, 30):
        m = Lasso(alpha=a, max_iter=50000).fit(X_tr_s, y_tr)
        r2 = r2z(y_va, m.predict(X_va_s))
        if r2 > best_r2:
            best_r2, best_a = r2, a
    preds["Lasso"] = Lasso(alpha=best_a, max_iter=50000).fit(X_tr_s, y_tr).predict(X_te_s)
    print(f"  Lasso alpha={best_a:.2e}")

    # (ii) ElasticNet - tune
    best_r2, best_a, best_l1 = -np.inf, None, None
    for l1 in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        for a in np.logspace(-5, 0, 30):
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=50000).fit(X_tr_s, y_tr)
            r2 = r2z(y_va, m.predict(X_va_s))
            if r2 > best_r2:
                best_r2, best_a, best_l1 = r2, a, l1
    preds["ElasticNet"] = ElasticNet(alpha=best_a, l1_ratio=best_l1,
                                     max_iter=50000).fit(X_tr_s, y_tr).predict(X_te_s)
    print(f"  ElasticNet alpha={best_a:.2e} l1={best_l1}")

    # (iii) RBF + Ridge/Lasso/ElasticNet
    from itertools import product as iprod
    for mname, MCls, agrid, extra in [
        ("RBF+Ridge", Ridge, np.logspace(0, 5, 20), {}),
        ("RBF+Lasso", Lasso, np.logspace(-5, 0, 20), {}),
        ("RBF+ElasticNet", ElasticNet, np.logspace(-5, 0, 20), {"l1_ratio": 0.5}),
    ]:
        best_r2, best_mod, best_rbf_obj = -np.inf, None, None
        for g, nc in iprod([0.001, 0.01, 0.1], [100, 300, 500]):
            rbf = RBFSampler(gamma=g, n_components=nc, random_state=42)
            Xr = rbf.fit_transform(X_tr_s)
            Xv = rbf.transform(X_va_s)
            for a in agrid:
                params = {"alpha": a, "max_iter": 50000, **extra}
                mod = MCls(**params).fit(Xr, y_tr)
                r2 = r2z(y_va, mod.predict(Xv))
                if r2 > best_r2:
                    best_r2 = r2
                    best_mod = mod
                    best_rbf_obj = rbf
        preds[mname] = best_mod.predict(best_rbf_obj.transform(X_te_s))
    print(f"  RBF variants done")

    # (iv) PLS
    best_r2, best_pls = -np.inf, None
    for nc in range(1, min(50, X_tr_s.shape[1]) + 1):
        pls = PLSRegression(n_components=nc).fit(X_tr_s, y_tr)
        r2 = r2z(y_va, pls.predict(X_va_s).ravel())
        if r2 > best_r2:
            best_r2 = r2
            best_pls = pls
    preds["PLS"] = best_pls.predict(X_te_s).ravel()
    print(f"  PLS done")

    # (v) GBR
    best_r2, best_gbr = -np.inf, None
    for combo in iprod([100, 300, 500], [2, 3, 4], [0.01, 0.05, 0.1], [0.8, 1.0]):
        ne, md, lr, ss = combo
        gbr = GradientBoostingRegressor(n_estimators=ne, max_depth=md, learning_rate=lr,
                                        subsample=ss, min_samples_leaf=10, random_state=42)
        gbr.fit(X_tr_s, y_tr)
        r2 = r2z(y_va, gbr.predict(X_va_s))
        if r2 > best_r2:
            best_r2 = r2
            best_gbr = gbr
    preds["GradientBoosting"] = best_gbr.predict(X_te_s)
    print(f"  GBR done")

    # ── Build portfolios from ML predictions ─────────────────────────────────
    ml_sharpes = {}
    ml_port_rets = {}
    for mname, pred_arr in preds.items():
        test_df[f"pred_{mname}"] = pred_arr
        port = portfolio_from_signal(test_df, f"pred_{mname}")
        sr = annualized_sharpe(port)
        ml_sharpes[mname] = sr
        ml_port_rets[mname] = port

    # ── Univariate Sharpe ratios on TEST period ──────────────────────────────
    uni_sharpes = univariate_sharpes_test(test_full, feature_cols)

    return ml_sharpes, uni_sharpes, ml_port_rets


# ══════════════════════════════════════════════════════════════════════════════
# RUN BOTH UNIVERSES
# ══════════════════════════════════════════════════════════════════════════════
lg_ml, lg_uni, lg_ports = prepare_and_model("largeml.pq", "LARGE CAPS")
sm_ml, sm_uni, sm_ports = prepare_and_model("smallml.pq", "SMALL CAPS")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT PORTFOLIO SHARPE RATIOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ML PORTFOLIO SHARPE RATIOS (Test Period)")
print("=" * 70)
print(f"{'Method':20s}  {'Large Cap':>12s}  {'Small Cap':>12s}")
print("-" * 50)
for m in ["OLS", "Ridge", "Lasso", "ElasticNet",
          "RBF+Ridge", "RBF+Lasso", "RBF+ElasticNet",
          "PLS", "GradientBoosting"]:
    lg_sr = lg_ml.get(m, np.nan)
    sm_sr = sm_ml.get(m, np.nan)
    print(f"{m:20s}  {lg_sr:+12.3f}  {sm_sr:+12.3f}")


print("\n" + "=" * 70)
print("TOP 5 UNIVARIATE SHARPE RATIOS (Test Period)")
print("=" * 70)
# Filter out NaN Sharpes before sorting
lg_top = sorted([(k,v) for k,v in lg_uni.items() if not np.isnan(v)],
                key=lambda x: x[1], reverse=True)[:5]
sm_top = sorted([(k,v) for k,v in sm_uni.items() if not np.isnan(v)],
                key=lambda x: x[1], reverse=True)[:5]

print(f"\n  Large Caps:")
for i, (c, s) in enumerate(lg_top, 1):
    print(f"    {i}. {c:30s}  {s:+.3f}")
print(f"\n  Small Caps:")
for i, (c, s) in enumerate(sm_top, 1):
    print(f"    {i}. {c:30s}  {s:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Side-by-side bar chart — ML models
# ══════════════════════════════════════════════════════════════════════════════
model_order = ["OLS", "Ridge", "Lasso", "ElasticNet",
               "RBF+Ridge", "RBF+Lasso", "RBF+ElasticNet",
               "PLS", "GradientBoosting"]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(model_order))
w = 0.35

lg_vals = [lg_ml[m] for m in model_order]
sm_vals = [sm_ml[m] for m in model_order]

bars1 = ax.bar(x - w/2, lg_vals, w, label="Large Cap", color="#3498db", edgecolor="white")
bars2 = ax.bar(x + w/2, sm_vals, w, label="Small Cap", color="#e74c3c", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
ax.set_title("ML Portfolio Sharpe Ratios: Large Cap vs. Small Cap (Test Period)", fontsize=13)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Value labels
for bar_set in [bars1, bars2]:
    for bar in bar_set:
        h = bar.get_height()
        if abs(h) > 0.3:
            ax.text(bar.get_x() + bar.get_width()/2, h + (0.08 if h > 0 else -0.25),
                    f"{h:+.1f}", ha="center", va="bottom" if h > 0 else "top",
                    fontsize=7, fontweight="bold")

plt.tight_layout()
plt.savefig("task_d_ml_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved task_d_ml_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Consolidated — Best Univariate vs. Best Linear ML vs. Best NL ML
# ══════════════════════════════════════════════════════════════════════════════
categories = [
    ("Best Univariate", lg_top[0][1], sm_top[0][1],
     f"LG: {lg_top[0][0]}", f"SM: {sm_top[0][0]}"),
    ("Best Linear ML",
     max(lg_ml[m] for m in ["Lasso", "ElasticNet", "Ridge"]),
     max(sm_ml[m] for m in ["Lasso", "ElasticNet", "Ridge"]),
     f"LG: {max(['Lasso','ElasticNet','Ridge'], key=lambda m: lg_ml[m])}",
     f"SM: {max(['Lasso','ElasticNet','Ridge'], key=lambda m: sm_ml[m])}"),
    ("Best Non-linear ML",
     max(lg_ml[m] for m in ["GradientBoosting", "PLS"]),
     max(sm_ml[m] for m in ["GradientBoosting", "PLS"]),
     f"LG: GradientBoosting", f"SM: {max(['GradientBoosting','PLS'], key=lambda m: sm_ml[m])}"),
]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(categories))
w = 0.3

lg_v = [c[1] for c in categories]
sm_v = [c[2] for c in categories]
labels = [c[0] for c in categories]

bars1 = ax.bar(x - w/2, lg_v, w, label="Large Cap", color="#3498db", edgecolor="white")
bars2 = ax.bar(x + w/2, sm_v, w, label="Small Cap", color="#e74c3c", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
ax.set_title("Best-in-Class Comparison: Large Cap vs. Small Cap (Test Period)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Annotations
for i, cat in enumerate(categories):
    ax.text(i - w/2, cat[1] + 0.15, cat[3], ha="center", fontsize=7, color="#2c3e50")
    ax.text(i + w/2, cat[2] + 0.15, cat[4], ha="center", fontsize=7, color="#2c3e50")

for bar_set in [bars1, bars2]:
    for bar in bar_set:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("task_d_bestclass_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved task_d_bestclass_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Cumulative returns — best performers from each universe
# ══════════════════════════════════════════════════════════════════════════════
best_lg_ml_name = max(lg_ml, key=lg_ml.get)
best_sm_ml_name = max(sm_ml, key=sm_ml.get)

lg_cum = (1 + lg_ports[best_lg_ml_name].sort_index()).cumprod()
sm_cum = (1 + sm_ports[best_sm_ml_name].sort_index()).cumprod()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(lg_cum.index, lg_cum.values,
        label=f"Large Cap - {best_lg_ml_name} (SR={lg_ml[best_lg_ml_name]:+.2f})",
        linewidth=1.5, color="#3498db")
ax.plot(sm_cum.index, sm_cum.values,
        label=f"Small Cap - {best_sm_ml_name} (SR={sm_ml[best_sm_ml_name]:+.2f})",
        linewidth=1.5, color="#e74c3c")

ax.set_xlabel("Date (yyyymm)", fontsize=11)
ax.set_ylabel("Cumulative Return ($1 invested)", fontsize=11)
ax.set_title("Cumulative Performance: Best ML Portfolio in Each Universe (Test Period)", fontsize=13)
ax.legend(fontsize=10)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

tick_vals = lg_cum.index[::60]
ax.set_xticks(tick_vals)
ax.set_xticklabels([str(d) for d in tick_vals], rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig("task_d_cumulative_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved task_d_cumulative_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL CONSOLIDATED TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL CONSOLIDATED TABLE")
print("=" * 70)

rows = []
# Univariate top 3
for i, ((lc, ls), (sc, ss)) in enumerate(zip(lg_top[:3], sm_top[:3])):
    rows.append({"Category": "Univariate", "Rank": i+1,
                 "Large Cap Method": lc, "Large Cap SR": ls,
                 "Small Cap Method": sc, "Small Cap SR": ss})

# ML models
for m in model_order:
    cat = "Linear ML" if m in ["OLS","Ridge","Lasso","ElasticNet"] else "Non-linear ML"
    rows.append({"Category": cat, "Rank": "",
                 "Large Cap Method": m, "Large Cap SR": lg_ml[m],
                 "Small Cap Method": m, "Small Cap SR": sm_ml[m]})

final = pd.DataFrame(rows)
final["Large Cap SR"] = final["Large Cap SR"].map(lambda x: f"{x:+.3f}")
final["Small Cap SR"] = final["Small Cap SR"].map(lambda x: f"{x:+.3f}")
print(final.to_string(index=False))
print()
