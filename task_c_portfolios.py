"""
Task (c): Portfolio Construction from ML Forecasts
====================================================
- Retrain every model from task (b)
- Generate OOS predictions on the Test set
- Form rank-weighted long-short portfolios each month
- Compute Annualized Sharpe Ratios
- Compare with best univariate characteristics (re-calculated on same Test period)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.kernel_approximation import RBFSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_parquet("largeml_prepared.pq")
splits = pd.read_parquet("splits.pq")

feature_cols = [c for c in df.columns if c not in ["permno", "yyyymm", "ret"]]

train_mask = splits["split"] == "train"
val_mask   = splits["split"] == "val"
test_mask  = splits["split"] == "test"

X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, "ret"].values
X_test  = df.loc[test_mask,  feature_cols].values
y_test  = df.loc[test_mask,  "ret"].values

test_df = df.loc[test_mask, ["permno", "yyyymm", "ret"]].copy().reset_index(drop=True)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train_s.shape}  Test: {X_test_s.shape}")
print(f"Test period: {test_df.yyyymm.min()} - {test_df.yyyymm.max()}")
print()


# ── 2. Helper functions ──────────────────────────────────────────────────────
def annualized_sharpe(monthly_returns):
    """Annualized Sharpe = (mean / std) * sqrt(12)."""
    m, s = monthly_returns.mean(), monthly_returns.std(ddof=1)
    if s == 0 or np.isnan(s):
        return np.nan
    return (m / s) * np.sqrt(12)


def portfolio_returns_from_signal(test_frame, signal_col):
    """
    Rank-weighted long-short portfolio from a cross-sectional signal.
    Signal = rank - mean(rank), normalized to unit leverage.
    Returns a Series of monthly portfolio returns indexed by yyyymm.
    """
    tmp = test_frame[["yyyymm", "ret", signal_col]].dropna()
    tmp["rank"] = tmp.groupby("yyyymm")[signal_col].rank(method="average")
    tmp["signal"] = tmp.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())
    tmp["abs_sum"] = tmp.groupby("yyyymm")["signal"].transform(lambda x: x.abs().sum())
    tmp["weight"] = tmp["signal"] / tmp["abs_sum"]
    port = (tmp["weight"] * tmp["ret"]).groupby(tmp["yyyymm"]).sum()
    return port


# ── 3. Retrain all ML models & collect test predictions ─────────────────────
print("Retraining all ML models and generating test predictions...")

models_config = {}

# (i) OLS
ols = LinearRegression().fit(X_train_s, y_train)
models_config["OLS"] = ols.predict(X_test_s)

# (ii) Ridge
ridge = Ridge(alpha=2.652948e+03, max_iter=50000).fit(X_train_s, y_train)
models_config["Ridge"] = ridge.predict(X_test_s)

# (ii) Lasso
lasso = Lasso(alpha=8.286428e-03, max_iter=50000).fit(X_train_s, y_train)
models_config["Lasso"] = lasso.predict(X_test_s)

# (ii) ElasticNet
enet = ElasticNet(alpha=8.286428e-03, l1_ratio=0.95, max_iter=50000).fit(X_train_s, y_train)
models_config["ElasticNet"] = enet.predict(X_test_s)

# (iii) RBF + Ridge
rbf_ridge = RBFSampler(gamma=0.001, n_components=500, random_state=42)
X_tr_rbf = rbf_ridge.fit_transform(X_train_s)
X_te_rbf = rbf_ridge.transform(X_test_s)
rbf_ridge_m = Ridge(alpha=1.4874, max_iter=50000).fit(X_tr_rbf, y_train)
models_config["RBF+Ridge"] = rbf_ridge_m.predict(X_te_rbf)

# (iii) RBF + Lasso
rbf_lasso = RBFSampler(gamma=0.001, n_components=500, random_state=42)
X_tr_rbf2 = rbf_lasso.fit_transform(X_train_s)
X_te_rbf2 = rbf_lasso.transform(X_test_s)
rbf_lasso_m = Lasso(alpha=2.3950e-04, max_iter=50000).fit(X_tr_rbf2, y_train)
models_config["RBF+Lasso"] = rbf_lasso_m.predict(X_te_rbf2)

# (iii) RBF + ElasticNet
rbf_enet = RBFSampler(gamma=0.001, n_components=500, random_state=42)
X_tr_rbf3 = rbf_enet.fit_transform(X_train_s)
X_te_rbf3 = rbf_enet.transform(X_test_s)
rbf_enet_m = ElasticNet(alpha=3.5622e-04, l1_ratio=0.5, max_iter=50000).fit(X_tr_rbf3, y_train)
models_config["RBF+ElasticNet"] = rbf_enet_m.predict(X_te_rbf3)

# (iv) PLS
pls = PLSRegression(n_components=1).fit(X_train_s, y_train)
models_config["PLS"] = pls.predict(X_test_s).ravel()

# (v) Gradient Boosting
gbr = GradientBoostingRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.01,
    subsample=0.8, min_samples_leaf=10, random_state=42
).fit(X_train_s, y_train)
models_config["GradientBoosting"] = gbr.predict(X_test_s)

print(f"  Generated predictions for {len(models_config)} models.")
print()


# ── 4. Construct ML portfolios ──────────────────────────────────────────────
print("Constructing rank-weighted long-short portfolios from ML predictions...")

ml_sharpe = {}
ml_port_returns = {}

for model_name, preds in models_config.items():
    test_df[f"pred_{model_name}"] = preds
    port_ret = portfolio_returns_from_signal(test_df, f"pred_{model_name}")
    sr = annualized_sharpe(port_ret)
    ml_sharpe[model_name] = sr
    ml_port_returns[model_name] = port_ret
    ann_mean = port_ret.mean() * 12
    ann_std = port_ret.std() * np.sqrt(12)
    print(f"  {model_name:20s}  Sharpe={sr:+.3f}  "
          f"Ann.Mean={ann_mean:+.4f}  Ann.Vol={ann_std:.4f}  Months={len(port_ret)}")

print()


# ── 5. Re-compute univariate portfolio Sharpe ratios on TEST period only ─────
print("=" * 70)
print("Re-computing univariate portfolio Sharpe ratios (TEST period only)")
print("=" * 70)

# Load raw prepared data for the test period
test_full = df.loc[test_mask].copy()

uni_sharpe = {}
for char in feature_cols:
    sub = test_full[["permno", "yyyymm", char, "ret"]].dropna(subset=[char, "ret"])
    if len(sub) == 0:
        continue
    sub["rank"] = sub.groupby("yyyymm")[char].rank(method="average")
    sub["signal"] = sub.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())
    sub["abs_sum"] = sub.groupby("yyyymm")["signal"].transform(lambda x: x.abs().sum())
    sub["weight"] = sub["signal"] / sub["abs_sum"]
    port_ret = (sub["weight"] * sub["ret"]).groupby(sub["yyyymm"]).sum()
    if len(port_ret) >= 12:
        uni_sharpe[char] = annualized_sharpe(port_ret)

uni_sorted = sorted(uni_sharpe.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 10 univariate characteristics (Test period):")
for rank, (char, sr) in enumerate(uni_sorted[:10], 1):
    print(f"  {rank:2d}. {char:30s}  Sharpe={sr:+.3f}")

print(f"\nBottom 5 univariate characteristics (Test period):")
for rank, (char, sr) in enumerate(uni_sorted[-5:], 1):
    print(f"  {rank:2d}. {char:30s}  Sharpe={sr:+.3f}")

# Pick the top 5 univariate for comparison
top5_uni = dict(uni_sorted[:5])
print()


# ── 6. Visualization ────────────────────────────────────────────────────────
print("Generating comparison plot...")

# Combine for plotting: top 5 univariate + all ML models
plot_data = {}
for name, sr in top5_uni.items():
    plot_data[f"Uni: {name}"] = sr
for name, sr in ml_sharpe.items():
    plot_data[f"ML: {name}"] = sr

# Sort by Sharpe
plot_sorted = dict(sorted(plot_data.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots(figsize=(14, 8))

names = list(plot_sorted.keys())
values = list(plot_sorted.values())
colors = ["#3498db" if n.startswith("Uni:") else "#e74c3c" for n in names]

bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", height=0.7)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Annualized Sharpe Ratio (Test Period)", fontsize=12)
ax.set_title("ML Portfolios vs. Best Univariate Characteristics\n"
             "(Rank-Weighted Long-Short, Test Period Only)", fontsize=13)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    x_pos = val + 0.02 if val >= 0 else val - 0.02
    ha = "left" if val >= 0 else "right"
    ax.text(x_pos, i, f"{val:+.2f}", va="center", ha=ha, fontsize=8, fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#3498db", label="Univariate Characteristic"),
                   Patch(facecolor="#e74c3c", label="ML Model")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("task_c_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to task_c_comparison.png")
print()


# ── 7. Cumulative return plot for top performers ─────────────────────────────
# Pick best univariate, best linear ML, and best nonlinear ML
best_uni_name = uni_sorted[0][0]
best_lin_name = max(["Lasso", "ElasticNet"], key=lambda x: ml_sharpe[x])
best_nonlin_name = "GradientBoosting"

# Compute univariate cumulative returns
sub = test_full[["permno", "yyyymm", best_uni_name, "ret"]].dropna(subset=[best_uni_name, "ret"])
sub["rank"] = sub.groupby("yyyymm")[best_uni_name].rank(method="average")
sub["signal"] = sub.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())
sub["abs_sum"] = sub.groupby("yyyymm")["signal"].transform(lambda x: x.abs().sum())
sub["weight"] = sub["signal"] / sub["abs_sum"]
uni_port = (sub["weight"] * sub["ret"]).groupby(sub["yyyymm"]).sum().sort_index()

lin_port = ml_port_returns[best_lin_name].sort_index()
nl_port  = ml_port_returns[best_nonlin_name].sort_index()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot((1 + uni_port).cumprod(), label=f"Uni: {best_uni_name} (SR={uni_sharpe[best_uni_name]:+.2f})",
        linewidth=1.5, color="#3498db")
ax.plot((1 + lin_port).cumprod(), label=f"ML: {best_lin_name} (SR={ml_sharpe[best_lin_name]:+.2f})",
        linewidth=1.5, color="#e67e22")
ax.plot((1 + nl_port).cumprod(), label=f"ML: {best_nonlin_name} (SR={ml_sharpe[best_nonlin_name]:+.2f})",
        linewidth=1.5, color="#e74c3c")

ax.set_xlabel("Date (yyyymm)", fontsize=11)
ax.set_ylabel("Cumulative Return ($1 invested)", fontsize=11)
ax.set_title("Cumulative Performance: Best Univariate vs. Best ML Portfolios (Test Period)", fontsize=13)
ax.legend(fontsize=10)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# Thin out x-axis labels — use actual yyyymm values for tick positions
tick_values = uni_port.index[::60]
ax.set_xticks(tick_values)
ax.set_xticklabels([str(d) for d in tick_values], rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig("task_c_cumulative.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to task_c_cumulative.png")
print()


# ── 8. Final summary table ──────────────────────────────────────────────────
print("=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)

rows = []
for name, sr in uni_sorted[:5]:
    rows.append({"Category": "Univariate", "Method": name, "Sharpe": sr})
for name in ["OLS", "Ridge", "Lasso", "ElasticNet",
             "RBF+Ridge", "RBF+Lasso", "RBF+ElasticNet",
             "PLS", "GradientBoosting"]:
    cat = "Linear ML" if name in ["OLS", "Ridge", "Lasso", "ElasticNet"] else "Non-linear ML"
    rows.append({"Category": cat, "Method": name, "Sharpe": ml_sharpe[name]})

final = pd.DataFrame(rows)
final = final.sort_values("Sharpe", ascending=False).reset_index(drop=True)
final["Sharpe"] = final["Sharpe"].map(lambda x: f"{x:+.3f}")
print(final.to_string(index=False))
print()
