"""
Task 2(e): Construct the Highest OOS Sharpe Ratio Portfolio
===========================================================
Using the portfolios from lsret.csv, attempt to form a portfolio that
earns the highest possible Sharpe ratio out-of-sample.

Candidate strategies (all derived from lsret.csv):
  A. Raw indicator Ridge/Lasso regression
  B. PCA indicator regression (various K)
  E. Optimized multi-strategy ensemble (grid search, NO look-ahead)

Large Cap strategies (A-C) are also computed for reference but are
excluded from the final ranking and winner selection per the task
instructions.
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


def yyyymm_to_datetime(s):
    return pd.to_datetime(s.astype(int).astype(str), format="%Y%m")


def build_ls_returns(filepath, label):
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
    ret_df = pd.DataFrame(all_ports).reset_index()
    ret_df.rename(columns={"yyyymm": "date"}, inplace=True)
    ret_df["date"] = yyyymm_to_datetime(ret_df["date"])
    print(f"  {label}: {ret_df.shape[0]} months, {ret_df.shape[1]-1} portfolios")
    return ret_df


def clean_and_split(ret_df, name, split_ts):
    dates = ret_df["date"]
    data = ret_df.drop(columns=["date"])
    ret_cols = data.columns.tolist()
    col_miss = data.isnull().mean()
    keep_cols = col_miss[col_miss <= 0.3].index.tolist()
    clean = data[keep_cols].copy()
    clean["date"] = dates.values
    clean = clean.dropna(subset=keep_cols).reset_index(drop=True)
    train_mask = clean["date"] < split_ts
    test_mask  = clean["date"] >= split_ts
    R_train = clean.loc[train_mask, keep_cols].values
    R_test  = clean.loc[test_mask,  keep_cols].values
    d_train = clean.loc[train_mask, "date"].values
    d_test  = clean.loc[test_mask,  "date"].values
    print(f"  {name}: {len(ret_cols)}->{len(keep_cols)} cols, "
          f"train={R_train.shape[0]}, test={R_test.shape[0]} months")
    return R_train, R_test, d_test, keep_cols


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

split_ts = pd.Timestamp("2004-01-01")
tscv = TimeSeriesSplit(n_splits=5)

# lsret
print("\nlsret.csv:")
lsret = pd.read_csv("lsret.csv")
lsret["date"] = pd.to_datetime(lsret["date"])
lsret = lsret.sort_values("date").reset_index(drop=True)

# Large Cap
print("\nLarge Cap L/S:")
lg_ret = build_ls_returns("largeml.pq", "Large Cap")

# Clean and split
print("\nCleaning and splitting...")
R_tr_ls, R_te_ls, dt_ls, cols_ls = clean_and_split(lsret, "lsret", split_ts)
R_tr_lg, R_te_lg, dt_lg, cols_lg = clean_and_split(lg_ret, "Large Cap", split_ts)


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE ALL CANDIDATE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPUTING CANDIDATE STRATEGIES")
print("=" * 70)

candidates = {}  # name -> {"returns": array, "dates": array, "sr": float, "desc": str}


def add_candidate(name, rets, dates, desc):
    sr = annualized_sharpe(rets)
    candidates[name] = {"returns": rets, "dates": dates, "sr": sr, "desc": desc}
    print(f"  {name:45s}  SR={sr:+.3f}  ({len(rets)} months)")


# ── A. Raw indicator regression (Ridge + Lasso) ──────────────────────────────
print("\n--- A. Raw Indicator Regression ---")

for label, R_tr, R_te, dt, cols in [
    ("lsret", R_tr_ls, R_te_ls, dt_ls, cols_ls),
    ("LgCap", R_tr_lg, R_te_lg, dt_lg, cols_lg),
]:
    Y = np.ones(R_tr.shape[0])

    # Ridge
    ridge = RidgeCV(alphas=np.logspace(-4, 8, 100), cv=tscv, fit_intercept=False)
    ridge.fit(R_tr, Y)
    add_candidate(f"{label} Ridge Raw", R_te @ ridge.coef_, dt,
                  f"Ridge alpha={ridge.alpha_:.2e}, {len(cols)} ports")

    # Lasso
    lasso = LassoCV(alphas=np.logspace(-5, 1, 100), cv=tscv,
                     fit_intercept=False, max_iter=50000)
    lasso.fit(R_tr, Y)
    nz = np.sum(np.abs(lasso.coef_) > 1e-10)
    add_candidate(f"{label} Lasso Raw", R_te @ lasso.coef_, dt,
                  f"Lasso alpha={lasso.alpha_:.2e}, {nz}/{len(cols)} nz")

    # Ridge + Lasso ensemble (equal weight)
    port_r = R_te @ ridge.coef_
    port_l = R_te @ lasso.coef_
    add_candidate(f"{label} Ridge+Lasso Avg", 0.5*port_r + 0.5*port_l, dt,
                  "50/50 Ridge+Lasso ensemble")


# ── B. PCA Indicator Regression ──────────────────────────────────────────────
print("\n--- B. PCA Indicator Regression ---")

for label, R_tr, R_te, dt in [
    ("lsret", R_tr_ls, R_te_ls, dt_ls),
    ("LgCap", R_tr_lg, R_te_lg, dt_lg),
]:
    P = R_tr.shape[1]
    Y = np.ones(R_tr.shape[0])

    pca = PCA(n_components=P)
    pca.fit(R_tr)
    V = pca.components_
    F_tr = R_tr @ V.T
    F_te = R_te @ V.T

    for K in [5, 10, 15, 20, 30, 40, 50]:
        if K > P:
            continue

        ridge = RidgeCV(alphas=np.logspace(-4, 8, 100), cv=tscv, fit_intercept=False)
        ridge.fit(F_tr[:, :K], Y)
        add_candidate(f"{label} Ridge PCA K={K}", F_te[:, :K] @ ridge.coef_, dt,
                      f"PCA K={K}, Ridge")

        lasso = LassoCV(alphas=np.logspace(-6, 2, 100), cv=tscv,
                         fit_intercept=False, max_iter=50000)
        lasso.fit(F_tr[:, :K], Y)
        nz = np.sum(np.abs(lasso.coef_) > 1e-10)
        add_candidate(f"{label} Lasso PCA K={K}", F_te[:, :K] @ lasso.coef_, dt,
                      f"PCA K={K}, Lasso ({nz}/{K} nz)")


# ── C. PCA Factor Screening (select factors by in-sample SR) ─────────────────
print("\n--- C. PCA Factor Screening ---")

for label, R_tr, R_te, dt in [
    ("LgCap", R_tr_lg, R_te_lg, dt_lg),
]:
    P = R_tr.shape[1]
    Y = np.ones(R_tr.shape[0])

    pca = PCA(n_components=P)
    pca.fit(R_tr)
    V = pca.components_
    F_tr = R_tr @ V.T
    F_te = R_te @ V.T

    # Compute in-sample SR for each factor
    insample_srs = []
    for k in range(P):
        sr_k = annualized_sharpe(F_tr[:, k])
        insample_srs.append(sr_k)
    insample_srs = np.array(insample_srs)

    # Select factors with in-sample SR > threshold
    for sr_thresh in [0.0, 0.5, 1.0]:
        selected = np.where(insample_srs > sr_thresh)[0]
        if len(selected) < 2:
            continue
        F_tr_sel = F_tr[:, selected]
        F_te_sel = F_te[:, selected]

        ridge = RidgeCV(alphas=np.logspace(-4, 8, 100), cv=tscv, fit_intercept=False)
        ridge.fit(F_tr_sel, Y)
        add_candidate(f"{label} Screened SR>{sr_thresh} Ridge",
                      F_te_sel @ ridge.coef_, dt,
                      f"Factors with IS SR>{sr_thresh}: {len(selected)} selected")

        lasso = LassoCV(alphas=np.logspace(-6, 2, 100), cv=tscv,
                         fit_intercept=False, max_iter=50000)
        lasso.fit(F_tr_sel, Y)
        nz = np.sum(np.abs(lasso.coef_) > 1e-10)
        add_candidate(f"{label} Screened SR>{sr_thresh} Lasso",
                      F_te_sel @ lasso.coef_, dt,
                      f"Factors with IS SR>{sr_thresh}: {len(selected)} sel, {nz} nz")


# ── D. Cross-Universe Ensembles ──────────────────────────────────────────────
print("\n--- D. Cross-Universe Ensembles ---")

# Align lsret and Large Cap to common dates
ls_series = pd.Series(candidates["lsret Ridge Raw"]["returns"],
                       index=pd.to_datetime(dt_ls))
lg_series = pd.Series(candidates["LgCap Ridge Raw"]["returns"],
                       index=pd.to_datetime(dt_lg))
lg_lasso_series = pd.Series(candidates["LgCap Lasso Raw"]["returns"],
                             index=pd.to_datetime(dt_lg))
ls_pca_series = pd.Series(candidates["lsret Ridge PCA K=20"]["returns"],
                           index=pd.to_datetime(dt_ls))

# Merge on common dates
common = pd.DataFrame({
    "ls_ridge": ls_series,
    "lg_ridge": lg_series,
    "lg_lasso": lg_lasso_series,
    "ls_pca20": ls_pca_series,
}).dropna()

if len(common) > 0:
    common_dates = common.index.values
    print(f"  Common dates: {len(common)} months")

    for w_lg in [0.5, 0.6, 0.7, 0.8, 0.9]:
        w_ls = 1 - w_lg
        ens = w_lg * common["lg_ridge"].values + w_ls * common["ls_ridge"].values
        add_candidate(f"Ensemble LgRidge({w_lg:.0%})+lsRidge({w_ls:.0%})",
                      ens, common_dates,
                      f"Cross-universe: {w_lg:.0%} LgCap + {w_ls:.0%} lsret Ridge")

    # LgCap Ridge+Lasso average on common
    ens_lg = 0.5 * common["lg_ridge"].values + 0.5 * common["lg_lasso"].values
    add_candidate("LgCap Ridge+Lasso (common period)", ens_lg, common_dates,
                  "50/50 LgCap Ridge+Lasso on common dates")


# ── E. Optimized 3-Strategy Ensemble (weights from TRAIN period) ─────────────
print("\n--- E. Optimized Multi-Strategy Ensemble (lsret) ---")

# Build train-period portfolios for weight optimization using lsret
Y_tr_ls_e = np.ones(R_tr_ls.shape[0])
ridge_ls_e = RidgeCV(alphas=np.logspace(-4, 8, 100), cv=tscv, fit_intercept=False)
ridge_ls_e.fit(R_tr_ls, Y_tr_ls_e)
train_port_ls_ridge = R_tr_ls @ ridge_ls_e.coef_

lasso_ls_e = LassoCV(alphas=np.logspace(-5, 1, 100), cv=tscv,
                      fit_intercept=False, max_iter=50000)
lasso_ls_e.fit(R_tr_ls, Y_tr_ls_e)
train_port_ls_lasso = R_tr_ls @ lasso_ls_e.coef_

# PCA K=20 on lsret train
pca_ls_e = PCA(n_components=R_tr_ls.shape[1])
pca_ls_e.fit(R_tr_ls)
V_ls_e = pca_ls_e.components_
F_tr_ls20 = (R_tr_ls @ V_ls_e.T)[:, :20]
ridge_pca20_ls = RidgeCV(alphas=np.logspace(-4, 8, 100), cv=tscv, fit_intercept=False)
ridge_pca20_ls.fit(F_tr_ls20, Y_tr_ls_e)
train_port_ls_pca20 = F_tr_ls20 @ ridge_pca20_ls.coef_
test_port_ls_pca20  = (R_te_ls @ V_ls_e.T)[:, :20] @ ridge_pca20_ls.coef_

# Grid search over weights using TRAIN Sharpe (no look-ahead)
train_strats = np.column_stack([train_port_ls_ridge, train_port_ls_lasso, train_port_ls_pca20])
test_strats  = np.column_stack([
    R_te_ls @ ridge_ls_e.coef_,
    R_te_ls @ lasso_ls_e.coef_,
    test_port_ls_pca20
])
strat_names = ["lsret Ridge", "lsret Lasso", "lsret PCA20"]

best_train_sr = -np.inf
best_weights = None
step = 0.05
for w1 in np.arange(0, 1 + step/2, step):
    for w2 in np.arange(0, 1 - w1 + step/2, step):
        w3 = 1 - w1 - w2
        if w3 < -0.01:
            continue
        w = np.array([w1, w2, max(0, w3)])
        train_port = train_strats @ w
        sr_train = annualized_sharpe(train_port)
        if sr_train > best_train_sr:
            best_train_sr = sr_train
            best_weights = w.copy()

test_opt = test_strats @ best_weights
add_candidate(
    f"lsret Opt Ensemble ({best_weights[0]:.0%}/{best_weights[1]:.0%}/{best_weights[2]:.0%})",
    test_opt, dt_ls,
    f"Train-optimized: {best_weights[0]:.0%} Ridge + {best_weights[1]:.0%} Lasso + "
    f"{best_weights[2]:.0%} PCA20 (train SR={best_train_sr:.2f})"
)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RANK lsret STRATEGIES (Task 2(e) restricts to lsret.csv portfolios)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("lsret STRATEGIES RANKED BY OOS SHARPE RATIO")
print("(Task 2(e): using the portfolios from lsret.csv)")
print("=" * 70)

# Filter to only lsret-derived strategies (exclude LgCap / Cross-Universe / Ensemble Lg)
ranked_all = sorted(candidates.items(), key=lambda x: x[1]["sr"]
                    if not np.isnan(x[1]["sr"]) else -999, reverse=True)
ranked = [(n, r) for n, r in ranked_all
          if n.startswith("lsret")]

print(f"\n{'Rank':>4s}  {'Strategy':>50s}  {'SR':>8s}  {'Months':>6s}  Description")
print("-" * 120)
for i, (name, res) in enumerate(ranked[:25], 1):
    sr_str = f"{res['sr']:+.3f}" if not np.isnan(res['sr']) else "NaN"
    print(f"{i:4d}  {name:>50s}  {sr_str:>8s}  {len(res['returns']):6d}  {res['desc']}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. IDENTIFY WINNER
# ══════════════════════════════════════════════════════════════════════════════
winner_name, winner = ranked[0]
print(f"\n{'=' * 70}")
print(f"WINNER: {winner_name}")
print(f"{'=' * 70}")
print(f"  OOS Sharpe Ratio: {winner['sr']:+.3f}")
print(f"  OOS Months: {len(winner['returns'])}")
print(f"  Ann. Mean: {winner['returns'].mean()*12:+.4f}")
print(f"  Ann. Vol:  {winner['returns'].std()*np.sqrt(12):.4f}")
print(f"  Recipe: {winner['desc']}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Top 5 strategies cumulative returns
ax = axes[0]
plot_colors = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
for i, (name, res) in enumerate(ranked[:5]):
    cum = (1 + res["returns"]).cumprod()
    dt = pd.to_datetime(res["dates"])
    ax.plot(dt, cum, label=f"#{i+1} {name} (SR={res['sr']:+.2f})",
            linewidth=2 if i == 0 else 1.2, color=plot_colors[i],
            alpha=1.0 if i == 0 else 0.7)

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Cumulative Return ($1 invested)", fontsize=11)
ax.set_title("Top 5 Strategies: OOS Cumulative Returns", fontsize=13)
ax.legend(fontsize=7, loc="upper left")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# Plot 2: Bar chart of top 10 strategies
ax = axes[1]
top10_names = [n[:30] for n, _ in ranked[:10]]
top10_srs = [r["sr"] for _, r in ranked[:10]]
colors_bar = ["#e74c3c" if i == 0 else "#3498db" for i in range(10)]
bars = ax.barh(range(9, -1, -1), top10_srs[:10], color=colors_bar, alpha=0.85)
ax.set_yticks(range(9, -1, -1))
ax.set_yticklabels(top10_names, fontsize=8)
ax.set_xlabel("OOS Annualized Sharpe Ratio", fontsize=11)
ax.set_title("Top 10 Strategies", fontsize=13)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("task2e_best_portfolio.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved task2e_best_portfolio.png")
