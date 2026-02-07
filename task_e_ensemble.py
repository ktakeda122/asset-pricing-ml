"""
Task (e): Ensemble Portfolio for Maximum OOS Sharpe Ratio
=========================================================
- Combine best Linear (ElasticNet) + best Non-linear (GradientBoosting)
- Optimize ensemble weights on Validation set
- Evaluate on Test set
- Run for both Large Cap and Small Cap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from itertools import product as iprod
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def annualized_sharpe(mr):
    m, s = mr.mean(), mr.std(ddof=1)
    return (m / s) * np.sqrt(12) if s > 0 and not np.isnan(s) else np.nan


def portfolio_sharpe_from_signal(yyyymm, ret, signal):
    """Build rank-weighted L/S portfolio and return its annualized Sharpe."""
    tmp = pd.DataFrame({"yyyymm": yyyymm, "ret": ret, "signal": signal}).dropna()
    tmp["rank"] = tmp.groupby("yyyymm")["signal"].rank(method="average")
    tmp["w"] = tmp.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())
    tmp["abs_sum"] = tmp.groupby("yyyymm")["w"].transform(lambda x: x.abs().sum())
    tmp["w"] = tmp["w"] / tmp["abs_sum"]
    port = (tmp["w"] * tmp["ret"]).groupby(tmp["yyyymm"]).sum()
    return annualized_sharpe(port), port


def r2_oos(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum(y_true**2)


def run_universe(filepath, label):
    """Full pipeline for one universe: train models, ensemble, evaluate."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    # ── Data prep ────────────────────────────────────────────────────────
    df = pd.read_parquet(filepath)
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    feature_cols = [c for c in df.columns if c not in ["permno", "yyyymm", "ret"]]
    df = df.dropna(subset=["ret"]).reset_index(drop=True)

    train_mask = df["yyyymm"] <= 194511
    val_mask   = (df["yyyymm"] > 194511) & (df["yyyymm"] <= 195711)
    test_mask  = df["yyyymm"] > 195711

    df[feature_cols] = df.groupby("yyyymm")[feature_cols].transform(
        lambda x: x.fillna(x.median()))
    df[feature_cols] = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(df.loc[train_mask, feature_cols].values)
    y_tr   = df.loc[train_mask, "ret"].values
    X_va_s = scaler.transform(df.loc[val_mask, feature_cols].values)
    y_va   = df.loc[val_mask, "ret"].values
    X_te_s = scaler.transform(df.loc[test_mask, feature_cols].values)
    y_te   = df.loc[test_mask, "ret"].values

    test_yyyymm = df.loc[test_mask, "yyyymm"].values
    test_ret    = df.loc[test_mask, "ret"].values
    val_yyyymm  = df.loc[val_mask, "yyyymm"].values
    val_ret     = df.loc[val_mask, "ret"].values

    print(f"  Train: {X_tr_s.shape}  Val: {X_va_s.shape}  Test: {X_te_s.shape}")

    # ── Train candidate models ───────────────────────────────────────────
    models = {}

    # Ridge - tune
    best_r2, best_a = -np.inf, None
    for a in np.logspace(0, 6, 40):
        m = Ridge(alpha=a, max_iter=50000).fit(X_tr_s, y_tr)
        r2 = r2_oos(y_va, m.predict(X_va_s))
        if r2 > best_r2: best_r2, best_a = r2, a
    models["Ridge"] = Ridge(alpha=best_a, max_iter=50000).fit(X_tr_s, y_tr)

    # Lasso - tune
    best_r2, best_a = -np.inf, None
    for a in np.logspace(-5, 0, 30):
        m = Lasso(alpha=a, max_iter=50000).fit(X_tr_s, y_tr)
        r2 = r2_oos(y_va, m.predict(X_va_s))
        if r2 > best_r2: best_r2, best_a = r2, a
    models["Lasso"] = Lasso(alpha=best_a, max_iter=50000).fit(X_tr_s, y_tr)

    # ElasticNet - tune
    best_r2, best_a, best_l1 = -np.inf, None, None
    for l1 in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        for a in np.logspace(-5, 0, 30):
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=50000).fit(X_tr_s, y_tr)
            r2 = r2_oos(y_va, m.predict(X_va_s))
            if r2 > best_r2: best_r2, best_a, best_l1 = r2, a, l1
    models["ElasticNet"] = ElasticNet(alpha=best_a, l1_ratio=best_l1,
                                      max_iter=50000).fit(X_tr_s, y_tr)

    # PLS - tune
    best_r2, best_nc = -np.inf, None
    for nc in range(1, min(50, X_tr_s.shape[1]) + 1):
        pls = PLSRegression(n_components=nc).fit(X_tr_s, y_tr)
        r2 = r2_oos(y_va, pls.predict(X_va_s).ravel())
        if r2 > best_r2: best_r2, best_nc = r2, nc
    models["PLS"] = PLSRegression(n_components=best_nc).fit(X_tr_s, y_tr)

    # GBR - tune
    best_r2, best_gbr = -np.inf, None
    for combo in iprod([100, 300, 500], [2, 3, 4], [0.01, 0.05, 0.1], [0.8, 1.0]):
        ne, md, lr, ss = combo
        gbr = GradientBoostingRegressor(n_estimators=ne, max_depth=md, learning_rate=lr,
                                        subsample=ss, min_samples_leaf=10, random_state=42)
        gbr.fit(X_tr_s, y_tr)
        r2 = r2_oos(y_va, gbr.predict(X_va_s))
        if r2 > best_r2: best_r2, best_gbr = r2, gbr
    models["GBR"] = best_gbr

    print(f"  Trained: {list(models.keys())}")

    # ── Get predictions ──────────────────────────────────────────────────
    val_preds = {}
    test_preds = {}
    for name, mod in models.items():
        if name == "PLS":
            val_preds[name]  = mod.predict(X_va_s).ravel()
            test_preds[name] = mod.predict(X_te_s).ravel()
        else:
            val_preds[name]  = mod.predict(X_va_s)
            test_preds[name] = mod.predict(X_te_s)

    # ── Individual model portfolio Sharpe ratios ─────────────────────────
    print(f"\n  Individual model portfolio Sharpe ratios (Test):")
    indiv_sharpes = {}
    indiv_ports = {}
    for name in models:
        sr, port = portfolio_sharpe_from_signal(test_yyyymm, test_ret, test_preds[name])
        indiv_sharpes[name] = sr
        indiv_ports[name] = port
        print(f"    {name:15s}  SR = {sr:+.3f}")

    # ── Ensemble: optimize weights on Validation set ─────────────────────
    print(f"\n  Optimizing ensemble weights on Validation set...")

    model_names = list(models.keys())
    val_pred_matrix  = np.column_stack([val_preds[n] for n in model_names])
    test_pred_matrix = np.column_stack([test_preds[n] for n in model_names])

    # Grid search over weight combinations (simplex)
    # Generate weight vectors that sum to 1
    best_val_sr = -np.inf
    best_weights = None
    weight_steps = np.arange(0, 1.05, 0.1)

    # For 5 models, full grid is too large. Use two-stage approach:
    # Stage 1: Pairwise best (ElasticNet + GBR)
    # Stage 2: Add other models with remaining weight

    # Stage 1: Simple pairwise ElasticNet + GBR
    print(f"\n  Stage 1: ElasticNet + GBR pairwise optimization")
    enet_idx = model_names.index("ElasticNet")
    gbr_idx  = model_names.index("GBR")

    best_pair_sr = -np.inf
    best_pair_w = None
    for w_enet in np.arange(0, 1.01, 0.05):
        w_gbr = 1.0 - w_enet
        signal = w_enet * val_pred_matrix[:, enet_idx] + w_gbr * val_pred_matrix[:, gbr_idx]
        sr, _ = portfolio_sharpe_from_signal(val_yyyymm, val_ret, signal)
        if not np.isnan(sr) and sr > best_pair_sr:
            best_pair_sr = sr
            best_pair_w = (w_enet, w_gbr)

    print(f"    Best: w_ElasticNet={best_pair_w[0]:.2f}, w_GBR={best_pair_w[1]:.2f}, "
          f"Val SR={best_pair_sr:+.3f}")

    # Stage 2: Search over all 5 models with coarser grid
    print(f"\n  Stage 2: Full 5-model weight optimization (step=0.1)")
    best_full_sr = -np.inf
    best_full_weights = None
    n_models = len(model_names)

    # Generate simplex grid with step 0.1
    def simplex_grid(n, step=0.1):
        """Generate all weight vectors of length n that sum to 1 with given step."""
        levels = int(round(1.0 / step))
        if n == 1:
            yield [1.0]
            return
        for i in range(levels + 1):
            w = i * step
            for rest in simplex_grid(n - 1, step):
                total = w + sum(rest)
                if abs(total - 1.0) < 1e-6:
                    yield [w] + rest
                elif total < 1.0 + 1e-6:
                    # Not enough yet, only yield if n-1 can fill
                    pass
        # Simpler approach: just iterate
        return

    # More efficient: iterate using recursion
    count = 0
    for w0 in np.arange(0, 1.01, 0.1):
        for w1 in np.arange(0, 1.01 - w0, 0.1):
            for w2 in np.arange(0, 1.01 - w0 - w1, 0.1):
                for w3 in np.arange(0, 1.01 - w0 - w1 - w2, 0.1):
                    w4 = 1.0 - w0 - w1 - w2 - w3
                    if w4 < -0.01:
                        continue
                    w4 = max(0, w4)
                    weights = np.array([w0, w1, w2, w3, w4])
                    signal = val_pred_matrix @ weights
                    sr, _ = portfolio_sharpe_from_signal(val_yyyymm, val_ret, signal)
                    count += 1
                    if not np.isnan(sr) and sr > best_full_sr:
                        best_full_sr = sr
                        best_full_weights = weights

    print(f"    Searched {count} weight combinations")
    print(f"    Best weights: {dict(zip(model_names, [f'{w:.2f}' for w in best_full_weights]))}")
    print(f"    Val SR = {best_full_sr:+.3f}")

    # ── Evaluate all ensemble strategies on Test ─────────────────────────
    print(f"\n  TEST SET RESULTS:")
    results = {}

    # 1) Equal-weight ElasticNet + GBR (the requested 50/50)
    sig_equal = 0.5 * test_preds["ElasticNet"] + 0.5 * test_preds["GBR"]
    sr_equal, port_equal = portfolio_sharpe_from_signal(test_yyyymm, test_ret, sig_equal)
    results["Ensemble (50/50 ENet+GBR)"] = {"sr": sr_equal, "port": port_equal}
    print(f"    Ensemble 50/50 ENet+GBR:     SR = {sr_equal:+.3f}")

    # 2) Optimized pairwise
    sig_pair = best_pair_w[0]*test_preds["ElasticNet"] + best_pair_w[1]*test_preds["GBR"]
    sr_pair, port_pair = portfolio_sharpe_from_signal(test_yyyymm, test_ret, sig_pair)
    results[f"Ensemble (opt ENet+GBR)"] = {"sr": sr_pair, "port": port_pair}
    print(f"    Ensemble opt ENet+GBR:       SR = {sr_pair:+.3f}  "
          f"(w={best_pair_w[0]:.2f}/{best_pair_w[1]:.2f})")

    # 3) Full optimized ensemble
    sig_full = test_pred_matrix @ best_full_weights
    sr_full, port_full = portfolio_sharpe_from_signal(test_yyyymm, test_ret, sig_full)
    results["Ensemble (opt 5-model)"] = {"sr": sr_full, "port": port_full}
    print(f"    Ensemble opt 5-model:        SR = {sr_full:+.3f}  "
          f"(weights: {dict(zip(model_names, [f'{w:.1f}' for w in best_full_weights]))})")

    # 4) Equal-weight all 5 models
    sig_all_eq = test_pred_matrix.mean(axis=1)
    sr_all_eq, port_all_eq = portfolio_sharpe_from_signal(test_yyyymm, test_ret, sig_all_eq)
    results["Ensemble (equal 5-model)"] = {"sr": sr_all_eq, "port": port_all_eq}
    print(f"    Ensemble equal 5-model:      SR = {sr_all_eq:+.3f}")

    # Best individual
    best_indiv_name = max(indiv_sharpes, key=indiv_sharpes.get)
    best_indiv_sr = indiv_sharpes[best_indiv_name]
    results[f"Best Individual ({best_indiv_name})"] = {
        "sr": best_indiv_sr, "port": indiv_ports[best_indiv_name]
    }
    print(f"    Best individual ({best_indiv_name:8s}):  SR = {best_indiv_sr:+.3f}")

    # Find the overall best
    best_ensemble_name = max(results, key=lambda k: results[k]["sr"])
    best_sr = results[best_ensemble_name]["sr"]
    print(f"\n    >>> BEST: {best_ensemble_name}  SR = {best_sr:+.3f}")

    return results, indiv_sharpes, indiv_ports


# ══════════════════════════════════════════════════════════════════════════════
# RUN BOTH UNIVERSES
# ══════════════════════════════════════════════════════════════════════════════
lg_results, lg_indiv, lg_ports = run_universe("largeml.pq", "LARGE CAPS")
sm_results, sm_indiv, sm_ports = run_universe("smallml.pq", "SMALL CAPS")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Plot 1: Bar chart comparing strategies
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

for ax, (res, title) in zip(axes, [(lg_results, "Large Cap"), (sm_results, "Small Cap")]):
    names = list(res.keys())
    srs = [res[n]["sr"] for n in names]

    colors = []
    for n in names:
        if "Best Individual" in n:
            colors.append("#95a5a6")
        elif "50/50" in n:
            colors.append("#3498db")
        elif "opt" in n and "5-model" in n:
            colors.append("#e74c3c")
        elif "opt" in n:
            colors.append("#e67e22")
        else:
            colors.append("#2ecc71")

    bars = ax.barh(range(len(names)), srs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Annualized Sharpe Ratio", fontsize=11)
    ax.set_title(f"{title}: Ensemble vs. Individual", fontsize=12)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, srs)):
        ax.text(val + 0.05, i, f"{val:+.2f}", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("task_e_ensemble.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved task_e_ensemble.png")


# Plot 2: Cumulative returns - best ensemble vs best individual for each universe
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (res, indiv, title) in zip(axes,
    [(lg_results, lg_indiv, "Large Cap"), (sm_results, sm_indiv, "Small Cap")]):

    best_ens_name = max([k for k in res if "Ensemble" in k], key=lambda k: res[k]["sr"])
    best_ind_name = max(indiv, key=indiv.get)

    ens_port = res[best_ens_name]["port"].sort_index()
    ind_port = res[f"Best Individual ({best_ind_name})"]["port"].sort_index()

    ax.plot(ens_port.index, (1 + ens_port).cumprod().values,
            label=f"{best_ens_name}\n(SR={res[best_ens_name]['sr']:+.2f})",
            linewidth=1.5, color="#e74c3c")
    ax.plot(ind_port.index, (1 + ind_port).cumprod().values,
            label=f"Best Individual: {best_ind_name}\n(SR={indiv[best_ind_name]:+.2f})",
            linewidth=1.5, color="#3498db")

    ax.set_xlabel("Date (yyyymm)", fontsize=10)
    ax.set_ylabel("Cumulative Return ($1)", fontsize=10)
    ax.set_title(f"{title}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ticks = ens_port.index[::80]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(d) for d in ticks], rotation=45, fontsize=7)

plt.suptitle("Cumulative Performance: Best Ensemble vs. Best Individual Model (Test Period)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("task_e_cumulative.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved task_e_cumulative.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY: HIGHEST ACHIEVABLE SHARPE RATIOS")
print("=" * 70)

for label, res, indiv in [("Large Cap", lg_results, lg_indiv),
                           ("Small Cap", sm_results, sm_indiv)]:
    best_ens_name = max([k for k in res if "Ensemble" in k], key=lambda k: res[k]["sr"])
    best_ind_name = max(indiv, key=indiv.get)
    overall_best = max(res, key=lambda k: res[k]["sr"])

    print(f"\n  {label}:")
    print(f"    Best Individual:  {best_ind_name:15s}  SR = {indiv[best_ind_name]:+.3f}")
    print(f"    Best Ensemble:    {best_ens_name}")
    print(f"                      SR = {res[best_ens_name]['sr']:+.3f}")
    improvement = res[best_ens_name]["sr"] - indiv[best_ind_name]
    pct = improvement / abs(indiv[best_ind_name]) * 100
    print(f"    Improvement:      {improvement:+.3f} ({pct:+.1f}%)")
print()
