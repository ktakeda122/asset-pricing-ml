"""
Task (b) - Methods (iii), (iv), (v): Non-linear & Advanced ML Models
=====================================================================
(iii) RBFSampler + Penalized Linear (Lasso, Ridge, ElasticNet)
(iv)  PLS Regression
(v)   Gradient Boosting Regressor

R^2_OOS = 1 - sum((y_true - y_pred)^2) / sum(y_true^2)
  (zero-forecast benchmark)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data and split ───────────────────────────────────────────────────
df = pd.read_parquet("largeml_prepared.pq")
splits = pd.read_parquet("splits.pq")

feature_cols = [c for c in df.columns if c not in ["permno", "yyyymm", "ret"]]

train_mask = splits["split"] == "train"
val_mask   = splits["split"] == "val"
test_mask  = splits["split"] == "test"

X_train, y_train = df.loc[train_mask, feature_cols].values, df.loc[train_mask, "ret"].values
X_val,   y_val   = df.loc[val_mask,   feature_cols].values, df.loc[val_mask,   "ret"].values
X_test,  y_test  = df.loc[test_mask,  feature_cols].values, df.loc[test_mask,  "ret"].values

# Standardize (fit on train only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train_s.shape}  Val: {X_val_s.shape}  Test: {X_test_s.shape}")
print()


def r2_oos(y_true, y_pred):
    """R^2_OOS = 1 - SS_res / sum(y^2)  [zero-forecast benchmark]."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return 1.0 - ss_res / ss_tot


# Store all results
all_results = {}


# ══════════════════════════════════════════════════════════════════════════════
# (iii) RBFSampler + Penalized Linear
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("(iii) RBF Random Fourier Features + Penalized Linear Models")
print("=" * 70)

# Tune RBF gamma and n_components, then penalized model alpha
rbf_gammas = [0.001, 0.01, 0.1]
rbf_n_components = [100, 300, 500]
alphas_l1    = np.logspace(-5, 0, 30)
alphas_ridge = np.logspace(0, 5, 30)

for model_name, ModelClass, alpha_grid in [
    ("RBF+Ridge", Ridge, alphas_ridge),
    ("RBF+Lasso", Lasso, alphas_l1),
    ("RBF+ElasticNet", ElasticNet, alphas_l1),
]:
    print(f"\n  --- {model_name} ---")
    best_r2 = -np.inf
    best_cfg = None
    best_model = None
    best_rbf = None

    for gamma, n_comp in product(rbf_gammas, rbf_n_components):
        rbf = RBFSampler(gamma=gamma, n_components=n_comp, random_state=42)
        X_tr_rbf = rbf.fit_transform(X_train_s)
        X_va_rbf = rbf.transform(X_val_s)

        for alpha in alpha_grid:
            params = {"alpha": alpha, "max_iter": 50000}
            if ModelClass == ElasticNet:
                params["l1_ratio"] = 0.5
            m = ModelClass(**params)
            m.fit(X_tr_rbf, y_train)
            r2 = r2_oos(y_val, m.predict(X_va_rbf))

            if r2 > best_r2:
                best_r2 = r2
                best_cfg = {"gamma": gamma, "n_comp": n_comp, "alpha": alpha}
                best_model = m
                best_rbf = rbf

    # Evaluate on test
    X_te_rbf = best_rbf.transform(X_test_s)
    r2_test = r2_oos(y_test, best_model.predict(X_te_rbf))

    print(f"  Best config: gamma={best_cfg['gamma']}, "
          f"n_comp={best_cfg['n_comp']}, alpha={best_cfg['alpha']:.4e}")
    print(f"  Val  R2_OOS: {best_r2:.6f}")
    print(f"  Test R2_OOS: {r2_test:.6f}")

    all_results[model_name] = {
        "val_r2": best_r2, "test_r2": r2_test, "config": best_cfg
    }

print()


# ══════════════════════════════════════════════════════════════════════════════
# (iv) PLS Regression
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("(iv) PLS Regression")
print("=" * 70)

n_comp_range = list(range(1, min(50, X_train_s.shape[1]) + 1))

best_pls_r2 = -np.inf
best_pls_n = None
best_pls_model = None

for n_comp in n_comp_range:
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train_s, y_train)
    pred_val = pls.predict(X_val_s).ravel()
    r2 = r2_oos(y_val, pred_val)

    if r2 > best_pls_r2:
        best_pls_r2 = r2
        best_pls_n = n_comp
        best_pls_model = pls

r2_pls_test = r2_oos(y_test, best_pls_model.predict(X_test_s).ravel())

print(f"  Best n_components: {best_pls_n}")
print(f"  Val  R2_OOS: {best_pls_r2:.6f}")
print(f"  Test R2_OOS: {r2_pls_test:.6f}")

all_results["PLS"] = {
    "val_r2": best_pls_r2, "test_r2": r2_pls_test,
    "config": {"n_components": best_pls_n}
}
print()


# ══════════════════════════════════════════════════════════════════════════════
# (v) Gradient Boosting Regressor
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("(v) Gradient Boosting Regressor")
print("=" * 70)

# Grid search over key hyperparameters
gbr_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}

keys = list(gbr_grid.keys())
vals = list(gbr_grid.values())
combos = list(product(*vals))
print(f"  Searching {len(combos)} hyperparameter combinations...")

best_gbr_r2 = -np.inf
best_gbr_params = None
best_gbr_model = None

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    gbr = GradientBoostingRegressor(
        random_state=42,
        min_samples_leaf=10,
        **params,
    )
    gbr.fit(X_train_s, y_train)
    pred_val = gbr.predict(X_val_s)
    r2 = r2_oos(y_val, pred_val)

    if r2 > best_gbr_r2:
        best_gbr_r2 = r2
        best_gbr_params = params
        best_gbr_model = gbr

    if (i + 1) % 18 == 0:
        print(f"    {i + 1}/{len(combos)} done (best val R2 so far: {best_gbr_r2:.6f})")

r2_gbr_test = r2_oos(y_test, best_gbr_model.predict(X_test_s))

print(f"\n  Best params: {best_gbr_params}")
print(f"  Val  R2_OOS: {best_gbr_r2:.6f}")
print(f"  Test R2_OOS: {r2_gbr_test:.6f}")

# Feature importance (top 10)
importances = best_gbr_model.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("\n  Top 10 features by importance:")
for rank, idx in enumerate(top_idx, 1):
    print(f"    {rank:2d}. {feature_cols[idx]:30s}  importance={importances[idx]:.4f}")

all_results["GradientBoosting"] = {
    "val_r2": best_gbr_r2, "test_r2": r2_gbr_test,
    "config": best_gbr_params
}
print()


# ══════════════════════════════════════════════════════════════════════════════
# Combined summary table (including linear models from task_b_linear.py)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("COMBINED RESULTS: All Methods - Test R^2_OOS")
print("R^2_OOS = 1 - SS_res / sum(y^2)  [zero-forecast benchmark]")
print("=" * 70)

# Linear results (hardcoded from previous run)
linear_results = {
    "OLS":        {"test_r2": -739.949663, "val_r2": -0.338073},
    "Ridge":      {"test_r2": -6.328697,   "val_r2":  0.123478},
    "Lasso":      {"test_r2":  0.165044,   "val_r2":  0.227058},
    "ElasticNet": {"test_r2":  0.170100,   "val_r2":  0.227208},
}

combined = {**linear_results, **all_results}

rows = []
for name, res in combined.items():
    rows.append({
        "Method": name,
        "Val R2_OOS": res["val_r2"],
        "Test R2_OOS": res["test_r2"],
    })

summary = pd.DataFrame(rows)
summary = summary.sort_values("Test R2_OOS", ascending=False).reset_index(drop=True)

# Format for display
disp = summary.copy()
disp["Val R2_OOS"]  = disp["Val R2_OOS"].map(lambda x: f"{x:+.4f}" if abs(x) < 100 else f"{x:+.1f}")
disp["Test R2_OOS"] = disp["Test R2_OOS"].map(lambda x: f"{x:+.4f}" if abs(x) < 100 else f"{x:+.1f}")
print(disp.to_string(index=False))
print()
