"""
Task (b) - Small Caps: All ML Methods (i)-(v)
==============================================
Data prep + OLS + Ridge/Lasso/ElasticNet + RBF variants + PLS + GBR
Same methodology as large-cap analysis, applied to smallml.pq.

R^2_OOS = 1 - sum((y_true - y_pred)^2) / sum(y_true^2)
  (zero-forecast benchmark)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.kernel_approximation import RBFSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from itertools import product
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_parquet("smallml.pq")
df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

id_cols = ["permno", "yyyymm"]
target_col = "ret"
feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

print("=" * 70)
print("DATA PREPARATION - SMALL CAPS")
print("=" * 70)
print(f"Raw: {df.shape[0]:,} rows, {len(feature_cols)} features")

# Drop missing returns
n_before = len(df)
df = df.dropna(subset=["ret"]).reset_index(drop=True)
print(f"Dropped {n_before - len(df)} rows with missing ret")
print(f"Working: {len(df):,} rows")

# Time-series split (same cutoffs as large-cap)
train_cutoff = 194511
val_cutoff   = 195711

train_mask = df["yyyymm"] <= train_cutoff
val_mask   = (df["yyyymm"] > train_cutoff) & (df["yyyymm"] <= val_cutoff)
test_mask  = df["yyyymm"] > val_cutoff

print(f"\nSplits:")
for label, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
    sub = df[mask]
    print(f"  {label:6s}  {sub.yyyymm.min()} - {sub.yyyymm.max()}  |  "
          f"{mask.sum():>6,} rows  |  {sub.permno.nunique():>3} stocks  |  "
          f"{sub.yyyymm.nunique():>3} months")

# Cross-sectional median imputation (per month, no look-ahead)
missing_before = df[feature_cols].isna().sum().sum()
total_cells = len(df) * len(feature_cols)

df[feature_cols] = (
    df.groupby("yyyymm")[feature_cols]
    .transform(lambda x: x.fillna(x.median()))
)
df[feature_cols] = df[feature_cols].fillna(0)

print(f"\nImputation: {missing_before:,} / {total_cells:,} feature cells were NaN "
      f"({missing_before/total_cells*100:.1f}%)")
print(f"Remaining NaNs: {df[feature_cols].isna().sum().sum()}")
print()

# Extract arrays
X_train, y_train = df.loc[train_mask, feature_cols].values, df.loc[train_mask, "ret"].values
X_val,   y_val   = df.loc[val_mask,   feature_cols].values, df.loc[val_mask,   "ret"].values
X_test,  y_test  = df.loc[test_mask,  feature_cols].values, df.loc[test_mask,  "ret"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train_s.shape}  Val: {X_val_s.shape}  Test: {X_test_s.shape}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def r2_oos(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return 1.0 - ss_res / ss_tot


def tune_model(ModelClass, alpha_grid, extra_params=None):
    best_r2 = -np.inf
    best_alpha = None
    best_model = None
    for alpha in alpha_grid:
        params = {"alpha": alpha, "max_iter": 50000}
        if extra_params:
            params.update(extra_params)
        model = ModelClass(**params)
        model.fit(X_train_s, y_train)
        r2 = r2_oos(y_val, model.predict(X_val_s))
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_model = model
    return best_model, best_alpha, best_r2


all_results = {}
alphas_l1    = np.logspace(-5, 0, 30)
alphas_ridge = np.logspace(0, 6, 40)


# ══════════════════════════════════════════════════════════════════════════════
# 3. (i) OLS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("(i) OLS")
print("=" * 70)
ols = LinearRegression().fit(X_train_s, y_train)
r2_val  = r2_oos(y_val, ols.predict(X_val_s))
r2_test = r2_oos(y_test, ols.predict(X_test_s))
print(f"  Val R2: {r2_val:.4f}   Test R2: {r2_test:.4f}")
all_results["OLS"] = {"val_r2": r2_val, "test_r2": r2_test, "info": "-"}


# ══════════════════════════════════════════════════════════════════════════════
# 4. (ii) Ridge / Lasso / ElasticNet
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("(ii) Penalized Linear Models")
print("=" * 70)

# Ridge
m, a, vr2 = tune_model(Ridge, alphas_ridge)
tr2 = r2_oos(y_test, m.predict(X_test_s))
n_nz = np.sum(np.abs(m.coef_) > 1e-10)
print(f"  Ridge       alpha={a:.2e}  Val R2={vr2:.4f}  Test R2={tr2:.4f}  coefs={n_nz}")
all_results["Ridge"] = {"val_r2": vr2, "test_r2": tr2, "info": f"a={a:.2e}"}

# Lasso
m, a, vr2 = tune_model(Lasso, alphas_l1)
tr2 = r2_oos(y_test, m.predict(X_test_s))
n_nz = np.sum(np.abs(m.coef_) > 1e-10)
print(f"  Lasso       alpha={a:.2e}  Val R2={vr2:.4f}  Test R2={tr2:.4f}  coefs={n_nz}")
all_results["Lasso"] = {"val_r2": vr2, "test_r2": tr2, "info": f"a={a:.2e}, {n_nz} feats"}

# ElasticNet (search over l1_ratio too)
best_enet_r2 = -np.inf
best_enet = None
for l1 in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    m_tmp, a_tmp, vr2_tmp = tune_model(ElasticNet, alphas_l1, {"l1_ratio": l1})
    if vr2_tmp > best_enet_r2:
        best_enet_r2 = vr2_tmp
        best_enet = (m_tmp, a_tmp, l1)

m, a, l1 = best_enet
tr2 = r2_oos(y_test, m.predict(X_test_s))
n_nz = np.sum(np.abs(m.coef_) > 1e-10)
print(f"  ElasticNet  alpha={a:.2e}  l1={l1}  Val R2={best_enet_r2:.4f}  Test R2={tr2:.4f}  coefs={n_nz}")
all_results["ElasticNet"] = {"val_r2": best_enet_r2, "test_r2": tr2, "info": f"a={a:.2e}, l1={l1}"}


# ══════════════════════════════════════════════════════════════════════════════
# 5. (iii) RBF + Penalized Linear
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("(iii) RBF Random Fourier Features + Penalized Linear")
print("=" * 70)

rbf_gammas = [0.001, 0.01, 0.1]
rbf_n_components = [100, 300, 500]

for model_name, ModelClass, alpha_grid, extra in [
    ("RBF+Ridge", Ridge, np.logspace(0, 5, 25), None),
    ("RBF+Lasso", Lasso, np.logspace(-5, 0, 25), None),
    ("RBF+ElasticNet", ElasticNet, np.logspace(-5, 0, 25), {"l1_ratio": 0.5}),
]:
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
            if extra:
                params.update(extra)
            mod = ModelClass(**params)
            mod.fit(X_tr_rbf, y_train)
            r2 = r2_oos(y_val, mod.predict(X_va_rbf))
            if r2 > best_r2:
                best_r2 = r2
                best_cfg = {"gamma": gamma, "n_comp": n_comp, "alpha": alpha}
                best_model = mod
                best_rbf = rbf

    X_te_rbf = best_rbf.transform(X_test_s)
    tr2 = r2_oos(y_test, best_model.predict(X_te_rbf))
    print(f"  {model_name:16s}  g={best_cfg['gamma']}  nc={best_cfg['n_comp']}  "
          f"a={best_cfg['alpha']:.2e}  Val R2={best_r2:.4f}  Test R2={tr2:.4f}")
    all_results[model_name] = {"val_r2": best_r2, "test_r2": tr2,
                               "info": f"g={best_cfg['gamma']}, nc={best_cfg['n_comp']}"}


# ══════════════════════════════════════════════════════════════════════════════
# 6. (iv) PLS Regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("(iv) PLS Regression")
print("=" * 70)

best_pls_r2 = -np.inf
best_pls_n = None
best_pls = None

for n_comp in range(1, min(50, X_train_s.shape[1]) + 1):
    pls = PLSRegression(n_components=n_comp).fit(X_train_s, y_train)
    r2 = r2_oos(y_val, pls.predict(X_val_s).ravel())
    if r2 > best_pls_r2:
        best_pls_r2 = r2
        best_pls_n = n_comp
        best_pls = pls

tr2 = r2_oos(y_test, best_pls.predict(X_test_s).ravel())
print(f"  n_components={best_pls_n}  Val R2={best_pls_r2:.4f}  Test R2={tr2:.4f}")
all_results["PLS"] = {"val_r2": best_pls_r2, "test_r2": tr2, "info": f"nc={best_pls_n}"}


# ══════════════════════════════════════════════════════════════════════════════
# 7. (v) Gradient Boosting Regressor
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("(v) Gradient Boosting Regressor")
print("=" * 70)

gbr_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}
keys = list(gbr_grid.keys())
combos = list(product(*gbr_grid.values()))
print(f"  Searching {len(combos)} combinations...")

best_gbr_r2 = -np.inf
best_gbr_params = None
best_gbr = None

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    gbr = GradientBoostingRegressor(random_state=42, min_samples_leaf=10, **params)
    gbr.fit(X_train_s, y_train)
    r2 = r2_oos(y_val, gbr.predict(X_val_s))
    if r2 > best_gbr_r2:
        best_gbr_r2 = r2
        best_gbr_params = params
        best_gbr = gbr
    if (i + 1) % 18 == 0:
        print(f"    {i+1}/{len(combos)} done (best val R2: {best_gbr_r2:.4f})")

tr2 = r2_oos(y_test, best_gbr.predict(X_test_s))
print(f"\n  Best: {best_gbr_params}")
print(f"  Val R2={best_gbr_r2:.4f}  Test R2={tr2:.4f}")

# Top 10 features
imp = best_gbr.feature_importances_
top_idx = np.argsort(imp)[::-1][:10]
print("  Top 10 features:")
for rank, idx in enumerate(top_idx, 1):
    print(f"    {rank:2d}. {feature_cols[idx]:30s}  imp={imp[idx]:.4f}")
all_results["GradientBoosting"] = {"val_r2": best_gbr_r2, "test_r2": tr2,
                                   "info": str(best_gbr_params)}


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SMALL-CAP RESULTS: R^2_OOS  [1 - SS_res / sum(y^2)]")
print("=" * 70)

rows = []
for name in ["OLS", "Ridge", "Lasso", "ElasticNet",
             "RBF+Ridge", "RBF+Lasso", "RBF+ElasticNet",
             "PLS", "GradientBoosting"]:
    r = all_results[name]
    rows.append({"Method": name, "Val R2_OOS": r["val_r2"], "Test R2_OOS": r["test_r2"],
                 "Info": r["info"]})

summary = pd.DataFrame(rows).sort_values("Test R2_OOS", ascending=False)

disp = summary.copy()
disp["Val R2_OOS"]  = disp["Val R2_OOS"].map(
    lambda x: f"{x:+.4f}" if abs(x) < 100 else f"{x:+.1f}")
disp["Test R2_OOS"] = disp["Test R2_OOS"].map(
    lambda x: f"{x:+.4f}" if abs(x) < 100 else f"{x:+.1f}")
print(disp.to_string(index=False))

# Save for later use
summary.to_csv("task_b_small_results.csv", index=False)
print("\nSaved to task_b_small_results.csv")
