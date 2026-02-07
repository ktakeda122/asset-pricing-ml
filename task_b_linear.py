"""
Task (b) - Methods (i) & (ii): OLS and Penalized Linear Models
===============================================================
Models: OLS, Lasso, Ridge, ElasticNet
Hyperparameter tuning on Validation set, evaluation on Test set.

R^2_OOS = 1 - sum((y_true - y_pred)^2) / sum(y_true^2)
  (zero-forecast benchmark, standard in asset pricing; Gu, Kelly & Xiu 2020)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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

print(f"Train:  {X_train.shape}")
print(f"Val:    {X_val.shape}")
print(f"Test:   {X_test.shape}")
print()

# ── 2. Standardize features (fit on train only) ─────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)


# ── 3. R^2_OOS with zero-forecast benchmark ─────────────────────────────────
def r2_oos(y_true, y_pred):
    """R^2_OOS = 1 - SS_res / SS_total, where SS_total = sum(y_true^2) (zero benchmark)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return 1.0 - ss_res / ss_tot


# ── 4. OLS ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("(i) OLS - Linear Regression")
print("=" * 70)
ols = LinearRegression()
ols.fit(X_train_s, y_train)

r2_ols_val  = r2_oos(y_val, ols.predict(X_val_s))
r2_ols_test = r2_oos(y_test, ols.predict(X_test_s))
print(f"  Val  R2_OOS: {r2_ols_val:.6f}")
print(f"  Test R2_OOS: {r2_ols_test:.6f}")
print()


# ── 5. Penalized models: hyperparameter search ──────────────────────────────
# Ridge needs much larger alphas (L2 penalty scale differs from L1)
alphas_l1    = np.logspace(-6, 0, 50)      # Lasso / ElasticNet: 1e-6 to 1
alphas_ridge = np.logspace(-2, 6, 60)      # Ridge: 0.01 to 1,000,000
l1_ratios = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]  # for ElasticNet


def tune_model(ModelClass, alpha_grid, extra_params=None):
    """Tune alpha on validation set. Returns best model, best alpha, val R2."""
    best_r2 = -np.inf
    best_alpha = None
    best_model = None
    results = []

    for alpha in alpha_grid:
        params = {"alpha": alpha, "max_iter": 50000}
        if extra_params:
            params.update(extra_params)
        model = ModelClass(**params)
        model.fit(X_train_s, y_train)
        pred_val = model.predict(X_val_s)
        r2 = r2_oos(y_val, pred_val)
        results.append((alpha, r2))

        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_model = model

    return best_model, best_alpha, best_r2, results


# ── 5a. Ridge ────────────────────────────────────────────────────────────────
print("=" * 70)
print("(ii-a) Ridge Regression")
print("=" * 70)
ridge_model, ridge_alpha, ridge_val_r2, ridge_results = tune_model(Ridge, alphas_ridge)
r2_ridge_test = r2_oos(y_test, ridge_model.predict(X_test_s))
print(f"  Best alpha:  {ridge_alpha:.6e}")
print(f"  Val  R2_OOS: {ridge_val_r2:.6f}")
print(f"  Test R2_OOS: {r2_ridge_test:.6f}")
n_nonzero = np.sum(np.abs(ridge_model.coef_) > 1e-10)
print(f"  Non-zero coefficients: {n_nonzero}/{len(feature_cols)}")
print()


# ── 5b. Lasso ────────────────────────────────────────────────────────────────
print("=" * 70)
print("(ii-b) Lasso Regression")
print("=" * 70)
lasso_model, lasso_alpha, lasso_val_r2, lasso_results = tune_model(Lasso, alphas_l1)
r2_lasso_test = r2_oos(y_test, lasso_model.predict(X_test_s))
print(f"  Best alpha:  {lasso_alpha:.6e}")
print(f"  Val  R2_OOS: {lasso_val_r2:.6f}")
print(f"  Test R2_OOS: {r2_lasso_test:.6f}")
n_nonzero = np.sum(np.abs(lasso_model.coef_) > 1e-10)
print(f"  Non-zero coefficients: {n_nonzero}/{len(feature_cols)}")
print()


# ── 5c. ElasticNet ───────────────────────────────────────────────────────────
print("=" * 70)
print("(ii-c) ElasticNet Regression")
print("=" * 70)
best_enet_r2 = -np.inf
best_enet_model = None
best_enet_alpha = None
best_enet_l1 = None

for l1 in l1_ratios:
    model, alpha, val_r2, _ = tune_model(
        ElasticNet, alphas_l1, extra_params={"l1_ratio": l1}
    )
    if val_r2 > best_enet_r2:
        best_enet_r2 = val_r2
        best_enet_model = model
        best_enet_alpha = alpha
        best_enet_l1 = l1

r2_enet_test = r2_oos(y_test, best_enet_model.predict(X_test_s))
print(f"  Best alpha:    {best_enet_alpha:.6e}")
print(f"  Best l1_ratio: {best_enet_l1}")
print(f"  Val  R2_OOS:   {best_enet_r2:.6f}")
print(f"  Test R2_OOS:   {r2_enet_test:.6f}")
n_nonzero = np.sum(np.abs(best_enet_model.coef_) > 1e-10)
print(f"  Non-zero coefficients: {n_nonzero}/{len(feature_cols)}")
print()


# ── 6. Summary table ────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY: Out-of-Sample R^2 (Test Set)")
print("R^2_OOS = 1 - SS_res / sum(y^2)  [zero-forecast benchmark]")
print("=" * 70)

summary = pd.DataFrame({
    "Model": ["OLS", "Ridge", "Lasso", "ElasticNet"],
    "Best Alpha": ["-", f"{ridge_alpha:.2e}", f"{lasso_alpha:.2e}",
                   f"{best_enet_alpha:.2e}"],
    "Extra Param": ["-", "-", "-", f"l1={best_enet_l1}"],
    "Val R2_OOS": [r2_ols_val, ridge_val_r2, lasso_val_r2, best_enet_r2],
    "Test R2_OOS": [r2_ols_test, r2_ridge_test, r2_lasso_test, r2_enet_test],
})
summary["Val R2_OOS"]  = summary["Val R2_OOS"].map(lambda x: f"{x:.6f}")
summary["Test R2_OOS"] = summary["Test R2_OOS"].map(lambda x: f"{x:.6f}")

print(summary.to_string(index=False))
print()
