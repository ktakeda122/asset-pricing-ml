"""
Data Preparation: Time-Series Splits & Missing Value Imputation
================================================================
- Train:  first 20 calendar years of data
- Val:    next 12 calendar years
- Test:   remainder
- Imputation: cross-sectional median (within each month) -- no look-ahead bias
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load and prepare data ─────────────────────────────────────────────────
df = pd.read_parquet("largeml.pq")
df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

id_cols = ["permno", "yyyymm"]
target_col = "ret"
feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

print(f"Raw dataset: {df.shape[0]} rows, {len(feature_cols)} features")
print(f"Date range:  {df.yyyymm.min()} -{df.yyyymm.max()}")
print()

# ── 2. Drop rows where target (ret) is missing ──────────────────────────────
n_before = len(df)
df = df.dropna(subset=["ret"]).reset_index(drop=True)
print(f"Dropped {n_before - len(df)} rows with missing ret (includes 'C' delistings)")
print(f"Working dataset: {df.shape[0]} rows")
print()

# ── 3. Chronological train / val / test splits ──────────────────────────────
start_yyyymm = df["yyyymm"].min()
start_year = start_yyyymm // 100  # 1925

train_end_year = start_year + 20   # 1945
val_end_year   = train_end_year + 12  # 1957

# Use November as the boundary month (data starts Dec 1925)
# Train: up to and including 194511
# Val:   194512 through 195711
# Test:  195712 onward
train_cutoff = train_end_year * 100 + 11   # 194511
val_cutoff   = val_end_year * 100 + 11     # 195711

train_mask = df["yyyymm"] <= train_cutoff
val_mask   = (df["yyyymm"] > train_cutoff) & (df["yyyymm"] <= val_cutoff)
test_mask  = df["yyyymm"] > val_cutoff

print("=" * 70)
print("TIME-SERIES SPLITS")
print("=" * 70)
for label, mask in [("Train", train_mask), ("Validation", val_mask), ("Test (OOS)", test_mask)]:
    sub = df[mask]
    print(f"  {label:15s}  {sub.yyyymm.min()} -{sub.yyyymm.max()}  |  "
          f"{len(sub):>6,} rows  |  {sub.permno.nunique():>3} stocks  |  "
          f"{sub.yyyymm.nunique():>3} months")
print(f"  {'Total':15s}  {'':21s}  {len(df):>6,} rows")
assert len(df) == train_mask.sum() + val_mask.sum() + test_mask.sum(), "Rows don't add up!"
print()

# ── 4. Cross-sectional median imputation (no look-ahead bias) ───────────────
# For each month, fill NaN features with the cross-sectional median of that month.
# If the entire cross-section is NaN for a feature in a given month, fill with 0.
print("Imputing missing features with cross-sectional median (per month)...")
missing_before = df[feature_cols].isna().sum().sum()
total_cells = len(df) * len(feature_cols)

df[feature_cols] = (
    df.groupby("yyyymm")[feature_cols]
    .transform(lambda x: x.fillna(x.median()))
)

# Any remaining NaN (entire cross-section was NaN) ->fill with 0
missing_after_median = df[feature_cols].isna().sum().sum()
df[feature_cols] = df[feature_cols].fillna(0)
missing_after_zero = df[feature_cols].isna().sum().sum()

print(f"  Feature cells before imputation:  {missing_before:>10,} / {total_cells:,} missing "
      f"({missing_before / total_cells * 100:.1f}%)")
print(f"  After cross-sectional median fill: {missing_after_median:>10,} remaining")
print(f"  After zero-fill (full-NaN months):  {missing_after_zero:>10,} remaining")
print()

# ── 5. Save prepared data & split indices ────────────────────────────────────
df.to_parquet("largeml_prepared.pq", index=False)

# Also save split masks for easy reuse
splits = pd.DataFrame({
    "permno": df["permno"],
    "yyyymm": df["yyyymm"],
    "split": np.where(train_mask, "train", np.where(val_mask, "val", "test")),
})
splits.to_parquet("splits.pq", index=False)

print("Saved: largeml_prepared.pq  (imputed features + ret)")
print("Saved: splits.pq            (train/val/test labels)")
print()

# ── 6. Final confirmation ───────────────────────────────────────────────────
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
for label, split_label in [("Train", "train"), ("Validation", "val"), ("Test (OOS)", "test")]:
    mask = splits["split"] == split_label
    sub = df[mask]
    print(f"  {label:15s}  {sub.yyyymm.min()} -{sub.yyyymm.max()}  |  "
          f"{mask.sum():>6,} obs  |  {sub.permno.nunique():>3} stocks  |  "
          f"{sub.yyyymm.nunique():>3} months")
print(f"  {'Total':15s}  {'':21s}  {len(df):>6,} obs")
print()
print(f"Features: {len(feature_cols)}")
print(f"Remaining NaNs in features: {df[feature_cols].isna().sum().sum()}")
print(f"Remaining NaNs in ret:      {df['ret'].isna().sum()}")
