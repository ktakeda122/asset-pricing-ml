"""
Task (a) - Small Caps: Rank-Weighted Long-Short Portfolios for Each Characteristic
===================================================================================
Same methodology as large-cap analysis, applied to smallml.pq.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load and prepare ─────────────────────────────────────────────────────
df = pd.read_parquet("smallml.pq")
df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

id_cols = ["permno", "yyyymm"]
target_col = "ret"
feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

print("=" * 70)
print("SMALL-CAP DATASET SUMMARY")
print("=" * 70)
print(f"  Shape:          {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Date range:     {df.yyyymm.min()} - {df.yyyymm.max()}")
print(f"  Unique months:  {df.yyyymm.nunique()}")
print(f"  Unique stocks:  {df.permno.nunique()}")
print(f"  Features:       {len(feature_cols)}")
n_ret_nan = df['ret'].isna().sum()
print(f"  Missing ret:    {n_ret_nan} (includes 'C' delistings)")
miss_pct = df[feature_cols].isna().sum().sum() / (len(df) * len(feature_cols)) * 100
print(f"  Feature NaN:    {miss_pct:.1f}% of all feature cells")
print()


# ── 2. Annualized Sharpe ────────────────────────────────────────────────────
def annualized_sharpe(monthly_returns):
    mean_r = monthly_returns.mean()
    std_r = monthly_returns.std(ddof=1)
    if std_r == 0 or np.isnan(std_r):
        return np.nan
    return (mean_r / std_r) * np.sqrt(12)


# ── 3. Loop over all characteristics ────────────────────────────────────────
results = {}

for i, char in enumerate(feature_cols):
    sub = df[["permno", "yyyymm", char, "ret"]].dropna(subset=[char, "ret"])
    if len(sub) == 0:
        results[char] = {"sharpe": np.nan, "mean": np.nan, "std": np.nan, "n_months": 0}
        continue

    sub["rank"] = sub.groupby("yyyymm")[char].rank(method="average")
    sub["signal"] = sub.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())
    sub["abs_sum"] = sub.groupby("yyyymm")["signal"].transform(lambda x: x.abs().sum())
    sub["weight"] = sub["signal"] / sub["abs_sum"]
    port_ret = (sub["weight"] * sub["ret"]).groupby(sub["yyyymm"]).sum()

    if len(port_ret) < 12:
        results[char] = {"sharpe": np.nan, "mean": np.nan, "std": np.nan, "n_months": len(port_ret)}
        continue

    sr = annualized_sharpe(port_ret)
    results[char] = {
        "sharpe": sr,
        "mean": port_ret.mean() * 12,
        "std": port_ret.std() * np.sqrt(12),
        "n_months": len(port_ret),
    }

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(feature_cols)} characteristics...")

print(f"  Done: {len(feature_cols)} characteristics processed.")


# ── 4. Compile results ──────────────────────────────────────────────────────
res_df = pd.DataFrame(results).T
res_df.index.name = "characteristic"
res_df = res_df.sort_values("sharpe", ascending=False)

# Save for later comparison
res_df.to_csv("task_a_small_sharpe.csv")

print("\n" + "=" * 70)
print("TOP 5 CHARACTERISTICS - SMALL CAPS (highest annualized Sharpe)")
print("=" * 70)
for rank, (char, row) in enumerate(res_df.head(5).iterrows(), 1):
    print(f"  {rank}. {char:30s}  Sharpe={row['sharpe']:+.3f}  "
          f"Ann.Mean={row['mean']:+.4f}  Months={int(row['n_months'])}")

print("\n" + "=" * 70)
print("BOTTOM 5 CHARACTERISTICS - SMALL CAPS (lowest annualized Sharpe)")
print("=" * 70)
valid = res_df.dropna(subset=["sharpe"])
for rank, (char, row) in enumerate(valid.tail(5).iterrows(), 1):
    print(f"  {rank}. {char:30s}  Sharpe={row['sharpe']:+.3f}  "
          f"Ann.Mean={row['mean']:+.4f}  Months={int(row['n_months'])}")

print(f"\nSummary: {len(valid)} characteristics with valid Sharpe ratios")
print(f"  Sharpe range: [{valid['sharpe'].min():.3f}, {valid['sharpe'].max():.3f}]")
print(f"  Median Sharpe: {valid['sharpe'].median():.3f}")


# ── 5. Bar plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))
colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in res_df["sharpe"].values]
ax.bar(range(len(res_df)), res_df["sharpe"].values, color=colors, width=1.0, edgecolor="none")

ax.set_xlabel("Characteristic (sorted by Sharpe Ratio)", fontsize=12)
ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
ax.set_title("Small Caps: Rank-Weighted Long-Short Portfolio Sharpe Ratios", fontsize=14)
ax.axhline(y=0, color="black", linewidth=0.8)

# Annotate top 5 and bottom 5
top5 = res_df.head(5)
bot5 = valid.tail(5)
for idx, (char, row) in enumerate(top5.iterrows()):
    ax.annotate(char, (idx, row["sharpe"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, rotation=45, ha="left")
for idx_offset, (char, row) in enumerate(bot5.iterrows()):
    pos = len(res_df) - 5 + idx_offset
    ax.annotate(char, (pos, row["sharpe"]),
                textcoords="offset points", xytext=(5, -15),
                fontsize=7, rotation=45, ha="left")

ax.set_xlim(-1, len(res_df))
plt.tight_layout()
plt.savefig("task_a_small_sharpe_ratios.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to task_a_small_sharpe_ratios.png")


# ── 6. Compare with large-cap top/bottom 5 ──────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON: LARGE CAP vs. SMALL CAP - TOP 5")
print("=" * 70)

# Large-cap top 5 from task_a (full sample)
large_top5 = ["AnnouncementReturn", "High52", "retConglomerate", "dVolPut", "ChangeInRecommendation"]
small_top5 = list(res_df.head(5).index)

print(f"  Large-cap top 5: {large_top5}")
print(f"  Small-cap top 5: {small_top5}")
overlap = set(large_top5) & set(small_top5)
print(f"  Overlap:         {overlap if overlap else 'NONE'}")
