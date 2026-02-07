"""
Task (a): Rank-Weighted Long-Short Portfolios for Each Characteristic
=====================================================================
For each characteristic, cross-sectionally rank stocks each month,
form a rank-weighted long-short portfolio, and compute the annualized Sharpe ratio.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load and prepare data ─────────────────────────────────────────────────
df = pd.read_parquet("largeml.pq")
df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

# Identify feature columns (everything except permno, yyyymm, ret)
id_cols = ["permno", "yyyymm"]
target_col = "ret"
feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
print(f"Number of characteristics: {len(feature_cols)}")
print(f"Dataset shape: {df.shape}")


# ── 2. Annualized Sharpe Ratio function ──────────────────────────────────────
def annualized_sharpe(monthly_returns):
    """Annualized Sharpe Ratio = (mean / std) * sqrt(12), excess of zero."""
    mean_r = monthly_returns.mean()
    std_r = monthly_returns.std(ddof=1)
    if std_r == 0 or np.isnan(std_r):
        return np.nan
    return (mean_r / std_r) * np.sqrt(12)


# ── 3. Compute rank-weighted portfolio returns for each characteristic ───────
results = {}

for i, char in enumerate(feature_cols):
    # Keep rows where both the characteristic and return are non-NaN
    sub = df[["permno", "yyyymm", char, "ret"]].dropna(subset=[char, "ret"])

    if len(sub) == 0:
        results[char] = {"sharpe": np.nan, "mean": np.nan, "std": np.nan, "n_months": 0}
        continue

    # Cross-sectional rank each month (ascending: low char → low rank)
    sub["rank"] = sub.groupby("yyyymm")[char].rank(method="average")

    # Signal = rank - cross-sectional mean rank  →  zero-mean weights
    sub["signal"] = sub.groupby("yyyymm")["rank"].transform(lambda x: x - x.mean())

    # Normalize: weight = signal / sum(|signal|)  →  unit leverage
    sub["abs_sum"] = sub.groupby("yyyymm")["signal"].transform(lambda x: x.abs().sum())
    sub["weight"] = sub["signal"] / sub["abs_sum"]

    # Portfolio return each month
    port_ret = (sub["weight"] * sub["ret"]).groupby(sub["yyyymm"]).sum()

    # Require at least 12 months to compute a meaningful Sharpe
    if len(port_ret) < 12:
        results[char] = {"sharpe": np.nan, "mean": np.nan, "std": np.nan, "n_months": len(port_ret)}
        continue

    sr = annualized_sharpe(port_ret)
    results[char] = {
        "sharpe": sr,
        "mean": port_ret.mean() * 12,  # annualized mean
        "std": port_ret.std() * np.sqrt(12),  # annualized std
        "n_months": len(port_ret),
    }

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(feature_cols)} characteristics...")

print(f"  Done: {len(feature_cols)} characteristics processed.")


# ── 4. Compile results ───────────────────────────────────────────────────────
res_df = pd.DataFrame(results).T
res_df.index.name = "characteristic"
res_df = res_df.sort_values("sharpe", ascending=False)

print("\n" + "=" * 70)
print("TOP 5 CHARACTERISTICS (highest annualized Sharpe)")
print("=" * 70)
for rank, (char, row) in enumerate(res_df.head(5).iterrows(), 1):
    print(f"  {rank}. {char:30s}  Sharpe={row['sharpe']:+.3f}  "
          f"Ann.Mean={row['mean']:+.4f}  Months={int(row['n_months'])}")

print("\n" + "=" * 70)
print("BOTTOM 5 CHARACTERISTICS (lowest annualized Sharpe)")
print("=" * 70)
for rank, (char, row) in enumerate(res_df.tail(5).iterrows(), 1):
    print(f"  {rank}. {char:30s}  Sharpe={row['sharpe']:+.3f}  "
          f"Ann.Mean={row['mean']:+.4f}  Months={int(row['n_months'])}")

# Stats
valid = res_df.dropna(subset=["sharpe"])
print(f"\nSummary: {len(valid)} characteristics with valid Sharpe ratios")
print(f"  Sharpe range: [{valid['sharpe'].min():.3f}, {valid['sharpe'].max():.3f}]")
print(f"  Median Sharpe: {valid['sharpe'].median():.3f}")


# ── 5. Bar plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))

colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in res_df["sharpe"].values]
ax.bar(range(len(res_df)), res_df["sharpe"].values, color=colors, width=1.0, edgecolor="none")

ax.set_xlabel("Characteristic (sorted by Sharpe Ratio)", fontsize=12)
ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
ax.set_title("Rank-Weighted Long-Short Portfolio: Annualized Sharpe Ratio by Characteristic", fontsize=14)
ax.axhline(y=0, color="black", linewidth=0.8)

# Label top 5 and bottom 5
top5 = res_df.head(5)
bot5 = res_df.tail(5)
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
plt.savefig("task_a_sharpe_ratios.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to task_a_sharpe_ratios.png")
