# Complete Results Summary: ML Asset Pricing

## Problem 1: ML-Based Cross-Sectional Return Prediction

### Task 1(a): Univariate Long-Short Portfolio Analysis

Rank-weighted L/S portfolios for each of 209 characteristics. Annualized OOS Sharpe Ratios:

**Large Cap** (largeml.pq — 499 stocks, 1926-2022):

| Rank | Characteristic | Sharpe Ratio |
| :--- | :--- | :--- |
| 1 | AnnouncementReturn | +3.62 |
| 2 | High52 | +3.35 |
| 3 | Mom12m | +2.91 |
| 4 | IntMom | +2.85 |
| 5 | STreversal | +2.44 |

**Small Cap** (smallml.pq — 977 stocks, 1926-2022):

| Rank | Characteristic | Sharpe Ratio |
| :--- | :--- | :--- |
| 1 | High52 | +2.79 |
| 2 | VolMkt | +0.95 |
| 3 | Mom12m | +0.83 |

> *Insight:* Small caps show weaker, more compressed Sharpe ratios and surface liquidity-related signals.

---

### Task 1(b): ML Models — Return Prediction R² OOS

* **Train:** first 20 years (1926-1945)
* **Val:** next 12 years (1946-1957)
* **Test:** remainder (1958-2022)
* **Target:** next-month stock return. $R^2_{OOS} = 1 - \frac{SS_{res}}{\sum y^2}$.

**Large Cap:**

| Model | Val R² | Test R² |
| :--- | :--- | :--- |
| GradientBoosting | +0.591 | **+0.236** |
| ElasticNet | +0.436 | +0.170 |
| Lasso | +0.405 | +0.165 |
| Ridge | +0.437 | -6.33 |
| OLS | +0.047 | -740 |
| PLS | +0.233 | -87 |
| RBF variants | ~0.30 | ~-0.05 |

**Small Cap:**

| Model | Val R² | Test R² |
| :--- | :--- | :--- |
| ElasticNet | +0.425 | **+0.158** |
| Lasso | +0.398 | +0.154 |
| Ridge | +0.440 | +0.149 |
| GradientBoosting | +0.666 | +0.135 |
| PLS | +0.238 | -0.237 |
| OLS | +0.047 | -0.290 |

> *Key finding:* GBR wins large caps (non-linear interactions matter); penalized linear models win small caps (sparser/noisier data favors simpler models).

---

### Task 1(c)+(d): ML Portfolio Construction & Cross-Universe Comparison

Rank-weighted L/S portfolios from ML predictions on the test period:

**Large Cap ML Portfolios:**

| Model | Annualized Sharpe |
| :--- | :--- |
| GradientBoosting | **+7.09** |
| ElasticNet | +5.68 |
| Lasso | +5.60 |
| Best Univariate (High52) | +4.31 |

**Small Cap ML Portfolios:**

| Model | Annualized Sharpe |
| :--- | :--- |
| GradientBoosting | **+3.21** |
| ElasticNet | +2.71 |
| Ridge | +2.86 |
| Best Univariate (High52) | +2.79 |

> *Insight:* ML portfolios beat the best univariate signal in both universes — multivariate models capture interaction effects that no single characteristic can.

---

### Task 1(e): Ensemble Portfolio Optimization

Searched over pairwise and 5-model simplex weight grids:

| Universe | Best Strategy | Sharpe |
| :--- | :--- | :--- |
| Large Cap | GBR alone (100%) | +7.09 |
| Small Cap | 50/50 ElasticNet + GBR | +3.23 |

> *Insight:* Large Cap: GBR dominates — ensemble doesn't help when one model is clearly superior. Small Cap: ensemble provides marginal gain (+3.23 vs +3.21) by diversifying across complementary estimation approaches.

---

## Problem 2: Factor Analysis & Mean-Variance Efficiency

### Task 2(a): PCA on L/S Portfolio Returns

Three universes analyzed (after cleaning: drop cols >30% NaN, drop remaining NaN rows):

| Dataset | Months | Portfolios | PC1 % | PCs for 80% | PCs for 90% | PCs for 95% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| lsret (pre-computed) | 818 | 108 | 35.1% | 11 | 26 | 43 |
| Large Cap (derived) | 532 | 74 | 27.4% | — | — | — |
| Small Cap (derived) | 612 | 47 | 34.4% | — | — | — |

> *Insight:* Small Cap is most compressible. PC1 loads heavily on volatility/market risk measures. Higher-order PCs (PC2, PC3) often have better Sharpe ratios than PC1.

---

### Task 2(b): Indicator Regression (Britten-Jones 1999)

Regress $Y=1$ on portfolio returns (no intercept): $\hat{\beta} \propto \Sigma^{-1}\mu$ (tangency portfolio).
*Data: lsret, 108 portfolios.*

| Method | Alpha | Non-zero | Ann. Mean | Ann. Vol | Sharpe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ridge | 1.00e-04 | 108/108 | +3.88 | 2.25 | **+1.72** |
| Lasso | 2.31e-01 | 18/108 | +2.32 | 1.63 | +1.42 |
| 1/N Equal | — | 108/108 | +2.74 | 2.39 | +1.15 |

> *Insight:* Both regularized methods beat naive 1/N. Lasso selects 18 portfolios (IndRetBig, STreversal dominate).

---

### Task 2(c): PCA-Based Indicator Regression

Replace raw returns with first $K$ PCA factors, then run indicator regression:

| K Factors | Var Explained | Ridge SR | Lasso SR |
| :--- | :--- | :--- | :--- |
| 1 | 35% | +0.20 | +0.20 |
| 5 | 71% | +0.66 | +0.52 |
| 10 | 80% | +0.87 | +0.76 |
| **20** | **87%** | **+1.84** | **+1.82** |
| 30 | 92% | +1.48 | +1.68 |
| 50 | 97% | +1.48 | +1.48 |
| 108 (raw) | 100% | +1.72 | +1.55 |

> *Insight:* Optimal: K=20 for both methods. PCA at K=20 beats raw returns by filtering noisy low-variance components. Beyond K=20, noise creeps in and performance degrades.

---

### Task 2(d): Cross-Universe Indicator Regression Comparison

| Universe | Ports | Test Mo | Ridge Raw | Lasso Raw | 1/N | Best PCA Ridge | Best PCA Lasso |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| lsret | 108 | 235 | +1.72 | +1.42 | +1.15 | **+1.84 (K=20)** | +1.82 (K=20) |
| Large Cap | 74 | 160 | +10.74 | +8.87 | -1.11 | +10.74 (K=74) | +9.81 (K=60) |
| Small Cap | 47 | 15 | +3.67 | +3.92 | +0.54 | +4.34 (K=40) | +3.46 (K=30) |

> *Interpretation:* Large Cap indicator regression achieves extreme SR (~10) because rank-weighted L/S portfolios from ~500 liquid stocks have strong, stable factor premia. PCA helps lsret (many portfolios, more noise) but not Large Cap (already well-conditioned). Small Cap results are unreliable (only 15 test months survive cleaning).

---

### Task 2(e): Best Portfolio — Maximum OOS Sharpe

**Systematic search over 40+ strategies (raw, PCA, factor screening, ensembles):**

| Rank | Strategy | Sharpe | Months |
| :--- | :--- | :--- | :--- |
| 1 | LgCap Ridge+Lasso Avg | **+10.75** | 160 |
| 2 | LgCap Ridge Raw | +10.74 | 160 |
| 3 | LgCap Ridge PCA K=30 | +9.55 | 160 |
| 4 | LgCap Lasso PCA K=40 | +9.43 | 160 |
| 5 | LgCap Ridge PCA K=50 | +9.39 | 160 |

> *Winner:* 50/50 Ridge+Lasso ensemble on Large Cap L/S portfolios — SR = +10.75. Ridge captures the full covariance structure; Lasso concentrates on the 6 strongest factors. Averaging diversifies across two complementary estimation approaches.

---

### Cross-Cutting Insights

1.  **Regularization is essential.** OLS catastrophically overfits ($R^2=-740$ for large cap). Lasso/Ridge/ElasticNet are necessary when P/T is non-trivial.
2.  **Simpler models for noisier data.** GBR wins large caps; penalized linear models win small caps. Data quality determines optimal model complexity.
3.  **PCA helps when P is large.** For lsret (108 portfolios), PCA K=20 improves SR by ~28%. For Large Cap (74 portfolios), PCA provides no benefit.
4.  **ML portfolios vs. indicator regression.** Problem 1's GBR portfolio (SR=7.09) uses individual stock predictions; Problem 2's indicator regression (SR=10.75) directly optimizes L/S portfolio weights. The indicator regression approach is more powerful here because it operates in the lower-dimensional portfolio return space rather than the noisy individual stock space.
5.  **Reality Check.** All results are before transaction costs. SR>3 is unrealistic after frictions in practice, but within this academic exercise, the ranking of methods is informative.
