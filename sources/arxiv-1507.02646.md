---
id: arxiv:1507.02646
type: paper
title: Pareto Smoothed Importance Sampling
url: https://arxiv.org/abs/1507.02646
retrieved: '2026-07-12'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# Pareto Smoothed Importance Sampling (PSIS)

### Core Problem
Importance sampling (IS) is used to estimate expectations $\text{E}_p(h) = \int h(\theta)p(\theta)d\theta$ using a proposal distribution $g(\theta)$. The self-normalized IS estimator is defined as:

$$
\hat{I}_h = \frac{\sum_{s=1}^{S}r_{s}h(\theta_{s})}{\sum_{s=1}^{S}r_{s}}, \quad r_{s} = \frac{p(\theta_{s})}{g(\theta_{s})}
$$

The stability of this estimator depends on the distribution of the importance ratios $r_s$. In high dimensions or when $g(\theta)$ is a poor approximation of $p(\theta)$, the ratios often exhibit heavy right tails. This can lead to infinite variance or a situation where the estimator is dominated by a few draws, making the results unstable and the variance-based error estimates unreliable, even if the ratios are technically bounded.

### Method/Recipe
Pareto Smoothed Importance Sampling (PSIS) stabilizes the estimator by replacing the largest importance ratios with expected order statistics derived from a Generalized Pareto Distribution (GPD).

**Step-by-Step Procedure:**
1. **Order Ratios:** Sort the raw importance ratios $r_s$ from lowest to highest.
2. **Determine Smoothing Subset:** Calculate the number of largest weights to be smoothed: $M = \lfloor \min(0.2S, 3\sqrt{S}) \rfloor$.
3. **Retain Bulk:** Set $w_s = r_s$ for $s = 1, \dots, S-M$.
4. **Define Threshold:** Set the cutpoint $\hat{u} = r_{S-M}$.
5. **Fit GPD:** Estimate the shape parameter $\hat{k}$ and scale parameter $\hat{\sigma}$ of the GPD using the $M$ largest ratios via the Zhang and Stephens (2009) approximate Bayesian method. The GPD density is:

$$
p(y \mid u, \sigma, k) = \frac{1}{\sigma} \left( 1 + k \left( \frac{y-u}{\sigma} \right) \right)^{-\frac{1}{k}-1} \quad (k \neq 0)
$$

6. **Smooth Tail:** Replace the $M$ largest weights with values consistent with the GPD tail:

$$
w_{S-M+z} = \min \left( F^{-1} \left( \frac{z-1/2}{M} \right), \max_s(r_s) \right) \quad \text{for } z = 1, \dots, M
$$

   where $F^{-1}$ is the inverse-CDF of the fitted GPD.
7. **Stability Check:** Report a warning if $\hat{k} > \min(1 - 1/\log_{10}(S), 0.7)$.

### Key Quantitative Results
The shape parameter $\hat{k}$ serves as a diagnostic for the reliability of the estimator:
*   **Reliability Thresholds:** For $S > 2000$, PSIS is generally reliable when $\hat{k} < 0.7$. If $\hat{k} > 0.7$, the bias dominates and variance-based Monte Carlo Standard Error (MCSE) estimates fail.
*   **Sample Size Heuristics:** The minimum sample size $S$ required to control RMSE is approximately $10^{1/(1-\hat{k})}$. The approximate effective sample size (ESS) is $\text{ESS}_{\hat{k}} \approx S / 10^{\hat{k}/(1-\hat{k})}$.
*   **Convergence Rates:** 
    *   For $\hat{k} < 0.5 - 0.5/\log_{10}(S)$, the convergence rate is $\sim S^{1/2}$ (standard CLT).
    *   For $\hat{k} > 0.5 + 0.5/\log_{10}(S)$, the rate slows to $\sim S^{1-k}$.
*   **Empirical Performance:** In high-dimensional tests (up to $D=1024$), $\hat{k}$ correctly diagnosed the collapse of ESS and convergence rates even when ratios were bounded, identifying cases where the sample size was insufficient to reach the asymptotic regime.

### Limitations
*   **High Shape Parameter:** When $\hat{k} > 0.7$, the estimator's convergence becomes too slow for practical use, and the bias becomes the dominant error source.
*   **Non-existent Mean:** If $\hat{k} > 1$, the mean of the ratio distribution is expected not to exist, rendering any estimate of the mean invalid.
*   **Assumptions:** The method relies on the working assumption that the tail of the importance ratio distribution lies in the domain of attraction of an extremal distribution (GPD).
*   **Small Sample Sizes:** For very small $S$, the reliability threshold for $\hat{k}$ is lower (e.g., $\hat{k} < 0.5$ for $S=100$).
