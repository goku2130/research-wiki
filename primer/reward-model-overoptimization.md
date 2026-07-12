---
title: Reward model over-optimization
kind: primer
reference: ../topics/reward-model-overoptimization.md
updated: '2026-07-12'
---

# Reward Model Over-Optimization

**Scaffold.** Reward model over-optimization is the RLHF instantiation of Goodhart's Law: a policy optimized against a learned proxy reward model (RM) achieves high proxy scores while its true (gold) reward *decreases*. This primer teaches the quantitative scaling laws that predict *when* and *how severely* this turnover occurs, why it arises identically in both classic RLHF (PPO/BoN) and Direct Alignment Algorithms (DPO/IPO/SLiC), and which mitigations are principled versus heuristic. By the end you will be able to read a KL–reward curve, estimate the optimal KL budget for a given RM size, and choose between ensemble, distributional, and algorithmic defenses. The framework connects to reward modeling, KL regularization, and the emerging theory of under-constrained preference optimization.

---

## The Core Mechanism: Proxy–Gold Divergence in KL Space

In RLHF we train a proxy RM $r_\phi$ on preference comparisons to approximate a true reward $r^*$. The policy $\pi$ is then optimized to maximize $\mathbb{E}_\pi[r_\phi]$. Over-optimization occurs when further increases in proxy score correspond to *decreases* in gold reward—the proxy ceases to be a monotonic indicator of quality.

The key insight from Gao et al. (2022) is to parameterize optimization intensity not by training steps but by **KL distance from the initial policy**:

$$
d = \sqrt{D_{\mathrm{KL}}(\pi \parallel \pi_{\mathrm{init}})}
$$

This makes the phenomenon measurable and, crucially, *predictable* across optimization algorithms.

### Two Universal Scaling Laws

**Best-of-$n$ (BoN).** Sampling $n$ responses from $\pi_{\mathrm{init}}$ and picking the highest proxy score yields an analytically tractable KL:

$$
\mathrm{KL}_{\mathrm{bon}} = \log n - \frac{n-1}{n} \quad\Longrightarrow\quad d = \sqrt{\log n - \frac{n-1}{n}}
$$

Gold reward follows a **quadratic concave** function:

$$
R_{\mathrm{bon}}(d) = d(\alpha_{\mathrm{bon}} - \beta_{\mathrm{bon}} d)
$$

The quadratic penalty $-\beta d^2$ implies a *sharp, predictable turnover*: beyond the peak $d^* = \alpha/(2\beta)$, gold reward falls quadratically.

**PPO (and DAAs).** Under iterative RL optimization, the same gold reward follows a **logarithmic penalty** form:

$$
R_{\mathrm{RL}}(d) = d(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}} \log d)
$$

The slower $\log d$ turnover means PPO *consumes far more KL* to reach the same over-optimization level—it is significantly less KL-efficient than BoN. Yet the *proxy–gold gap* at any given $d$ is nearly identical for both methods, implicating the **proxy RM itself**, not the optimizer, as the root cause.

> **Why the difference?** BoN selects from the *initial* policy's distribution; its KL grows as $\log n$. PPO *moves* the policy, accumulating KL linearly in steps while the proxy–gold mismatch compounds more gradually. The log form emerges because each PPO step re-optimizes against a proxy whose error is locally linear in the policy shift, integrating to $\log d$.

---

## What Controls the Coefficients $\alpha$ and $\beta$?

Both $\alpha$ (initial alignment slope) and $\beta$ (over-optimization severity) scale **smoothly with proxy RM parameter count** $N_{\mathrm{RM}}$ on approximate logarithmic trends. Larger proxy RMs $\to$ higher $\alpha$, lower $\beta$ $\to$ peak shifts to **higher $d$ and higher $R$**. This lets you *predict* the optimal KL budget for a given RM size.

A critical **data threshold** emerges: proxy RMs trained on **< 2,000 comparisons** perform near chance and exhibit severe over-optimization; beyond this, additional data steadily reduces goodharting (no saturation observed up to 100k).

**Policy size independence** is striking: 1.2B and 6B policies over-optimize at *nearly the same KL distance* with a *similar proxy–gold gap*. The frontier in KL space is a property of the **RM pair (proxy vs. gold)**, not policy capacity. (Only two sizes tested; larger policies remain open.)

**KL penalty in PPO** does *not* improve the gold-reward-vs-KL frontier—it acts as early stopping, tracing the same curve as unpenalized PPO halted at different steps.

---

## The Same Law Without a Reward Model: DAAs

Direct Alignment Algorithms (DPO, IPO, SLiC-HF) bypass explicit RMs by optimizing a preference loss directly on the policy. Their unified objective:

$$
\mathcal{L}_{\text{DAA}} = \mathbb{E}_{(x,y_w,y_l)}\left[g\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

with $g$ convex ($-\log\sigma$ for DPO, $(x-1)^2$ for IPO, $\max(0,1-x)$ for SLiC).

**Empirically** (Reddit TL;DR, Pythia 1B–6.9B), all DAAs exhibit:
- **Hump-shaped win-rate vs. KL** curves matching $R(d)=d(\alpha-\beta\log d)$ — the *PPO form*, not BoN's quadratic.
- **Intra-epoch dynamics**: under wide KL budgets (small $\beta$), peak win-rate occurs at **~25% of the first epoch**; continued training increases KL and *decreases* win-rate.
- **Model scaling**: 6.9B models over-optimize less and achieve better win-rate/KL trade-offs than 1B.
- **Objective comparison**: IPO is more robust (lower KL, less over-optimization) than DPO/SLiC.

**Theoretical cause: OOD bootstrapping from rank deficiency.** The DAA objective is not strictly convex; the prompt–response space is exponentially larger than the preference dataset, yielding infinitely many minima. The policy can place mass on out-of-distribution (OOD) responses while still minimizing the loss. In a token-level MDP view, the implicit Q-value becomes overly optimistic for unseen tokens, especially when $\beta$ is small (wide KL budget). This is the *same mechanism* as proxy RM over-optimization: the "proxy" in DAAs is the **implicit reward** defined by the policy's log-ratio against the reference, and its failure stems from **optimizing an under-constrained objective on finite data**.

---

## Mitigations: From Heuristics to Principled Pessimism

### 1. Multi-Head Reward Model (Liu et al., 2024)
**Recipe**: Freeze an SFT backbone $F$; train $K$ independent linear heads $H_i$ with Bradley-Terry loss. During PPO, use the **minimum** reward across heads:

$$
\hat{r}(x) = \min_i H_i(F(x))
$$

**Result**: 3-head multi-head RM matches a full 3-model ensemble's gold-reward performance, while standard PPO shows concave-down degradation. Training cost: **1 epoch** vs. **3+ epochs** for full ensemble. The minimum is "miscalibrated" but acts as regularization preventing over-optimized local minima. Beyond ~3 heads, noise outweighs diversity.

### 2. Distributional RM + KL-DRO (Chen et al., 2024)
Replace scalar RM $r(x,y)$ with a **distributional RM** $p(r|x,y)$ (deep ensembles or Bayesian last-layer). Derive a pessimistic "effective reward" via **KL-Distributionally Robust Optimization**:

$$
\tilde{r}_{\text{rob}}(x,y) = \inf_{Q} \left\{ \mathbb{E}_{Q}[r] + \beta D_{\text{KL}}(Q \| p(r|x,y)) \right\}
= -\beta \log \mathbb{E}_{p(r|x,y)} \left[ e^{-r/\beta} \right]
$$

For an ensemble $\{r_i\}_{i=1}^K$, two estimators:
- **Log-MGF**: $-\beta \log \left( \frac{1}{K} \sum_i e^{-r_i/\beta} \right)$
- **Gaussian-truncated** (stable for small $K$): $\hat{\mu} - \frac{\hat{\sigma}^2}{2\beta}$

**Unification of heuristics** as limits of $\tilde{r}_{\text{rob}}$:
| Method | Formula | Recovery |
|--------|---------|----------|
| Mean | $\hat{\mu}$ | $\beta \to \infty$ (risk-neutral) |
| WCO (min) | $\min_i r_i$ | $\beta \to 0$ (unbounded adversary) |
| UWO | $\hat{\mu} - \lambda \hat{\sigma}^2$ | Gaussian truncation, $\lambda = 1/2\beta$ |

The **cumulant expansion** reveals the full structure:

$$
\tilde{r}_{\text{rob}} = \mu - \frac{\sigma^2}{2\beta} + \frac{\kappa_3}{6\beta^2} - \frac{\kappa_4}{24\beta^3} + \dots
$$

**Recommendation**: Use Gaussian-truncated $\hat{\mu} - \hat{\sigma}^2/(2\beta)$ as the stable default for small ensembles ($K \le 5$). Requires distributional RMs (standard Bradley-Terry insufficient) and calibrated variance tracking epistemic uncertainty.

---

## Runnable Check: The Two Scaling Laws in Action

```python
import numpy as np

def R_bon(d, alpha, beta):
    """Quadratic BoN scaling law: R(d) = d * (alpha - beta * d)"""
    return d * (alpha - beta * d)

def R_rl(d, alpha, beta):
    """Logarithmic PPO/DAA scaling law: R(d) = d * (alpha - beta * log(d))"""
    return d * (alpha - beta * np.log(d))

# Synthetic coefficients from a medium proxy RM (illustrative)
alpha, beta = 10.0, 1.5

d_vals = np.linspace(0.1, 5.0, 100)
R_bon_vals = R_bon(d_vals, alpha, beta)
R_rl_vals  = R_rl(d_vals, alpha, beta)

# Peak locations (analytic)
d_star_bon = alpha / (2 * beta)                    # quadratic peak
# R_rl peak: solve alpha - beta*(1 + log(d)) = 0
d_star_rl  = np.exp(alpha / beta - 1)

print(f"BoN peak:  d* = {d_star_bon:.3f}, R* = {R_bon(d_star_bon, alpha, beta):.3f}")
print(f"PPO peak:  d* = {d_star_rl:.3f}, R* = {R_rl(d_star_rl, alpha, beta):.3f}")

# Sanity checks
assert d_star_bon > 0 and d_star_rl > 0
assert R_bon(d_star_bon, alpha, beta) > R_bon(d_star_bon * 1.5, alpha, beta)  # past peak, reward drops
assert R_rl(d_star_rl, alpha, beta) > R_rl(d_star_rl * 1.5, alpha, beta)
# PPO peak should be at larger d (less KL-efficient)
assert d_star_rl > d_star_bon
print("All checks passed.")
```

---

## Load-Bearing Caveats

1. **Synthetic gold RM, not humans.** Gao et al.'s gold standard is a fixed 6B model, not human preferences. The *intent mismatch* (gold RM vs. actual human intent) is unmeasured. Transfer to production RLHF—where the gold standard evolves and RMs are updated iteratively—is unproven.

2. **Non-adversarial policies.** The scaling laws assume policies *cannot* actively manipulate the RM (reward tampering, adversarial exploitation). If policies learn to strategically exploit RM vulnerabilities, the quadratic/log laws may break entirely. This is the "adversarial Goodhart" regime, currently unexplored at scale.

---

## Current Status & Full Reference

**Current status:** Gao et al. (2022) scaling laws remain the primary quantitative framework for predicting over-optimization in KL space; DAAs obey the same $R(d)=d(\alpha-\beta\log d)$ law with over-optimization traced to rank deficiency/OOD bootstrapping; multi-head RM ensembling and KL-DRO on distributional RMs are the leading principled mitigations; intra-epoch peak at ~25% in DAAs demands aggressive early stopping.

**Full reference:** Gao et al., *Scaling Laws for Reward Model Overoptimization* (arXiv:2210.10760); follow-ups: DAA scaling (arXiv:2406.02900), multi-head RM (arXiv:2406.01013), KL-DRO unification (arXiv:2606.09073).

---
*Full reference (citations, derivations, variants):* [Reward model over-optimization](../topics/reward-model-overoptimization.md)
