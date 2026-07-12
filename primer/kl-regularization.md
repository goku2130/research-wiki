---
title: KL regularization in RLHF
kind: primer
reference: ../topics/kl-regularization.md
updated: '2026-07-12'
---

# KL Regularization in RLHF: The Safety Tether That Keeps Policies Honest

**What this is.** KL regularization is the central guardrail in RLHF: a reverse-KL penalty that anchors the learned policy $\pi_\theta$ to a frozen reference model $\pi_{\text{ref}}$ (usually the SFT checkpoint). By the end of this primer you will understand *why* the penalty takes the form it does, how the Boltzmann optimum falls out of it, why per-token placement became standard, and—most critically—why the choice of KL *estimator* ($k_1$, $k_2$, $k_3$) and its placement (in-reward vs. as-loss) changes the actual gradient your optimizer sees. This connects directly to PPO, GRPO, DPO, and the newer token-level action-value methods (KLQ) and behavior-supported alternatives (BSPO).

---

## The Core Mechanism: A Boltzmann Anchor

The canonical RLHF objective maximizes expected reward minus a reverse-KL penalty:

$$
\mathcal{J}_{\mathrm{RLHF}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \bigl[ r(x,y) \bigr] - \beta \; \mathbb{D}_{\mathrm{KL}}\bigl( \pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x) \bigr), \qquad \beta > 0.
$$

For a *fixed* reward function, the pointwise optimal policy has a closed form—the **Boltzmann (Gibbs) distribution**:

$$
\pi^{*}(y|x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y|x) \, \exp\!\Bigl(\frac{1}{\beta} r(x,y)\Bigr), \quad Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\!\Bigl(\frac{1}{\beta} r(x,y)\Bigr).
$$

**Why this works.** The reference model $\pi_{\text{ref}}$ is a strong, general prior (the SFT model). The reward $r$ pulls probability mass toward high-reward completions; the KL term pulls back toward $\pi_{\text{ref}}$. The temperature $\beta$ controls the trade-off: small $\beta$ $\to$ sharp, reward-greedy policy; large $\beta$ $\to$ policy stays close to $\pi_{\text{ref}}$. This is the "safety tether" that prevents **over-optimization**—the phenomenon where a powerful optimizer finds high-reward but qualitatively broken outputs (repetition, language switching, plausible-but-wrong reasoning).

**Per-token decomposition.** The sequence-level KL decomposes exactly under the chain rule:

$$
\mathbb{D}_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}}) = \mathbb{E}_{y\sim\pi_\theta}\Bigl[ \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})} \Bigr].
$$

Per-token placement (now standard in PPO and GRPO) folds the penalty into the reward at each step, giving denser gradient signals and integrating naturally with GAE. The per-token contribution is simply $\text{KL}_t = \log \pi_\theta(y_t|\cdot) - \log \pi_{\text{ref}}(y_t|\cdot)$.

---

## The Estimator Zoo: Not All KLs Are Equal for Optimization

Liu et al. (2025) identified a critical design axis: whether the KL term sits **in the reward** (multiplied by the score function $\nabla_\theta \log \pi_\theta$) or **as a loss** (differentiated directly). Three per-token estimators of the reverse-KL divergence $\delta = \pi_\theta/\pi_{\text{ref}}$ are in common use:

| Estimator | Formula | Value estimation | Gradient-equivalent coefficient $c(y)$ |
|-----------|---------|------------------|----------------------------------------|
| $k_1$ | $-\log \delta$ | Unbiased | $\log \delta$ (in reward) / **0** (as loss) |
| $k_2$ | $\frac{1}{2}(\log \delta)^2$ | Biased | $\log \delta$ (as loss) |
| $k_3$ | $\delta - 1 - \log \delta$ | Unbiased, always $\ge 0$ | $1 - \delta$ (as loss) |

**The gradient-equivalence theorem (on-policy).** The true RKL objective gradient is

$$
\nabla_\theta \mathcal{J}_{\mathrm{RKL}} = \mathbb{E}_{x,y\sim\pi_\theta}\Bigl[ \bigl(\log \tfrac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\bigr) \nabla_\theta \log \pi_\theta(y|x) \Bigr].
$$

This **target gradient** is matched exactly by:
- $k_1$ **in reward** (coefficient $c = \log \delta$)
- $k_2$ **as loss** (gradient-equivalent coefficient $c = \log \delta$)

**The traps.**
- **$k_1$ as loss is vacuous.** Its gradient is $\mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$ (zero-mean score identity). It adds noise but *zero expected regularization signal*. GRPO with $k_1$-as-loss behaves like a no-KL baseline and fails to constrain drift.
- **$k_3$ as loss is a biased first-order approximation.** Its gradient-equivalent coefficient is $c_{3'} = 1 - \delta$, which approximates $-\log \delta$ only near $\delta \approx 1$. The bias expands as

$$
\mathrm{Bias}(\delta) = (-\log \delta) - (1 - \delta) = \tfrac{1}{2}(\delta-1)^2 - \tfrac{1}{3}(\delta-1)^3 + \cdots
$$

  This induces **pathological asymmetry**:
  - *Over-coverage* ($\delta \to 0$): $1-\delta \to 1$ (weak, saturating pull) vs. $-\log \delta \to +\infty$ (strong pull back).
  - *Under-coverage* ($\delta \to \infty$): $1-\delta \to -\infty$ (explosive push away) vs. $-\log \delta \to -\infty$ logarithmically.
  Moreover, $\mathrm{Var}[1-\delta] = \chi^2(\pi_{\text{ref}}\|\pi_\theta)$, which can be **infinite** if support conditions are violated. In a Gaussian illustration, $k_3$ exhibited sample std **8.8** vs. $k_1$'s **0.69**.

**Experimental verdict (GRPO, math reasoning, $\beta=0.5$).** $k_2$-as-loss gives the strongest regularization, lowest reward/length variance, tightest coupling to reference, and highest entropy (best exploration). $k_3$-as-loss shows weaker constraints, larger probability gaps, reduced entropy, and greater drift/instability despite transient reward gains.

> **Takeaway:** "Unbiased value estimation $\neq$ good gradients." For optimization, use $k_1$ *in reward* (PPO-style) or $k_2$ *as loss* (GRPO-style). Avoid $k_1$-as-loss and treat $k_3$-as-loss with caution.

---

## Runnable Check: The $k_3$ Bias and Variance Pathology

The following snippet demonstrates the $k_3$ gradient-equivalent coefficient's asymmetry and variance explosion compared to the true RKL coefficient $\log \delta$. It uses a simple discrete distribution where $\pi_{\text{ref}}$ has full support but $\pi_\theta$ can assign near-zero or huge mass.

```python
import numpy as np

def kl_coefficients(delta):
    """Return true RKL coeff (log delta), k3-as-loss coeff (1 - delta), and bias."""
    true_coeff = np.log(delta)
    k3_coeff = 1.0 - delta
    bias = true_coeff - k3_coeff
    return true_coeff, k3_coeff, bias

# Test asymmetry: over-coverage (delta << 1) and under-coverage (delta >> 1)
deltas = np.array([1e-4, 1e-2, 0.5, 1.0, 2.0, 100.0, 1e4])
print("delta     | true_coeff (log δ) | k3_coeff (1-δ) | bias")
print("----------|--------------------|----------------|-------")
for d in deltas:
    t, k, b = kl_coefficients(d)
    print(f"{d:>9.4g} | {t:>18.4f} | {k:>14.4f} | {b:>7.4f}")

# Variance explosion: sample delta from a distribution where pi_ref has mass but pi_theta can be tiny
np.random.seed(0)
# Simulate delta = pi_theta / pi_ref where pi_ref ~ Uniform(0.01, 0.1), pi_theta ~ Uniform(1e-6, 1.0)
pi_ref = np.random.uniform(0.01, 0.1, 10000)
pi_theta = np.random.uniform(1e-6, 1.0, 10000)
delta = pi_theta / pi_ref

true_coeff = np.log(delta)
k3_coeff = 1.0 - delta

print(f"\nTrue coeff (log δ):  mean={true_coeff.mean():.3f}, std={true_coeff.std():.3f}")
print(f"k3 coeff (1 - δ):    mean={k3_coeff.mean():.3f}, std={k3_coeff.std():.3f}")
print(f"Variance ratio (k3/true): {(k3_coeff.var() / true_coeff.var()):.1f}x")

# Assert the pathology: k3 variance >> true variance, and bias is large for extreme deltas
assert k3_coeff.var() > 10 * true_coeff.var(), "k3 variance should explode relative to true coeff"
assert abs(kl_coefficients(1e-4)[2]) > 5.0, "Bias at delta=1e-4 should be large (true_coeff ~ -9.2, k3_coeff ~ 1)"
assert abs(kl_coefficients(1e4)[2]) > 5000, "Bias at delta=1e4 should be huge (true_coeff ~ 9.2, k3_coeff ~ -1e4)"
print("\nAll assertions passed: k3-as-loss coefficient is biased and high-variance.")
```

**Output interpretation.** The table shows the sign-flip and magnitude mismatch: for $\delta=10^{-4}$, the true coefficient is $-9.2$ (strong pull back) while $k_3$ gives $+1.0$ (weak push *away*). For $\delta=10^4$, true is $+9.2$ (logarithmic push away) while $k_3$ gives $-10^4$ (explosive). The variance ratio confirms the $\chi^2$ explosion.

---

## Two Load-Bearing Disagreements

### 1. $k_3$ estimator quality: value estimation vs. gradient optimization
Early work (John, 2020) claimed $k_3$ is "strictly better" because it is unbiased and always non-negative as a *value estimator*. Liu et al. (2025) show this claim **fails for gradient optimization**: $k_3$-as-loss has pathological asymmetry, bias that grows quadratically away from $\delta=1$, and potentially infinite variance. The "strictly better" property holds for *estimating the KL value* but diverges completely when that estimator is differentiated. **$k_2$-as-loss dominates empirically** (stronger regularization, lower variance, better exploration), yet most public GRPO implementations still use $k_3$.

### 2. KL vs. behavior-supported regularization
Standard RLHF uses KL to the SFT reference as a universal anchor. BSPO (2025) argues KL is a **blunt instrument**: it penalizes *all* deviations, including high-quality out-of-distribution (OOD) responses that the reward model would actually rate well. Instead, BSPO learns the **behavior policy** $\beta(a|s)$ (the empirical next-token distribution in the reward-model training data) and only disadvantages actions with $\beta(a|s)=0$. In synthetic gold-reward setups, BSPO achieves higher true reward and maintains low unsupported-token counts where KL fails at smaller distances. The tension: KL is reference-model-dependent but simple; behavior support is data-dependent but requires accurate $\beta$ estimation (ScoreLM) and assumes RM training data covers desired behaviors.

---

## Current Status & Full Reference

**Current status:** Per-token KL-in-reward (PPO) and $k_3$-as-loss (GRPO) remain production workhorses; $k_2$-as-loss is gaining traction as the community recognizes gradient-equivalence matters more than value-estimation unbiasedness; multi-reference RLHF (exact geometric-mixture solution) shows 66% win-rate at 7B but is untested at frontier scale; KLQ matches PPO with cleaner foundations and higher judge win-rates; BSPO challenges KL's universality but assumes perfect behavior-policy estimation.

**Full reference:** See the comprehensive fact-checked article "KL regularization in RLHF" (sources: arxiv:2510.01555, arxiv:2502.01203, arxiv:2508.17000, arxiv:2503.18130, arxiv:2411.04625, arxiv:2305.18290, arxiv:2203.02155, arxiv:1909.08593, rlhfbook:regularization, huggingface:kl-regularization, openrlhf:kl-penalty, mbrenndoerfer:kl-divergence-penalty).

---
*Full reference (citations, derivations, variants):* [KL regularization in RLHF](../topics/kl-regularization.md)
