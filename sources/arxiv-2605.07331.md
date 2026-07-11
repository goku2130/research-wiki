---
id: arxiv:2605.07331
type: paper
title: 'Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token
  Perspective'
url: https://arxiv.org/html/2605.07331v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

**Core Problem**
Off-policy policy gradient estimation in large language model post-training relies on importance sampling (IS) to correct distribution mismatch between the behavior policy $\pi_b$ and the target policy $\pi_\theta$. Existing algorithms face a fundamental bias-variance dilemma: token-level IS ratios (e.g., GRPO, PPO) ignore prefix state distribution mismatch, introducing systematic bias into gradient estimates, while full-sequence ratios provide exact trajectory-level correction but suffer from high variance due to the multiplicative accumulation of per-token ratios. Length-normalized sequence ratios (e.g., GSPO) improve numerical stability but deviate from exact IS correction, remaining biased.

**Methodology & Recipe**
The authors propose Cumulative Token Policy Optimization (CTPO), which resolves this dilemma through a theoretically grounded IS formulation and adaptive regularization. The optimization recipe proceeds as follows: (1) Compute the cumulative token IS ratio at each generation position $t$ by multiplying per-token ratios up to $t$, ensuring exact prefix correction for each token-level gradient term. (2) Recognize that the log-cumulative ratio accumulates variance linearly with position, causing fixed clipping ranges to disproportionately suppress late-token gradients. (3) Implement position-adaptive clipping by scaling log-space clip bounds proportionally to $\sqrt{t}$, matching the standard deviation growth of the cumulative log-ratio and yielding consistent regularization across positions. (4) Integrate this ratio and clipping into a group-relative policy optimization framework that estimates advantages via outcome-level reward normalization without a separate critic network.

**Key Formulas**
The cumulative token IS ratio is defined as $\rho_t^{\text{cum}} = \prod_{t'=1}^t r_{t'}$, where $r_{t'} = \pi_\theta(a_{t'} \mid s_{t'}) / \pi_b(a_{t'} \mid s_{t'}))$. Under an independence assumption across positions, its variance is $\text{Var}(\rho_t^{\text{cum}}) = \prod_{t'=1}^t (1 + \chi_{t'}^2) - 1$, which is strictly lower than the full-sequence variance. The log-space variance grows linearly: $\text{Var}(\log \rho_t^{\text{cum}}) = t\sigma^2$. Position-adaptive clipping thresholds scale as $\varepsilon_{\text{high}}(t) = \varepsilon_{\text{high}} \cdot t^{0.5}$ and $\varepsilon_{\text{low}}(t) = \varepsilon_{\text{low}} \cdot t^{0.5}$, defining the trust region $\rho_t^{\text{cum}} \in [e^{-\varepsilon_{\text{low}}(t)}, e^{\varepsilon_{\text{high}}(t)}]$. The final CTPO objective is:

$$
\mathcal{J}_{\text{CTPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \rho_{i,t}^{\text{cum}} A_i, \text{clip} \left( \rho_{i,t}^{\text{cum}}, e^{-\varepsilon_{\text{low}}(t)}, e^{\varepsilon_{\text{high}}(t)} \right) A_i \right) \right]
$$

**Quantitative Results**
Evaluated in tool-integrated reasoning (TIR) on Qwen3-4B and Qwen3-14B across AIME 25/26, BRUMO 25, and HMMT 25 benchmarks (avg@32), CTPO achieves the highest average performance. On Qwen3-4B, CTPO reaches 51.4, outperforming GSPO (47.7) by 3.7 points (+7.8% relative). On Qwen3-14B, CTPO achieves 58.8, surpassing GSPO (55.5) by 3.3 points (+5.9%). Ablation studies confirm position-adaptive clipping contributes 3.1 average points over fixed clipping (58.8 vs. 55.7). Empirical analysis shows fixed clipping yields a monotonically increasing clip rate up to ~20% for late tokens, whereas adaptive clipping maintains a uniform 5–10% rate across positions.

**Limitations**
The methodology is currently validated exclusively within the tool-integrated reasoning setting, constrained to trajectories of up to five interaction turns and a maximum of 8,000 tokens per response. The authors explicitly note that extending CTPO to broader agentic environments requiring longer-horizon interactions and richer environmental feedback remains a direction for future work. Additionally, while the strict variance reduction holds generally, the exact variance decomposition formula relies on an independence assumption across token positions.
