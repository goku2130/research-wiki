---
id: arxiv:2602.21765
type: paper
title: Generalisation of RLHF under Reward Shift and Clipped KL Regularisation
url: https://arxiv.org/abs/2602.21765
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Generalisation of RLHF under Reward Shift and Clipped KL Regularisation

### Core Problem
The generalisability of Reinforcement Learning from Human Feedback (RLHF) is often poorly understood, particularly regarding two practical implementation details:
1. **Reward Shift:** Reward models are typically trained on preference data from earlier or mixed behaviour policies, but the RLHF process optimises the current policy on its own rollouts. This shift can lead the policy into regions where the reward model is unreliable.
2. **Clipped KL Regularisation:** To stabilise training, the KL divergence is estimated from sampled log-probability ratios and clipped (truncated) to a threshold $\tau$. This introduces a systematic objective mismatch (bias) between the empirical surrogate and the target population objective.

### Method and Theoretical Framework
The authors develop a generalisation theory using a change-of-measure decomposition and PAC-Bayes tools to bound the discrepancy between the empirical objective $\widehat{J}_{n,K}^{\phi,\tau}(\theta)$ and the target population objective $J^{\star}(\theta)$.

**Step-by-Step Decomposition:**
The generalisation error is decomposed into three interpretable components:
1. **Sampling Error:** The noise induced by observing a finite number of prompts $n$ and a finite number of rollouts $K$ per prompt.
2. **Reward Shift Error:** The gap between the learned reward $\hat{r}_\phi$ and the target reward $r^*$, amplified by the distribution shift between the training distribution $D_{\text{train}}$ and the policy-induced distribution $D_\theta$.
3. **KL Clipping Error:** The bias introduced by using the clipped log-ratio $\ell_\theta^\tau$ instead of the exact log-ratio $\ell_\theta$.

### Key Formulas
The **Fixed-Policy Generalisation Bound** (Theorem 1) states that with probability at least $1 - \delta$:

$$
\left| \widehat{J}_{n, K}^{\phi, \tau}(\theta)-J^{\star}(\theta) \right| \leq (1+2 \beta \tau)\left(\sqrt{\frac{\log (4 / \delta)}{2 n}}+\sqrt{\frac{\log (4 / \delta)}{2 n K}}\right) + \mathcal{C}(\theta) \sqrt{L_{\text {train }}^{(2)}(\phi)} + \beta \mathbb{E}_{(X, Y) \sim D_{\theta}}\left[\left|\ell_{\theta}(X, Y)-\ell_{\theta}^{\tau}(X, Y)\right|\right]
$$

Where:
*   **Coverage Coefficient:** $\mathcal{C}(\theta) = \sqrt{1 + \chi^2(D_{\theta} \| D_{\text{train}})}$, quantifying the shift from training to deployment.
*   **Training Error:** $L_{\text{train}}^{(2)}(\phi) = \mathbb{E}_{(X,Y)\sim D_{\text{train}}}[(\hat{r}_{\phi}(X,Y)-r^{*}(X,Y))^{2}]$.
*   **KL Regularisation Strength:** $\beta$.

For data-dependent policy selection, the **PAC-Bayes Bound** (Theorem 2) replaces the fixed $\theta$ with an average over a posterior $Q$, adding a complexity term $\text{KL}(Q \| P)$ to the sampling error.

### Key Quantitative Results and Practical Implications
The theory provides concrete guidelines for hyperparameter tuning and resource allocation:

**1. Optimal KL Clipping Threshold ($\tau^*$):**
The threshold $\tau$ manages a bias-variance trade-off. The authors derive that $\tau^*$ should be the $(1 - 2\alpha_{n,K,\delta})$-quantile of the log-ratio magnitude $|\ell_\theta(X,Y)|$ under $D_\theta$, where:

$$
\alpha_{n,K,\delta} = \sqrt{\frac{\log(4/\delta)}{2n}} + \sqrt{\frac{\log(4/\delta)}{2nK}}
$$

This implies that as the evaluation budget ($n, K$) increases, $\tau$ should increase to reduce bias.

**2. Budget Allocation:**
*   **Uniform Cost:** If prompts and rollouts cost the same, the bound is minimised at $K^* = 1$ (prioritising prompt coverage).
*   **Prefill/Decode Cost:** Given prefill cost $c_{\text{prefill}}$ and decode cost $c_{\text{decode}}$, the optimal rollouts per prompt is:

$$
K^* = \max \left\{ 1, \left( \frac{c_{\text{prefill}}}{c_{\text{decode}}} \right)^{2/3} \right\}
$$

*   **Variance-Aware Allocation:** A refined rule considers prompt-level variance $\sigma^2_{\text{prompt}}$ and rollout-level variance $\sigma^2_{\text{rollout}}$:

$$
K^* \approx \max \left\{ 1, \sqrt{\frac{c_{\text{prefill}}}{c_{\text{decode}}} \cdot \frac{\sigma^2_{\text{rollout}}}{\sigma^2_{\text{prompt}}}} \right\}
$$

### Stated Limitations
The theoretical guarantees rely on several assumptions:
*   **Absolute Continuity:** $D_\theta$ must be absolutely continuous with respect to $D_{\text{train}}$, and the $\chi^2$ divergence must be finite.
*   **Boundedness:** The squared training error $L_{\text{train}}^{(2)}(\phi)$ must be bounded.
*   **Integrability:** The exact log ratio $\ell_\theta$ must be integrable in deployment ($\mathbb{E}[|\ell_\theta|] < \infty$).
*   **Heuristic Calibration:** The authors note that selecting $\tau$ from the evaluation sample is a "practical calibration heuristic" unless sample-splitting or further uniformity arguments are employed.
