---
id: arxiv:2604.02986
type: paper
title: Mitigating Reward Hacking in RLHF via Advantage Sign Robustness
url: https://arxiv.org/abs/2604.02986
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-hacking
---

# Mitigating Reward Hacking in RLHF via Advantage Sign Robustness

### Core Problem
In Reinforcement Learning from Human Feedback (RLHF), policies often suffer from **reward hacking**, where the model maximizes a learned proxy reward model (RM) while true quality plateaus or degrades. The authors hypothesize that reward hacking is primarily caused by **flipped advantage signs**: when a proxy RM incorrectly predicts the sign of a completion's advantage, the policy gradient update increases the likelihood of a poor response instead of decreasing it.

### Method: Sign-Certified Policy Optimization (SignCert-PO)
SignCert-PO is an entirely post-hoc method that introduces robustness at the policy optimization stage without retraining the RM or requiring access to the original RM training data. It identifies and down-weights completions whose advantage signs are "fragile" under RM parameter perturbations.

**Step-by-Step Recipe:**
1. **Sampling:** For a given prompt $x$, sample a group of $K$ completions $\{y^{(1)}, \dots, y^{(K)}\}$ from the current policy $\pi_\phi$.
2. **Advantage Estimation:** Compute the group-relative advantage for each completion $j$ using the Dr.GRPO formulation:

$$
A_{j}(\theta) = r_{\theta}(x,y^{(j)}) - \frac{1}{K}\sum_{k=1}^{K}r_{\theta}(x,y^{(k)})
$$

3. **Radius Calculation:** To avoid the computational cost of full-parameter gradients, the authors assume a linear RM head $r_\theta(x, y) = w^\top h_\psi(x, y) + b$. They calculate the **certified sign-preservation radius** $\Delta_j$, which is the smallest perturbation to the head parameters $w$ that would flip the sign of $A_j$:

$$
\Delta_{j} = \frac{|A_{j}(w)|}{\|h_{\psi}(x,y^{(j)}) - \bar{h}\|_{2}}, \quad \text{where } \bar{h} = \frac{1}{K}\sum_{k=1}^{K}h_{\psi}(x,y^{(k)})
$$

4. **Adaptive Budgeting:** Set a perturbation budget $\epsilon$ as the $q_t$-th quantile of $\{1/\Delta_j\}$ across all completions in the current batch.
5. **Re-weighting:** Compute a robustness coefficient $\rho_j^*$ for each completion:

$$
\rho_{j}^{*} = 1 - \frac{\epsilon}{\Delta_{j}}
$$

6. **Policy Update:** Update the policy using the re-weighted policy gradient:

$$
\widehat{\nabla}_{\phi}J(\phi,\theta) = \mathbb{E}_{x}\left[\sum_{j=1}^{K}\rho_{j}^{*} \cdot A_{j}(w) \cdot \nabla_{\phi}\log\pi_{\phi}(y^{(j)}|x)\right]
$$

### Key Quantitative Results
SignCert-PO was evaluated on TL;DR summarization and AlpacaFarm benchmarks using Pythia and Qwen2.5 models.

*   **Win Rates (Gold RM):**
    *   **TL;DR (Pythia 1B):** SignCert-PO achieved a **60.0%** win rate, significantly outperforming Dr.GRPO (**21.0%**) and AdvPO (**47.0%**).
    *   **TL;DR (Qwen2.5 3B):** SignCert-PO achieved **91.8%**, compared to Dr.GRPO's **90.2%**.
    *   **AlpacaFarm (Qwen2.5 3B):** SignCert-PO achieved **52.3%**, outperforming Dr.GRPO (**46.9%**) and BSPO (**43.8%**).
*   **Efficiency:** The method adds negligible computational overhead. On Pythia 1B, the wall-clock time per step was **7.47s** for SignCert-PO versus **7.54s** for Dr.GRPO.
*   **Data Sensitivity:** The largest gains were observed when preference data was limited (e.g., 1 epoch of training), suggesting SignCert-PO is highly effective when the proxy RM is weak.
*   **Robustness:** The authors found a Spearman rank correlation of **0.72** between $\Delta_j$ and sign preservation under whole-model perturbations, indicating the linear-head approximation is a valid proxy for general RM sensitivity.

### Stated Limitations
*   **Simplifying Assumptions:** The theoretical derivation relies on a linear head perturbation model.
*   **Conservatism:** The per-sample adversary formulation is more conservative than a shared-adversary variant (where one perturbation is shared across the group).
*   **Generalization:** Extending the certified sign-preservation radius to general, non-linear RMs without incurring the cost of full-parameter gradient norm computation remains an open problem.
