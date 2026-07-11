---
id: arxiv:2510.01555
type: paper
title: 'Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization'
url: https://arxiv.org/abs/2510.01555
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

The paper "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization" by Liu et al. addresses the implementation of Kullback-Leibler (KL) divergence regularization in Reinforcement Learning from Human Feedback (RLHF).

**Core Problem:**
The core problem identified is that current RLHF methods, particularly those using KL regularization (e.g., GRPO), often design and evaluate KL terms based on principles of numerical value estimation rather than their functional role as optimization losses. This misplaced emphasis can lead to ineffective or unstable gradient signals, even if the value estimator itself is considered "unbiased." Specifically, the paper challenges the assumption that good value estimation properties translate to effective gradients for policy optimization.

**Method/Recipe Step by Step:**

1.  **Establish a Unified Framework:** The authors propose a framework that connects two styles of KL implementation:
    *   **`k_n` in reward:** The KL term `k_n` acts as a detached coefficient for the policy's score function.
    *   **`k_n` as loss:** The KL term `k_n` is used as a direct loss function through which gradients are propagated.
    This framework demonstrates that `k_n` as loss can always be analyzed via an equivalent gradient coefficient in the `k_n` in reward style, unifying the perspectives.

2.  **Analyze `k_1` as loss (Counterexample):**
    *   The term `k_1(x) = -log(delta(x))` where `delta(x) = p(x)/q(x)` is an unbiased estimator of KL divergence.
    *   When used "as loss," its gradient is derived.
    *   **Result:** The gradient `∇_θ J_{k_1 as loss}(θ) = E_{x~D, y~π_θ(·|x)} [∇_θ log π_θ(y|x)]`.
    *   **Finding:** This gradient is entirely independent of the reference policy `π_ref` and has an expectation of exactly zero due to the zero-mean score identity. This means it provides no KL regularization signal, only noise.

3.  **Derive Principled RKL Loss and Identify Equivalences:**
    *   The exact on-policy gradient of the Reverse KL (RKL) objective `J_{RKL}(θ) = E_{x~D} [D_{KL}(π_θ(·|x) || π_{ref}(·|x))]` is derived.
    *   **Resulting Target Gradient:** `∇_θ J_{RKL}(θ) = E_{x~D, y~π_θ(·|x)} [ (log(π_θ(y|x)/π_{ref}(y|x))) ∇_θ log π_θ(y|x) ]`.
    *   **Theorem 5.1 (On-policy gradient equivalence):** Under on-policy conditions, two objectives have the same expected gradient as the target RKL gradient:
        *   `J_{k_1 in reward}(θ) = E_{x~D, y~π_θ(·|x)} [ (log(π_θ(y|x)/π_{ref}(y|x))) log π_θ(y|x) ]`
        *   `J_{k_2 as loss}(θ) = E_{x~D, y~π_θ(·|x)} [ (1/2) (log(π_θ(y|x)/π_{ref}(y|x)))^2 ]`
    *   **Finding:** `k_1` in reward and `k_2` as loss are gradient-equivalent and represent the theoretically sound implementations for RKL regularization.

4.  **Analyze `k_3` as loss (Approximation):**
    *   The term `k_3(x) = delta(x) - 1 - log(delta(x))`.
    *   When used "as loss," its gradient is derived.
    *   **Resulting Gradient-Equivalent Coefficient:** `c_{3'}(y) = 1 - delta(y)`.
    *   **Finding:** `k_3` as loss is a first-order, biased approximation of the principled `−log(delta)` coefficient. This approximation is only accurate when `delta ≈ 1`.
    *   **Issues with `k_3` as loss:**
        *   **Bias:** `1 - delta ≠ -log(delta)` for `delta ≠ 1`.
        *   **Pathological asymmetry:** Over-coverage (`delta → 0`) leads to a weak, saturating restoring force (`1 - delta → 1`), while the principled `−log(delta) → +∞`. Under-coverage (`delta → ∞`) leads to `1 - delta → −∞` much faster than `−log(delta)` decays logarithmically, inducing explosive updates.
        *   **Statistical instability:** The variance of `1 - delta` is `χ^2(π_ref || π_θ)`, which is notoriously unstable and can be infinite if support conditions are violated.

5.  **Propose Off-Policy Correction:**
    *   Common off-policy implementations of `k_n` as loss methods are biased due to neglected importance sampling (IS).
    *   **Correction:** Convert `k_n` as loss into an equivalent `k_n'` in reward coefficient by taking the derivative with respect to `log π_θ`. Then, apply standard PPO importance sampling and clipping to this `k_n'` coefficient, either in a combined form with the reward or as a separate clipped head.

**Key Formulas (in LaTeX):**

*   **KL Divergence:**
    `$D_{\mathrm{KL}}(q \parallel p) = \mathbb{E}_{x \sim q} \left[ \log \frac{q(x)}{p(x)} \right]$`
*   **KL Estimators:**
    `$k_1(x) = - \log \delta(x)$`
    `$k_2(x) = \frac{1}{2} \left(\log \delta(x)\right)^2$`
    `$k_3(x) = \delta(x) - 1 - \log \delta(x)$`
    where `$\delta(x) = p(x) / q(x)$`.
*   **RLHF Objective:**
    `$\mathcal{J}_{\mathrm{RLHF}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot | x)} [ r(x, y) ] - \beta D_{\mathrm{KL}} \left(\pi_{\theta}(\cdot | x) \parallel \pi_{\mathrm{ref}}(\cdot | x)\right)$`
*   **Gradient of `k_1` as loss:**
    `$\nabla_{\theta} \mathcal{J}_{k_1 \mathrm{as loss}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot | x)} [ \nabla_{\theta} \log \pi_{\theta}(y | x) ] = 0$` (in expectation)
*   **Principled RKL Gradient (Target):**
    `$\nabla_{\theta} \mathcal{J}_{\mathrm{RKL}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot | x)} \left[ \left(\log \frac{\pi_{\theta}(y | x)}{\pi_{\mathrm{ref}}(y | x)}\right) \nabla_{\theta} \log \pi_{\theta}(y | x) \right]$`
*   **`k_1` in reward objective (gradient-equivalent to target):**
    `$\mathcal{J}_{k_1 \mathrm{in reward}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot | x)} \left[ \left(\log \frac{\pi_{\theta}(y | x)}{\pi_{\mathrm{ref}}(y | x)}\right) \log \pi_{\theta}(y | x) \right]$`
*   **`k_2` as loss objective (gradient-equivalent to target):**
    `$\mathcal{J}_{k_2 \mathrm{as loss}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot | x)} \left[ \frac{1}{2} \left(\log \frac{\pi_{\theta}(y | x)}{\pi_{\mathrm{ref}}(y | x)}\right)^2 \right]$`
*   **Gradient-equivalent coefficient for `k_3` as loss:**
    `$c_{3'}(y) := 1 - \delta(y)$`
*   **Bias of `k_3` as loss:**
    `$\mathrm{Bias}(\delta) = (- \log \delta) - (1 - \delta) = \frac{1}{2} (\delta - 1)^2 - \frac{1}{3} (\delta - 1)^3 + O \big((\delta - 1)^4 \big)$`
*   **Variance of `k_3` as loss coefficient:**
    `$\mathrm{Var}_{y \sim \pi_{\theta}} [ 1 - \delta(y) ] = \chi^2(\pi_{ref}(\cdot | x) \parallel \pi_{\theta}(\cdot | x))$`
*   **Off-policy correction (general `k_n` as loss to `k_n'` in reward):**
    `$k_{n'}(\pi_{\theta}(y | x), \pi_{\mathrm{ref}}(y | x)) := \left. \frac{\partial}{\partial \log \pi_{\theta}} k_n(\pi_{\theta}(y | x), \pi_{\mathrm{ref}}(y | x)) \right|_{\log \pi = \log \pi_{\theta}(y | x)}$`

**Key Quantitative Results and Numbers:**

*   **Experimental Validation:** Controlled GRPO experiments on a mathematical reasoning task (OpenR1-Math-220k, NuminaMath 1.5 prompts, Deepseek-R1 traces, Qwen2.5-Math-1.5B actor model) with `β = 0.5`.
*   **`k_1` as loss:** Empirically, its training trajectories are nearly indistinguishable from a baseline with no KL penalty. It fails to keep the actor model closer to the reference and can even cause it to drift further away at later stages, especially on larger models (7B-scale). This confirms its theoretical ineffectiveness as a regularizer.
*   **`k_2` as loss vs. `k_3` as loss:** Both successfully regularize the policy.
    *   `k_2` as loss shows **greater training stability** and **stronger regularization**, reflected in lower variance of rewards and response lengths. It maintains tighter coupling to the reference policy (smaller actor-reference probability gap, "Logprob Diff with Smooth") and slightly higher entropy, suggesting better exploration.
    *   `k_3` as loss is a first-order surrogate that imposes **weaker constraints**, yielding larger probability gaps and reduced entropy. It exhibits greater drift and instability despite temporary performance gains.
*   **Gaussian Case for `k_3` variance:** For `p ~ N(µ_p, σ_p^2)` and `q ~ N(µ_q, σ_q^2)`, the variance of `k_3` is finite if and only if `σ_q^2 > σ_p^2 / 2`. An empirical demonstration with `p ~ N(0, 1)` and `q ~ N(0.1, 0.2)` (violating `0.2^2 > 1^2 / 2`) shows `k_3` having a sample standard deviation several times larger than `k_1` (e.g., 8.8244 vs 0.6912 in one run), indicating severe statistical instability.

**Stated Limitations:**

*   The equivalence between `k_1` in reward and `k_2` as loss requires **on-policy sampling**. For off-policy updates, importance sampling corrections are necessary.
*   The analysis simplifies the entire response `y` as a single action, rather than token-level probabilities used in standard sequential PPO.
*   Normalization techniques (e.g., batch normalization, group normalization) are biased but often crucial heuristics for practical stability, even if omitted from simplified theoretical analyses.
*   The `k_3` estimator's claim of being "strictly better" (John, 2020) does not hold in the general case, as potential issues of severe bias and infinite variance can arise when distribution support or tails differ significantly. Its application requires careful verification of assumptions.
*   The `k_3` as loss formulation is a first-order proxy that is accurate only when the policy is very close to the reference (`delta ≈ 1`).
