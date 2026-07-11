---
id: arxiv:2310.12036
type: paper
title: A General Theoretical Paradigm to Understand Learning from Human Preferences
url: https://arxiv.org/abs/2310.12036
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

# Summary: A General Theoretical Paradigm to Understand Learning from Human Preferences

### Core Problem
The authors address theoretical and practical instabilities in Learning from Human Preferences (LHP), specifically within Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO). Both methods rely on the approximation that pairwise preferences can be substituted with pointwise rewards via the Bradley-Terry (BT) model. This assumption is problematic when preferences are deterministic or nearly deterministic, as it can lead to overfitting. In such cases, the optimal policy may ignore the KL-regularization term—which is intended to prevent model drift—and converge to degenerate, "greedy" solutions (e.g., pushing action probabilities to 0 or 1) regardless of the regularization strength $\tau$.

### The $\Psi$PO Framework
To unify and analyze these methods, the authors propose the $\Psi$-preference optimization ($\Psi$PO) objective. This general objective balances the maximization of a non-linear function of preference probabilities with a KL-divergence penalty:

$$
\max_{\pi}\mathop{\mathbb{E}}_{\substack{x\sim \rho \\ y\sim \pi (\cdot |x)\\ y^{\prime}\sim \mu (\cdot |x)}}\left[\Psi (p^{*}(y\succ y^{\prime}|x))\right] - \tau D_{\mathrm{KL}}(\pi \mid \mid \pi_{\mathrm{ref}})
$$

where $\Psi$ is an arbitrary non-decreasing mapping, $\pi_{\text{ref}}$ is the reference policy, and $\tau$ is the regularization parameter. 

*   **RLHF/DPO Connection:** When $\Psi(q) = \log(q/(1 - q))$, the $\Psi$PO objective recovers the optimal policies of RLHF and DPO, provided the BT model holds.
*   **IPO (Identity Preference Optimization):** By setting $\Psi$ to the identity mapping, the authors derive IPO, which directly optimizes total preferences and bypasses the BT assumption.

### IPO Method and Recipe
IPO is designed to be computationally efficient and avoid the need for an explicit reward model or RL. The optimization process follows these steps:

1.  **Objective Definition:** Maximize the total preference of policy $\pi$ over $\mu$ minus the KL divergence:

$$
\max _ {\pi} p _ {\rho} ^ {*} (\pi \succ \mu) - \tau D _ {\mathrm{KL}} (\pi | | \pi_{\mathrm{ref}})
$$

2.  **Root-Finding Formulation:** The optimal policy is characterized by the gap between log-likelihood ratios. Define the log-ratio $h_\pi$ as:

$$
h _ {\pi} (y, y ^ {\prime}, x) = \log \left(\frac {\pi (y | x) \pi_{\mathrm{ref}} (y ^ {\prime} | x)}{\pi (y ^ {\prime} | x) \pi_{\mathrm{ref}} (y | x)}\right)
$$

3.  **Sampled Loss Implementation:** To learn from a dataset $\mathcal{D}$ of preferred ($y_w$) and dispreferred ($y_l$) generations, IPO minimizes the following squared loss:

$$
\underset {(y _ {w}, y _ {l}, x) \sim D} {\mathbb {E}} \left[ \left(h _ {\pi} (y _ {w}, y _ {l}, x) - \frac{\tau^{-1}}{2}\right)^2 \right]
$$

    This regresses the difference in log-likelihood ratios to a constant $\frac{\tau^{-1}}{2}$, ensuring the policy remains regularized toward $\pi_{\text{ref}}$.

### Key Quantitative Results
The authors demonstrate IPO's superiority over DPO through illustrative bandit examples:

*   **Asymptotic Deterministic Setting:** With two actions where $p^*(y_1 \succ y_2) = 1$, DPO converges to a deterministic policy $\pi^*(y_1) = 1$ regardless of $\tau$. In contrast, IPO converges to $\pi^*(y_1) = \sigma(0.5\tau^{-1})$, meaning $\tau$ effectively controls the distance from $\pi_{\text{ref}}$.
*   **Sampled Preferences:** In a 3-action space using the Adam optimizer (LR 0.01, mini-batch 9, 18,000 steps), DPO was found to:
    *   Push the probability of a dominating action to 1 regardless of $\tau$.
    *   Push the probability of an action that never wins to 0 regardless of $\tau$.
    *   Ignore $\pi_{\text{ref}}$ entirely when certain action pairs are unobserved.
*   **IPO Performance:** In all the above scenarios, IPO avoided degenerate solutions and remained close to $\pi_{\text{ref}}$ proportional to the strength of $\tau$.

### Stated Limitations
*   **Support Requirements:** The uniqueness of the global minimum for the IPO loss $L(\pi)$ is guaranteed only if the support of the behavior policy $\mu$ coincides with the support of the reference policy $\pi_{\text{ref}}$. If $\text{Supp}(\mu) \neq \text{Supp}(\pi_{\mathrm{ref}})$, multiple solutions may exist.
*   **Experimental Scale:** The empirical results are provided via "minimal" illustrative bandit examples; the authors note that scaling these results to complex generative language models is a subject for future work.
