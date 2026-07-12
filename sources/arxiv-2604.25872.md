---
id: arxiv:2604.25872
type: paper
title: 'When Errors Can Be Beneficial: A Categorization of Imperfect Rewards for Policy
  Gradient'
url: https://arxiv.org/abs/2604.25872
retrieved: '2026-07-12'
maturity: comprehensive
topic: rl-for-math-and-code
---

# Summary: When Errors Can Be Beneficial

### Core Problem
Reinforcement learning (RL) for language models typically relies on **proxy reward functions** ($r_{\mathrm{P}}$) because **ground truth rewards** ($r_{\mathrm{G}}$) are rarely available. Standard evaluation metrics, such as ranking accuracy, treat any discrepancy where $r_{\mathrm{P}} \neq r_{\mathrm{G}}$ as strictly harmful. This work challenges that assumption, arguing that certain reward errors are benign or even beneficial by preventing the policy from stalling on outputs with mediocre ground truth rewards.

### Method and Technical Setting
The authors analyze policy gradient optimization using a **linear softmax policy** in bandit environments (single input $x$, finite output set $\mathcal{Y}$):

$$
\pi_\theta(y) := \frac{\exp(\langle \phi(y), \theta \rangle)}{\sum_{z \in \mathcal{Y}} \exp(\langle \phi(z), \theta \rangle)}
$$

where $\phi(y)$ is a feature vector. They analyze the optimization via **gradient flow**:

$$
\frac{d}{dt}\theta_t = \nabla V_{\mathrm{P}}(\theta_t)
$$

The dynamics of the logit for an output $y$ are governed by:

$$
\frac{d}{dt} \langle \phi(y), \theta_t \rangle = \pi_{\theta_t}(y) A_{\mathrm{P}}(y; \theta_t) \cdot \|\phi(y)\|^2 + \sum_{z \in \mathcal{Y} \setminus \{y\}} \pi_{\theta_t}(z) A_{\mathrm{P}}(z; \theta_t) \cdot \langle \phi(z), \phi(y) \rangle
$$

where $A_{\mathrm{P}}(z; \theta_t) := r_{\mathrm{P}}(z) - V_{\mathrm{P}}(\theta_t)$ is the advantage.

#### Reward Error Categorization
The authors categorize errors based on their effect on the increase of ground truth reward:
1.  **Harmful Errors:** 
    *   **Type I (Reward Hacking):** High $r_{\mathrm{P}}$ for low $r_{\mathrm{G}}$ outputs.
    *   **Type II (Stalling):** Mediocre $r_{\mathrm{P}}$ for low $r_{\mathrm{G}}$ outputs, causing the policy to concentrate and stall on suboptimal outputs.
2.  **Benign Errors:** 
    *   **Type I:** $r_{\mathrm{P}}(y)$ is lower than the initial expected proxy reward $V_{\mathrm{P}}(\theta_0)$, meaning $y$ does not attract probability.
    *   **Type II:** $y$ has negligible probability under the initial policy $\pi_{\theta_0}$.
3.  **Beneficial Errors:** 
    *   **Type I:** Assigning low $r_{\mathrm{P}}$ to an output with mediocre $r_{\mathrm{G}}$ prevents the policy from stalling, steering it toward the optimal output $y_{\star}$.

### Key Quantitative Results
**Theorem 1/2 (Optimization Gap):** In a setting with orthonormal features, if the initial probability $\pi_{\theta_0}(y_{\star})$ is sufficiently small relative to a mediocre output $y_{\mathrm{med}}$, the time $t_{\star}$ to achieve high ground truth reward when maximizing $r_{\mathrm{G}}$ directly is $\Omega(\pi_{\theta_0}(y_{\star})^{-14/13})$. In contrast, maximizing a proxy reward $r_{\mathrm{P}}$ that assigns low reward to $y_{\mathrm{med}}$ requires only $t_{\star}^{\mathrm{no-med}} = \mathcal{O}(\pi_{\theta_0}(y_{\star})^{-1})$. The gap $t_{\star} - t_{\star}^{\mathrm{no-med}}$ can be arbitrarily large.

**RLHF Evaluation:** The authors propose **Harm-Aware Ranking Accuracy (HAcc)**, which ignores incorrect rankings if the dispreferred output's reward is below the empirical expected proxy reward $\hat{V}_P(x; \theta)$:

$$
\text{HAcc}(r_P; \mathcal{S}) := \frac{1}{|\mathcal{S}|} \sum_{(x,y^+, y_k^-) \in \mathcal{S}} \mathbb{1} \left[ \max_{k \in [K]} r_P(x, y_k^-) < \max\{r_P(x, y^+), \hat{V}_P(x; \theta)\} \right]
$$

Experiments across Llama, OLMo, and Qwen families show HAcc correlates better with ground truth reward increase than standard accuracy, though correlations remain below 0.4.

**Verifiable Rewards:** In instruction-following tasks, rewarding partial correctness (e.g., 0.5 reward for 1 of 2 constraints) can be detrimental if the initial policy is significantly more likely to produce partially correct outputs than fully correct ones, leading the policy to learn only the easier constraint.

### Stated Limitations
*   **Theoretical Simplifications:** The analysis is limited to linear softmax policies, exact gradients (rather than sample-based), and bandit environments with single inputs.
*   **Evaluation Gaps:** Even with HAcc, the correlation between reward model accuracy and actual RL performance is weak, suggesting ranking-based metrics are inherently limited.
*   **Reward Design:** Partial rewards may still be beneficial if the initial probability of producing fully correct outputs is near-zero, necessitating adaptive reward schemes.
