---
id: arxiv:2212.14405
type: paper
title: Offline Policy Optimization in RL with Variance Regularization
url: https://arxiv.org/abs/2212.14405
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Offline Policy Optimization with Variance Regularization (OVR)

### Core Problem
Offline reinforcement learning (RL) faces significant challenges due to **distributional shift**, where a mismatch between the fixed offline dataset $\mathcal{D}$ and the target policy $\pi$ leads to high variance and the overestimation of value functions. This over-optimism often drives agents into state-action regions where the behavior is poor and data is scarce. While existing methods attempt to mitigate overestimation, they often struggle with the high variance inherent in off-policy value estimation.

### Method: Offline Variance Regularization (OVR)
The authors propose a framework that regularizes the offline policy optimization objective by minimizing the variance of marginalized importance sampling (IS) returns. To avoid the "double sampling" issue—a common bottleneck in variance-constrained RL where the gradient of the variance requires multiple independent samples—the authors employ **Fenchel duality**. This transforms the variance minimization into a minimax optimization problem, allowing the variance to be handled via dual variables.

#### Step-by-Step Recipe
The OVR algorithm can be used to augment any existing offline policy optimization algorithm (e.g., BCQ) as follows:

1.  **Estimate Distribution Ratio:** Use a DICE-based algorithm to estimate the stationary state-action distribution ratio $\omega(s, a) = \frac{d_{\pi}(s, a)}{d_{\mathcal{D}}(s, a)}$.
2.  **Compute Dual Variables:** Calculate the dual variables $\nu(s, a)$ using a closed-form solution: $\nu(s, a) = \omega(s, a) \cdot \tilde{r}(s, a)$.
3.  **Augment Rewards:** Define augmented rewards $\tilde{r}(s, a)$ to incorporate the variance penalty:

$$
\tilde{r}(s, a) \equiv [r - \lambda \nu r - \lambda r^2](s, a)
$$

    where $\lambda \geq 0$ is the regularization weight.
4.  **Policy Improvement:** Perform a policy update using a standard offline RL algorithm, replacing the original reward $r$ with the augmented reward $\tilde{r}(s, a)$ to optimize the variance-regularized value function $\tilde{Q}^{\pi_{\theta}}(s, a)$.
5.  **Iterate:** Repeat the process until convergence.

### Key Formulas
The overall max-min optimization objective is formulated as:

$$
\max_{\pi_{\theta}} \min_{\nu} J(\pi_{\theta}, \nu) := \mathbb{E}_{s \sim \mathcal{D}}[Q^{\pi_{\theta}}(s, \pi_{\theta}(s))] - \lambda \mathbb{E}_{(s, a) \sim d_{\mathcal{D}}} \left[ -\frac{1}{2}\nu^2 + \nu \omega r + \omega r^2 \right]
$$

The authors prove that this regularizer provides a lower bound to the original objective. Specifically, for any $0 < \delta \leq 1$ and $N > 0$, with probability at least $1-\delta$:

$$
J(\pi) \geq \mathbb{E}_{(s, a) \sim d_{\mathcal{D}}} [\omega_{\pi/\mathcal{D}}(s, a) r(s, a)] - \sqrt{\frac{1-\delta}{\delta}} \text{Var}_{(s, a) \sim d_{\mathcal{D}}} [\omega_{\pi/\mathcal{D}}(s, a) r(s, a)]
$$

### Quantitative Results
OVR was evaluated on D4RL continuous control benchmarks, primarily augmenting the BCQ algorithm. Results indicate that OVR is most effective when there is a large distribution mismatch (e.g., medium-quality datasets).

**Normalized Returns (Selected Tasks):**
| Domain | Task | BCQ | BCQ+OVR |
| :--- | :--- | :---: | :---: |
| Gym | hopper-medium | 57.76 | **71.24** |
| Gym | walker-medium | 27.13 | **33.90** |
| Gym | hopper-medium-expert | 37.20 | **44.68** |
| Gym | walker-medium-expert | 29.00 | **34.53** |
| Adroit | pen-human | 56.58 | **64.12** |
| Adroit | hammer-expert | 108.38 | **119.32** |

In "expert" datasets, improvements were marginal because the demonstrations were already optimal. In "random" datasets, OVR provided no significant gain (e.g., HalfCheetah-random remained at 0.00), as the lack of quality data prevents effective policy improvement regardless of regularization.

### Stated Limitations
*   **Random Datasets:** The approach does not improve performance when the logging policy is entirely random, as the underlying data lacks the necessary signal for optimization.
*   **Distribution Ratio Estimation:** The method relies on estimating the stationary distribution ratio $\omega(s, a)$. While this reduces variance compared to trajectory-wise IS, it introduces potential bias.
*   **Hyperparameter Sensitivity:** The effectiveness of the regularization depends on the trade-off parameter $\lambda$.
