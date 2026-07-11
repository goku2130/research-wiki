---
id: arxiv:2510.00502
type: paper
title: Diffusion Alignment as Variational Expectation-Maximization
url: https://arxiv.org/abs/2510.00502
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Diffusion Alignment as Variational Expectation-Maximization (DAV)

### Core Problem
Diffusion alignment aims to optimize pretrained diffusion models for specific downstream objectives (e.g., aesthetic quality or biological activity). Existing alignment methods typically rely on reinforcement learning (RL) or direct backpropagation. However, RL-based approaches often utilize a reverse-KL objective, which is "mode-seeking" and prone to mode collapse and premature convergence. Direct backpropagation depends on brittle gradient signals from reward functions, often leading to severe reward over-optimization. There is a need for a framework that maximizes rewards while preserving the diversity and naturalness of the original pretrained model.

### Method
The authors propose **Diffusion Alignment as Variational Expectation-Maximization (DAV)**, which formulates alignment as an iterative process alternating between an E-step and an M-step to maximize the marginal likelihood of an optimality variable $\mathcal{O}$.

#### Step-by-Step Recipe:
1.  **E-step (Posterior Inference):** The goal is to discover high-reward, multi-modal trajectories $\tau$ from the variational posterior distribution $\eta_k$.
    *   **Target Distribution:** The optimal posterior $\eta_k^*$ is a reward-tilted distribution:

$$
\eta_{k}^*(x_{t-1}|x_{t}) \propto p_{\theta^{k}}(x_{t-1}|x_{t}) \exp(Q_{\text{soft},\theta^{k}}^*(x_{t},x_{t-1})/\alpha)
$$

    *   **Two-Stage Local Search:** Since sampling from $\eta_k^*$ is intractable, DAV uses:
        *   **Proposal Construction:** A proposal distribution $\hat{\eta}_k$ is created using gradient guidance (via first-order Taylor expansion of the reward function) to shift the mean of the prior distribution.
        *   **Importance Sampling:** $M$ particles are drawn from $\hat{\eta}_k$ and refined using importance weights $w_{t-1}^m = \frac{\eta^*(x_{t-1}^m|x_t)}{\hat{\eta}(x_{t-1}^m|x_t)}$ to correct the mismatch between the proposal and the true posterior.
2.  **M-step (Amortization):** The knowledge from the trajectories discovered in the E-step is distilled back into the model parameters $\theta$.
    *   **Forward-KL Projection:** Unlike RL's reverse-KL, the M-step minimizes the forward-KL divergence, which is "mode-covering" and encourages the model to cover all diverse modes found during the E-step.
    *   **Regularization (DAV-KL):** To prevent capability loss, a KL-divergence penalty is added to constrain the updated model from deviating too far from the initial pretrained policy $p_{\theta^0}$.

### Key Formulas
The alignment objective is to maximize the Evidence Lower Bound (ELBO) $\mathcal{J}_{\alpha,\gamma}(\eta,p_{\theta})$:

$$
\mathcal{J}_{\alpha,\gamma}(\eta,p_{\theta}) = \mathbb{E}_{\tau\sim\eta}\left[\sum_{t=1}^{T}\gamma^{T-t}\left(\frac{r(x_{t},x_{t-1})}{\alpha}+\text{l o g}\frac{p_{\theta}(x_{t-1}\mid x_{t})}{\eta(x_{t-1}\mid x_{t})}\right)\right]
$$

where $\gamma \in (0,1]$ is a discount factor and $\alpha$ is a temperature parameter.

The soft Q-function is approximated using Tweedie’s formula:

$$
Q_{\text{soft}}^*(x_t, x_{t-1}) \approx \gamma^{t-1} r(\hat{x}_0(x_{t-1}))
$$

The training objective for the **DAV-KL** variant is:

$$
\mathcal{L}_{\text{DAV-KL}}(\theta) = \mathbb{E}_{\tau\sim\eta_{k}^{*}}\left[\sum_{t=1}^{T}-\text{l o g}p_{\theta}(x_{t-1}|x_{t})\right] + \lambda D_{\text{KL}}(p_{\theta}(x_{t-1}|x_{t})||p_{\theta^{0}}(x_{t-1}|x_{t}))
$$

### Key Quantitative Results
DAV was evaluated on continuous (text-to-image) and discrete (DNA sequence) tasks:

*   **Text-to-Image (Stable Diffusion v1.5):**
    *   **Reward:** DAV achieved an aesthetic score of **8.04**, significantly higher than DDPO (6.83) and DRaFT (7.22).
    *   **Alignment:** DAV maintained an ImageReward score of **0.95**, whereas DDPO (0.27) and DRaFT (0.19) suffered from severe over-optimization.
    *   **Diversity:** DAV-KL improved ImageReward to **1.13** and maintained higher diversity (LPIPS-A of 0.58) compared to DDPO (0.48).
    *   **Test-time Search:** DAV Posterior achieved the highest aesthetic score of **9.18**.

*   **DNA Sequence Design (Discrete Diffusion):**
    *   DAV outperformed baselines (DRAKES, DDPO, VIDD) in the trade-off between predicted enhancer activity (Pred-Activity), sequence diversity (Levenshtein distance), and naturalness (3-mer Pearson Correlation).

### Stated Limitations
1.  **Computational Overhead:** The test-time search required in the E-step introduces significant computational costs during training.
2.  **Approximation Error:** The soft Q-function relies on Tweedie’s formula, which can be inaccurate at high noise levels (large timesteps $x_t$), potentially guiding the search toward suboptimal distributions.
