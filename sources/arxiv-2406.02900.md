---
id: arxiv:2406.02900
type: paper
title: Scaling Laws for Reward Model Overoptimization in Direct Preference Optimization
url: https://arxiv.org/abs/2406.02900
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-model-overoptimization
---

# Summary: Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms

### Core Problem
In classical Reinforcement Learning from Human Feedback (RLHF), "reward over-optimization" (or reward hacking) occurs when a policy exploits the imperfections of a learned proxy reward model, leading to a divergence where the proxy reward increases while true output quality declines. Direct Alignment Algorithms (DAAs), such as Direct Preference Optimization (DPO), bypass the explicit reward modeling phase. However, they still exhibit similar degradation patterns. This work seeks to formalize the reward over-optimization problem for DAAs, identify scaling laws governing this behavior, and explain the underlying theoretical cause.

### Method and Recipe
The authors investigate over-optimization across different objectives, model scales, and training regimes using the following steps:

1.  **Unification of DAAs**: The authors unify DPO, Identity Preference Optimization (IPO), and Sequence Likelihood Calibration (SLiC-HF) under a general DAA objective:

$$
\mathcal{L}_{\text{DAA}}(\pi_{\theta}; \pi_{\text{ref}}) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}[g(\beta \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)})]
$$

    where $g$ is a convex loss function (e.g., $-\log \sigma(x)$ for DPO, $(x-1)^2$ for IPO, and $\max(0, 1-x)$ for SLiC).
2.  **Empirical Evaluation**: Using the Reddit TL;DR summarization dataset and the Pythia model family (1B, 2.8B, and 6.9B parameters), the authors vary the $\beta$ parameter (controlling the KL budget) and measure win-rates against human answers as judged by GPT-4.
3.  **Dynamics Analysis**: They track intra-epoch training dynamics via intermediate checkpoints to observe when performance peaks relative to KL divergence.
4.  **Length Correlation Analysis**: To test for feature exploitation, they apply a linear regression to the implicit reward:

$$
\log \frac{\pi_{\theta}(y^{(i)}|x^{(i)})}{\pi_{ref}(y^{(i)}|x^{(i)})} = \hat{\gamma}|y^{(i)}| + \epsilon^{(i)}
$$

5.  **Theoretical Formalization**: The authors use a regression interpretation to demonstrate that the DAA objective is under-constrained (rank deficient) when the prompt-response space is exponentially larger than the preference dataset.
6.  **Toy MDP Validation**: A Tree MDP is used to visualize how probability mass is allocated to out-of-distribution (OOD) trajectories during training.

### Key Quantitative Results
*   **Over-optimization Trends**: All objectives exhibit a "hump-shaped" performance curve; as the KL budget increases, win-rates initially rise and then deteriorate.
*   **Intra-epoch Dynamics**: Under wider KL budgets, models often reach peak performance within the first 25% of a single epoch, after which performance declines as KL divergence increases.
*   **Model Scaling**: Larger models (6.9B) are less prone to over-optimization and achieve better win-rate/KL trade-offs than smaller models (1B).
*   **Objective Comparison**: IPO is less prone to over-optimization and maintains lower KL values under the same constraints compared to DPO and SLiC.
*   **Scaling Law Fit**: The authors find that the reward scaling law $R(d) = d(\alpha - \beta \log d)$, where $d = \sqrt{D_{KL}(\pi||\pi_{ref})}$, accurately relates KL divergence to win-rates, halving the RMSE compared to a quadratic fit.
*   **Length Exploitation**: Smaller models and models with smaller KL budgets extrapolate more strongly on simple features like length (higher $R^2$ in length regression).

### Theoretical Findings
The authors argue that DAAs suffer from "OOD bootstrapping." Because the DAA objective is not strictly convex and the data matrix is rank-deficient, there are infinite minima. This allows the policy to place significant probability mass on OOD responses that do not appear in the preference dataset while still minimizing the loss. In a token-level MDP interpretation, the Q-value estimate becomes overly optimistic for unseen tokens, particularly when $\beta$ is small.

### Stated Limitations
*   **Scale**: Due to computational constraints, the authors could not characterize these effects at larger model scales.
*   **Resolution**: The work identifies and formalizes the problem but does not provide a method to resolve or mitigate the over-optimization.
*   **Preference Modeling**: The analysis still assumes an underlying model of human preferences, which the authors acknowledge is an imperfect and ongoing area of research.
