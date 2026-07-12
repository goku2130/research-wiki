---
id: arxiv:2507.03068
type: paper
title: Mitigating Goal Misgeneralization via Minimax Regret
url: https://arxiv.org/abs/2507.03068
retrieved: '2026-07-12'
maturity: comprehensive
topic: sycophancy-and-misgeneralization
---

# Mitigating Goal Misgeneralization via Minimax Regret

### Core Problem
The authors address **goal misgeneralization**, a failure mode in reinforcement learning (RL) where an agent learns a **proxy goal** that incentivizes behavior similar to the intended (true) goal in training but diverges in novel deployment environments. This typically occurs during a **proxy-distinguishing distribution shift**, where the training distribution $\Lambda^{\text{Train}}$ consists primarily of "non-distinguishing" levels (where the proxy and true goals align), while the deployment distribution $\Lambda^{\text{Deploy}}$ consists primarily of "distinguishing" levels (where they diverge). Standard RL objectives, such as Maximum Expected Value (MEV), are susceptible to this because policies pursuing the proxy goal can remain approximately optimal if distinguishing situations are sufficiently rare.

### Method and Recipe
The authors propose using the **Minimax Expected Regret (MMER)** objective to mitigate this risk. The core idea is to employ an adversarial training framework—specifically **Unsupervised Environment Design (UED)**—where an adversary identifies and amplifies the training signal from high-regret, proxy-distinguishing situations.

**Step-by-Step Training Process:**
1. **Environment Setup:** Define an Underspecified MDP (UMDP) with a space of levels $\Theta$ and a true goal $R$.
2. **Adversarial Level Selection:** Instead of sampling levels from a fixed distribution (Domain Randomization), a regret-maximizing adversary selects a level distribution $\Lambda$ that maximizes the agent's expected regret.
3. **Policy Optimization:** The agent is trained using PPO (Proximal Policy Optimization) to maximize expected return on the levels provided by the adversary.
4. **Regret Estimation:** The adversary updates its level buffer based on estimated regret. The authors test two estimators:
    * **Max-latest:** A domain-agnostic estimator using sample-based maximum returns.
    * **Oracle-latest:** A domain-specific estimator using graph algorithms to compute exact maximum returns.
5. **Complexity Compounding (ACCEL):** To improve exploration, the ACCEL method applies stochastic edits to levels in the buffer to generate similar, high-regret situations.

### Key Formulas
The **expected return** of policy $\pi$ in level $\theta$ under goal $R$ is:

$$
V^{R}(\pi;\theta) = \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^{t}R(s_{t},a_{t},s_{t+1})\right]
$$

The **expected regret** in level $\theta$ is the shortfall compared to an optimal policy $\pi'$:

$$
G^{R}(\pi;\theta) = \max_{\pi^{\prime}\in\Pi}V^{R}(\pi^{\prime};\theta) - V^{R}(\pi;\theta)
$$

The **MMER objective** seeks a policy that minimizes the maximum possible expected regret across all level distributions $\Lambda \in \Delta(\Theta)$:

$$
\min_{\pi\in\Pi} \max_{\Lambda\in\Delta(\Theta)} \mathbb{E}_{\theta\sim\Lambda}[G^{R}(\pi;\theta)]
$$

### Key Quantitative Results
The authors evaluated three grid-world environments (*Cheese in the Corner*, *Cheese on a Dish*, and *Keys and Chests*) by varying the proportion of distinguishing levels in training ($\alpha$).

*   **MEV Susceptibility:** Domain Randomization (DR) exhibited goal misgeneralization whenever $\alpha$ was small.
*   **MMER Robustness:** In *Cheese in the Corner*, all UED methods (PLR and ACCEL) remained robust to goal misgeneralization at $\alpha = 1\text{e-}2$ (1%), whereas DR failed.
*   **Algorithm Comparison:** ACCEL generally outperformed $\text{PLR}^\perp$ (Robust Prioritized Level Replay) due to its ability to synthesize new distinguishing levels via edits.
*   **Estimation Impact:** In the *Keys and Chests* environment, DR actually outperformed ACCEL when using the **max-latest** estimator. This was attributed to biased regret estimates in complex environments where high returns are unlikely to be found by chance.

### Stated Limitations
*   **Implementation Gap:** While MMER is provably robust in the theoretical limit, current UED methods do not always find perfect MMER policies and can still exhibit misgeneralization if $\alpha$ is extremely low.
*   **Regret Estimation:** Reliable estimation of the maximum possible return ($\max V^R$) is a significant challenge; biased estimators can lead the adversary astray, particularly in complex, multi-stage tasks.
*   **Adversarial Discovery:** If distinguishing levels are entirely absent or too rare to be sampled initially, the adversary may fail to amplify them.
