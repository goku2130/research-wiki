---
id: arxiv:2312.00886
type: paper
title: Nash Learning from Human Feedback
url: https://arxiv.org/abs/2312.00886
retrieved: '2026-07-12'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Nash Learning from Human Feedback (NLHF)

### Core Problem
Traditional Reinforcement Learning from Human Feedback (RLHF) typically relies on learning a scalar reward model based on the Bradley-Terry (BT) model from pairwise human preferences. The authors identify three primary limitations of this approach:
1. **Preference Limitations:** BT models cannot accommodate the full spectrum of human preferences, such as non-transitive preferences (where $y_1 \succ y_2$, $y_2 \succ y_3$, and $y_3 \succ y_1$).
2. **Alignment Issues:** Maximizing a scalar reward (like an Elo score) may not align with maximizing the probability of winning against a population, potentially leading to deterministic policies that ignore the diversity of human preferences.
3. **Data Sensitivity:** Reward models are sensitive to the distribution of data used for training, necessitating complete relearning when data distributions shift during iterative RLHF.

### Method
NLHF replaces the reward model with a **preference model** $P(y \succ y' | x)$, representing the probability that a human prefers response $y$ over $y'$ given prompt $x$. The objective is to find the **Nash equilibrium** ($\pi^*$) of this preference model:

$$
\pi^{*} = \arg \max_{\pi} \min_{\pi'} \mathcal{P}(\pi \succ \pi')
$$

#### Regularized Preference Model
To ensure the policy remains close to a reference policy $\mu$, the authors introduce a KL-regularized preference:

$$
\mathcal{P}_{\tau}(\pi \succ \pi') = \mathcal{P}(\pi \succ \pi') - \tau \mathbf{KL}_{\rho}(\pi, \mu) + \tau \mathbf{KL}_{\rho}(\pi', \mu)
$$

where $\mathbf{KL}_{\rho}(\pi, \mu) = \mathbb{E}_{x \sim \rho}[\text{KL}(\pi(\cdot|x), \mu(\cdot|x))]$.

#### Nash-MD Algorithm (Tabular)
For tabular policies, the authors propose **Nash-MD**, a mirror descent variant that converges to the regularized Nash equilibrium. The recipe is as follows:
1. **Define a regularized policy** $\pi_t^\mu$ as a geometric mixture of the current policy $\pi_t$ and reference policy $\mu$:

$$
\pi_{t}^{\mu}(y) = \frac{\pi_{t}(y)^{1-\eta_{t}\tau}\mu(y)^{\eta_{t}\tau}}{\sum_{y'} \pi_{t}(y')^{1-\eta_{t}\tau}\mu(y')^{\eta_{t}\tau}}
$$

2. **Update the policy** using a mirror descent step:

$$
\pi_{t+1} = \arg \max_{\pi} \left[ \eta_t \mathcal{P}(\pi \succ \pi_t^\mu) - \text{KL}(\pi, \pi_t^\mu) \right]
$$

   Which simplifies to: $\pi_{t+1}(y) \propto \pi_{t}^{\mu}(y) \exp(\eta_{t}\mathcal{P}(y \succ \pi_{t}^{\mu}))$.

#### Deep Learning Implementation
For LLMs, the authors introduce two policy-gradient variants:
*   **Nash-MD-PG:** The alternative policy $\pi'$ is a geometric mixture: $\pi'(y|x) \propto (\pi_\theta(y|x))^{1-\beta}(\mu(y|x))^\beta$, where $\beta \in [0, 1]$.
*   **Nash-EMA-PG:** The alternative policy $\pi'$ is generated using an exponential moving average (EMA) of past policy parameters.

### Key Quantitative Results
*   **Convergence:** For tabular policies, Nash-MD exhibits last-iterate convergence to the Nash equilibrium $\pi_\tau^*$ in KL-divergence at a rate of $O(1/T)$. Specifically, with $\eta_t = 2/(\tau(t+2))$, $\text{KL}(\pi_\tau^*, \pi_T) \leq \frac{8}{\tau^2(T+1)}$.
*   **Experimental Performance:** In a text summarization task using the TL;DR dataset and PaLM 2 Large for evaluation:
    *   **Nash-MD-PG** outperformed RLHF, Self-Play ($\beta=0$), and Best-Response ($\beta=1$).
    *   Optimal performance was observed with mixture parameters $\beta \in [0.125, 0.375]$, suggesting that playing against a mixture of the current and reference policies is superior to playing against either alone.

### Stated Limitations
The authors note that the experimental comparison between NLHF and RLHF is not entirely "fair" because the two frameworks use fundamentally different models (a preference model vs. a reward model). Consequently, the experiments were intended to illustrate the practical implementation and potential of the NLHF approach rather than to establish absolute superiority over RLHF.
