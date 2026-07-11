---
id: arxiv:2406.04896
type: paper
title: Stabilizing Extreme Q-learning by Maclaurin Expansion
url: https://arxiv.org/abs/2406.04896
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Stabilizing Extreme Q-learning by Maclaurin Expansion

### Core Problem
Extreme Q-learning (XQL) is an in-sample offline reinforcement learning method that assumes Bellman errors follow a Gumbel distribution. While this allows XQL to model the soft optimal value function without evaluating out-of-distribution actions, it suffers from two primary issues:
1. **Instability:** The Gumbel loss function contains an exponential term that can produce excessively large or small gradients, leading to slow convergence or total learning collapse.
2. **Distribution Mismatch:** The assumption of a Gumbel distribution requires independence between state-action pairs and time steps, which is rarely guaranteed in practice due to the use of neural networks and techniques like target networks. The actual error distribution often resembles a mixture of Normal and Gumbel distributions.

### Method: Maclaurin Expanded Extreme Q-learning (MXQL)
The authors propose **Maclaurin Expanded Extreme Q-learning (MXQL)**, which replaces the standard Gumbel loss with an **Expanded Gumbel loss** derived via Maclaurin expansion. This modification allows for a tunable trade-off between stability (associated with Normal distributions/L2 loss) and optimality (associated with Gumbel distributions/soft Q-learning).

#### Step-by-Step Recipe
1. **Value Function Estimation:** Instead of the standard Gumbel loss, the value function $V(s)$ is learned using the Expanded Gumbel loss.
2. **Expansion Order Selection:** A hyperparameter $n$ (selected from even numbers) is chosen to determine the order of the expansion. 
    - When $n=2$, the loss is equivalent to the $L_2$ loss, assuming a normal error distribution and behaving like SARSA.
    - As $n \to \infty$, the loss converges to the Gumbel loss, assuming a Gumbel error distribution and behaving like soft Q-learning.
3. **Q-Function Learning:** The Q-function is learned using the standard least squares method, consistent with XQL.
4. **Hyperparameter Tuning:** The method requires tuning $\beta$ (which controls conservatism and stability) and $n$ (which controls the distribution assumption and the stability-optimality trade-off).

### Key Formulas
The **Expanded Gumbel loss** for the value function $V$ is defined as:

$$
L(V)=\mathbb{E}_{s,a\sim\mu}\left[\sum_{j=2}^{n}\frac{(Q(s,a)-V(s))^{j}}{j!\beta^{j}}\right]
$$

The **Q-function loss** remains the standard mean squared Bellman error:

$$
L (Q) = \mathbb {E} _ {s, a, s ^ {\prime} \sim \mathcal {D}} [ \left(r (s, a) + \gamma V \left(s ^ {\prime}\right) - Q (s, a)\right) ^ {2} ]
$$

### Key Quantitative Results
MXQL was evaluated on DM Control (online RL) and D4RL (offline RL) benchmarks.

**Online RL (DM Control):**
MXQL ($n=8$) demonstrated significantly higher stability and performance than XQL, particularly with small $\beta$ values where XQL often failed to learn.
- **QuadrupedRun-v0:** MXQL (best $\beta=1$) achieved $896.0 \pm 51.4$ compared to XQL (best $\beta=5$) at $730.2 \pm 303.8$.
- **HopperHop-v0:** MXQL (best $\beta=1$) achieved $362.7 \pm 115.7$ vs. XQL (best $\beta=2$) at $287.4 \pm 9.1$.

**Offline RL (D4RL):**
MXQL outperformed XQL in most tasks and showed competitive results against other baselines.
- **AntMaze-umaze:** MXQL achieved a score of $88.3 \pm 2.1$, significantly higher than XQL's $47.7$.
- **Kitchen-mixed:** MXQL achieved $71.9 \pm 3.6$ compared to XQL's $40.4$.
- **Hopper-med-exp:** MXQL achieved $105.1 \pm 10.1$, outperforming XQL's $94.0$.

### Stated Limitations
The authors note that while MXQL improves upon XQL, the soft optimal value estimation used by both methods may not be well-suited for all tasks. Specifically, in **AntMaze medium and large** tasks, MXQL underperforms compared to Implicit Q-Learning (IQL). Additionally, there is an inherent trade-off where smaller $n$ values increase stability but result in a non-maximizing estimation of the value function.
