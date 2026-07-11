---
id: arxiv:1802.01561
type: paper
title: Off-Policy Deep Reinforcement Learning without Exploration
url: https://arxiv.org/abs/1802.01561
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# IMPALA: Importance Weighted Actor-Learner Architectures

### Core Problem
The primary challenge addressed by IMPALA is the scalability of deep reinforcement learning (RL) when training a single agent on a large collection of diverse tasks. Existing distributed methods, such as A3C, communicate gradients from workers to a central parameter server, which can be inefficient. While decoupling acting (data collection) from learning (gradient updates) allows for higher throughput via GPU acceleration, it introduces "policy lag." This lag occurs because the behavior policy $\mu$ used by actors to generate trajectories is often several updates behind the target policy $\pi$ maintained by the learner, rendering the data off-policy and potentially destabilizing learning.

### Method and Recipe
IMPALA utilizes a decoupled actor-learner architecture combined with a novel off-policy correction algorithm called **V-trace**.

**1. Distributed Architecture**
*   **Actors:** Multiple actors independently generate trajectories of experience (states, actions, rewards) using a local policy $\mu$. At the start of each trajectory, actors retrieve the latest parameters from the learner.
*   **Communication:** Actors send trajectories, corresponding policy distributions $\mu(a_i|x_i)$, and initial LSTM states to the learner via a queue.
*   **Learner:** One or more learners use GPUs to perform updates on minibatches of trajectories. To maximize throughput, the learner parallelizes time-independent operations (e.g., applying convolutional networks to all time steps in a batch simultaneously) and employs XLA compilation and cuDNN optimizations.

**2. V-trace Algorithm**
V-trace corrects the discrepancy between the behavior policy $\mu$ and the target policy $\pi$ using truncated importance sampling (IS) weights.

*   **V-trace Target:** The $n$-step target for the value function $V(x_s)$ is defined as:

$$
v_s = V(x_s) + \sum_{t=s}^{n+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta V
$$

    where the temporal difference $\delta V$ is:

$$
\delta V = \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
$$

*   **Truncated IS Weights:** The weights $\rho_t$ and $c_i$ are defined as:

$$
\rho_t = \min(\bar{\rho}, \frac{\pi(a_t|x_t)}{\mu(a_t|x_t)}), \quad c_i = \min(\bar{c}, \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)})
$$

    The truncation level $\bar{\rho}$ determines the fixed point of the update (the policy being evaluated), while $\bar{c}$ acts as a variance reduction technique that affects convergence speed but not the final solution.
*   **Policy Gradient Update:** The policy parameters $\omega$ are updated in the direction of:

$$
\rho_\pi \nabla_\omega \log \pi_\omega(a_x|x_0) (r_s + \gamma v_{s+1} - V_\phi(x_i))
$$

    where $q_s = r_s + \gamma v_{s+1}$ serves as an estimate of the state-action value $Q^\pi(x_0, a_x)$.

### Key Quantitative Results
*   **Throughput:** IMPALA achieved a throughput of **250,000 frames per second**, making it over 30 times faster than single-machine A3C.
*   **DMLab-30:** On a set of 30 diverse tasks, IMPALA (deep architecture with Population Based Training) achieved a mean capped human normalized score of **49.4%**, significantly outperforming distributed A3C's **23.8%**.
*   **Atari-57:** A single IMPALA agent trained on all 57 Atari games simultaneously reached a **59.7% median human normalized score**, remaining competitive with A3C shallow experts.
*   **Wall-clock Efficiency:** IMPALA with a single learner reached the same performance in approximately **10 hours** that A3C approached after **7.5 days**.

### Stated Limitations
The authors note that using a truncation level of $\bar{\rho} = 1$ for importance weights reduces gradient variance but introduces a bias into the value function estimate. Additionally, while IMPALA was competitive on the Atari-57 suite, the authors acknowledge that the ALE environment is particularly challenging for multi-task learning and often prone to negative transfer between tasks.
