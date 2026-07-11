---
id: arxiv:1802.01561
type: paper
title: 'IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner
  Architectures'
url: https://arxiv.org/abs/1802.01561
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# IMPALA: Importance Weighted Actor-Learner Architectures

### Core Problem
The primary challenge addressed by IMPALA is the scalability of deep reinforcement learning (RL) when training a single agent across a large collection of diverse tasks. Traditional distributed architectures, such as A3C, communicate gradients from workers to a central parameter server, which can be inefficient and slow. While decoupling acting from learning (sending trajectories instead of gradients) increases throughput, it introduces "policy lag"—a discrepancy between the behavior policy $\mu$ used by actors to generate data and the target policy $\pi$ being updated by the learner. This off-policy lag can lead to instability and reduced data efficiency.

### Method and Recipe
IMPALA employs a decoupled actor-learner architecture combined with a novel off-policy correction algorithm called **V-trace**.

**1. Distributed Architecture:**
*   **Actors:** Multiple actors independently generate trajectories of experience. At the start of each trajectory, an actor retrieves the latest policy parameters $\pi$ from the learner to update its local policy $\mu$. It runs for $n$ steps, then sends the resulting trajectory (states, actions, rewards, policy distributions $\mu(a_i|x_i)$, and initial LSTM states) to the learner via a queue.
*   **Learner:** One or more learners use GPUs to perform updates on min-batches of trajectories. To maximize throughput, the learner parallelizes time-independent operations (e.g., applying convolutional networks to all time steps in a batch simultaneously) and utilizes XLA compilation and cuDNN optimizations.

**2. V-trace Algorithm:**
V-trace corrects the policy lag by using truncated importance sampling.
*   **Value Target:** The $n$-step V-trace target $v_s$ for the value function $V(x_s)$ is defined as:

$$
v_s \stackrel{\text{def}}{=} V(x_s) + \sum_{t=s}^{n+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta V
$$

    where the temporal difference $\delta V$ is:

$$
\delta V \stackrel{\text{def}}{=} \rho_t \left( r_t + \gamma V(x_{t+1}) - V(x_t) \right)
$$

*   **Truncated Weights:** The algorithm uses two sets of truncated importance sampling weights:

$$
\rho_t \stackrel{\text{def}}{=} \min\left(\bar{\rho}, \frac{\pi(a_t|x_t)}{\mu(a_t|x_t)}\right), \quad c_i \stackrel{\text{def}}{=} \min\left(\bar{c}, \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)}\right)
$$

    The weight $\rho_t$ determines the fixed point of the update (the policy being evaluated), while $c_i$ acts as a variance reduction technique that affects the convergence speed without altering the fixed point.

**3. Actor-Critic Update:**
*   **Value Function:** Updated via gradient descent on the $L_2$ loss between $V_\phi(x_i)$ and the V-trace target $v_{n,i}$.
*   **Policy:** Updated in the direction of the policy gradient:

$$
\rho_\pi \nabla_\omega \log \pi_\pi(a_x|x_0) \left(r_s + \gamma v_{n+1} - V_\phi(x_i)\right)
$$

    An entropy bonus is added to prevent premature convergence.

### Key Quantitative Results
*   **Throughput:** IMPALA achieved data throughput rates of **250,000 frames per second**, making it over 30 times faster than single-machine A3C.
*   **DMLab-30:** In a multi-task setting of 30 diverse tasks, IMPALA (deep architecture with Population Based Training) achieved a mean capped human normalised score of **49.4%**, significantly outperforming distributed A3C (**23.8%**).
*   **Atari-57:** A single agent trained on all 57 Atari games simultaneously reached a **59.7%** median human normalised score, remaining competitive with expert baselines.
*   **Wall-clock Efficiency:** On DMLab-30, IMPALA with one learner reached the same performance in approximately **10 hours** that A3C approached after **7.5 days**.

### Stated Limitations
The authors note that the use of truncation levels ($\bar{\rho}$) in V-trace introduces a bias; if $\bar{\rho} < \infty$, the algorithm converges to the value function of a policy $\pi_\rho$ that lies between the behavior policy $\mu$ and the target policy $\pi$. Additionally, while the multi-task agent was competitive on Atari-57, the authors acknowledge that the ALE suite is a particularly challenging multi-task environment often characterized by negative transfer between tasks.
