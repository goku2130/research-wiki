---
id: arxiv:1602.01783
type: paper
title: Asynchronous Methods for Deep Reinforcement Learning (A3C)
url: https://arxiv.org/abs/1602.01783
retrieved: '2026-07-12'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# Asynchronous Methods for Deep Reinforcement Learning (A3C)

### Core Problem
Combining deep neural networks with online reinforcement learning (RL) is traditionally unstable because the sequence of observed data is non-stationary and updates are strongly correlated. Previous solutions relied on experience replay to decorrelate data, but this approach is memory-intensive and restricts the agent to off-policy RL algorithms.

### Method
The authors propose a lightweight framework that replaces experience replay with asynchronous execution of multiple actor-learners in parallel across multiple instances of the environment. This parallelism decorrelates the agents' data, stabilizing the training of both on-policy and off-policy algorithms.

While the authors present asynchronous variants of one-step Sarsa, one-step Q-learning, and n-step Q-learning, the primary focus is the **Asynchronous Advantage Actor-Critic (A3C)**.

#### A3C Recipe:
1.  **Architecture**: A shared convolutional neural network (CNN) is used, featuring one softmax output for the policy $\pi(a_t|s_t; \theta)$ and one linear output for the value function $V(s_t; \theta_v)$. Non-output layers are shared.
2.  **Parallelism**: Multiple actor-learner threads run on a single multi-core CPU. Each thread interacts with its own environment copy and maintains thread-specific parameters $\theta'$ and $\theta'_v$.
3.  **Interaction**: The agent selects actions according to the policy $\pi$ for up to $t_{max}$ steps or until a terminal state is reached.
4.  **Update**: After $t_{max}$ steps, the agent computes gradients using n-step returns (forward view).
5.  **Advantage Estimation**: The policy is updated using an estimate of the advantage function:

$$
A(s_t, a_t; \theta, \theta_v) = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k}; \theta_v) - V(s_t; \theta_v)
$$

    where $k$ is upper-bounded by $t_{max}$.
6.  **Entropy Regularization**: To discourage premature convergence to suboptimal deterministic policies, an entropy term $H$ is added to the objective function:

$$
\nabla_{\theta'} \log \pi(a_t|s_t; \theta')(R_t - V(s_t; \theta_v)) + \beta \nabla_{\theta'} H(\pi(s_t; \theta'))
$$

    where $\beta$ controls the regularization strength.
7.  **Optimization**: The framework uses **Shared RMSProp**, where the moving average of squared gradients $g$ is shared across threads:

$$
g = \alpha g + (1 - \alpha) \Delta \theta^2 \text{ and } \theta \leftarrow \theta - \eta \frac{\Delta \theta}{\sqrt{g + \epsilon}}
$$

### Key Quantitative Results
*   **Atari 2600**: A3C significantly outperformed previous state-of-the-art methods. A3C with an LSTM agent achieved a mean human-normalized score of **623.0%** and a median of **112.6%** over 57 games.
*   **Training Efficiency**: A3C achieved these results in **4 days** using **16 CPU cores**, whereas previous GPU-based methods (e.g., Prioritized DQN) required 8 days. After only 1 day of training, A3C matched the average human-normalized score of Dueling Double DQN.
*   **Scalability**: Training speedup was roughly linear with the number of threads. Using 16 threads resulted in at least an order of magnitude speedup across seven tested Atari games.
*   **Other Domains**:
    *   **TORCS**: A3C reached **75% to 90%** of human tester scores in approximately 12 hours of training.
    *   **Labyrinth**: An A3C LSTM agent achieved an average score of approximately **50** in navigating random 3D mazes from visual input.
    *   **MuJoCo**: A3C found solutions for continuous motor control tasks typically within a few hours.

### Stated Limitations
The authors note that while asynchronous training removes the need for experience replay, incorporating it into the asynchronous framework could further improve data efficiency by reusing old data. This would be particularly beneficial in domains like TORCS, where environment interaction is more computationally expensive than model updates. Additionally, the authors suggest that the advantage function estimation could be improved using Generalized Advantage Estimation (GAE) or that the network could benefit from a dueling architecture.
