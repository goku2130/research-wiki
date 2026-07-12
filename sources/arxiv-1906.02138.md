---
id: arxiv:1906.02138
type: paper
title: Exploration with Unreliable Intrinsic Reward in Multi-Agent Reinforcement Learning
url: https://arxiv.org/abs/1906.02138
retrieved: '2026-07-12'
maturity: comprehensive
topic: entropy-and-exploration
---

# Exploration with Unreliable Intrinsic Reward in Multi-Agent Reinforcement Learning

### Core Problem
Directed exploration in multi-agent reinforcement learning (MARL) is hindered by three primary challenges: the exponential growth of joint-action spaces (making visitation counts infeasible), the difficulty of credit assignment when multiple agents cause an outcome, and partial observability, which reduces the reliability of count estimates. While intrinsic rewards based on uncertainty can accelerate learning, the authors find they are often "unreliable" in decentralized settings, creating detracting incentives that can prevent agents from converging to an optimal policy.

### Method: Centrally-Assisted Exploration (ICQL)
The authors propose a framework where a centralized agent is used exclusively during training to drive exploration, while decentralized agents (e.g., Independent Q-learning, IQL) learn the final policy from a shared replay buffer.

#### 1. The Central Agent
To avoid the computational cost of maximizing over the joint-action space, the central agent uses a COMA-style critic $q_{\psi}^{a}$ that conditions on the global state $s_t$, all other agents' actions $\mathbf{u}_t^{-a}$, and the agent's own previous action $u_{t-1}^{a}$. 

The framework employs a **local maximization (lmax)** procedure: it initializes joint actions $\mathbf{u}_t$ with the greedy actions of the decentralized agents and iteratively updates each $u_t^a$ to maximize $q_{\psi}^{a}$. To accelerate the transport of intrinsic rewards to earlier states, the authors use a $Q(\lambda)$ implementation. The target $G_{t}^{\lambda}$ is computed backwards:

$$
G_{t}^{\lambda} := r_{t} + (1-\lambda)\gamma \text{lmax}_{\bar{u}} q_{\psi^{\prime}}^{a}(\bar{u}^{a}|s_{t+1}, \bar{u}^{-a}, u_{t}^{a}) + \lambda\gamma G_{t+1}^{\lambda}
$$

The parameters $\psi$ are optimized by minimizing the expected squared error:

$$
\text{min}_{\psi} \mathbb{E} \left[ \sum_{t=0}^{T-1} \sum_{a=1}^{n} (G_{t}^{\lambda} - q_{\psi}^{a}(u_{t}^{a}|s_{t}, \mathbf{u}_{t}^{-a}, u_{t-1}^{a}))^2 \right]
$$

#### 2. Collaborative Intrinsic Reward
To handle large action spaces and evolving network parameters, the authors modify the standard uncertainty estimate:
*   **State-based Counts:** Instead of counting state-action pairs $N_t(s_t, u_t)$, they use a heuristic based on the count of the next state $N_t(s_{t+1})$.
*   **Collaborative Reward:** To prevent diverging incentives, all agents receive the same intrinsic reward, defined as the largest uncertainty estimate among all agents.
*   **Parameter Evolution:** To account for changing representations $\phi(x_t)$, they use an exponentially decaying average of the inverted matrix $\mathbf{C}_t$:

$$
\mathbf{C}_{t} := (1-\alpha)\mathbf{C}_{t-1} + \sum_{a=1}^{n} \phi_{(x_{t})}^{a} \phi_{(x_{t})}^{a\top}
$$

The final intrinsic reward $r_t^+$ incorporates a bias $b_t$ to reward only above-average novelty.

#### 3. Training Recipe
*   **Sampling:** To avoid out-of-distribution issues for the decentralized agents, the sampling process is shared: 50% of episodes are controlled by the intrinsically-rewarded central agent, and 50% by the decentralized IQL agents.
*   **Execution:** The central agent is discarded after training; only the decentralized agents are deployed.

### Key Quantitative Results
The method was tested on a partially observable predator-prey grid-world (41x10) with 4 agents hunting "valley" prey (reward 5) and "mountain" prey (reward 10).

*   **IQL (Baseline):** Slowest learning.
*   **IQL + Intrinsic Reward:** Accelerated initial learning but failed to find the optimal policy, as the intrinsic bonus distracted agents during convergence.
*   **ICQL:** Learned faster than both baselines and successfully converged to the optimal solution.
*   **Hyperparameters:** $\gamma=0.99$, learning rate $0.0005$, batch size 32, intrinsic reward magnitude $\sigma=1$, decay constant $\alpha=0.0002$, and constant bias $b_t=0.01$.

### Limitations
The authors state that intrinsic rewards are a "blessing and a curse"; while they enable directed exploration, they are inherently unreliable and can introduce detracting incentives. The proposed solution requires a centralized agent during training, and the use of $N_t(s_{t+1})$ as a proxy for state-action visitation is noted as a heuristic.
