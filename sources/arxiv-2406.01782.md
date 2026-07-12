---
id: arxiv:2406.01782
type: paper
title: Multi-agent assignment via state augmented reinforcement learning
url: https://arxiv.org/abs/2406.01782
retrieved: '2026-07-12'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Multi-Agent Assignment via State Augmented Reinforcement Learning

## Core Problem
The authors address the multi-agent assignment problem where a team of $N$ agents must collectively satisfy $M$ conflicting task requirements. Specifically, the goal is to ensure that $M$ different regions of a state space ($\mathcal{S}_m$) are visited with a minimum average frequency $c_m$. Standard reinforcement learning (RL) regularization techniques, which weight rewards to balance conflicting goals, often fail to produce feasible policies or converge in these scenarios. The challenge is exacerbated when $\sum c_m > 1$, meaning a single agent cannot satisfy the requirements alone, necessitating coordination among agents who may only have access to local state information.

## Method
The proposed solution utilizes a state-augmented constrained RL approach where Lagrangian multipliers $\lambda$ are treated as states of an augmented Markov Decision Process (MDP). Rather than seeking a static dual optimum, the system allows $\lambda$ to oscillate, inducing policies that alternate between tasks to satisfy constraints sequentially.

### Step-by-Step Recipe
1.  **Offline Training**: 
    *   A family of policies $\pi_{\theta_n}(A_{tn} | S_{tn}, \lambda)$ is learned for each agent $n$.
    *   The reward is parameterized by $\lambda$ to create a regularized objective.
    *   Policies are optimized using a distributed policy gradient method where each agent estimates a local state-action value function $Q_{n, \lambda}^{\pi_\theta}$ based on global rewards.
2.  **Distributed Online Execution**:
    *   **Stochastic Dual Updates**: Agents update local copies of the multipliers $\lambda_{m,n}$ using stochastic gradient descent over a finite rollout horizon $T_0$.
    *   **Gossiping Protocol**: To eliminate the need for a central controller or global state access, agents use a gossiping framework. They exchange binary reward indicators $\mathbb{1}[S_{tn} \in \mathcal{S}_m]$ with neighbors over a communication network (graph $\mathcal{G}$).
    *   **Consensus**: Through local exchanges, agents reach a consensus on whether a zone was occupied at time $\tau$, allowing them to compute accurate dual gradients.
    *   **Policy Execution**: Agents execute the pre-trained policy $\pi_{\theta_n}$ using their current local multiplier $\lambda_{m,n}$.

## Key Formulas
The team reward for task $m$ is defined as:

$$
r_m(S_t, A_t) = \max_{n=1, \dots, N} \mathbb{1}[S_{tn} \in \mathcal{S}_m]
$$

The Lagrangian associated with the feasibility problem is:

$$
\mathcal{L}(\pi, \lambda) = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}_{S_t, A_t \sim \pi} \left[ \sum_{t=0}^{T-1} \sum_{m=1}^M \lambda_m (r_m(S_t, A_t) - c_m) \right]
$$

The distributed policy gradient for agent $n$ is given by:

$$
\nabla_{\theta_n} \mathcal{L}(\pi_\theta, \lambda) = \mathbb{E}_{S_n, A_n \sim \rho} \left[ \nabla_{\theta_n} \log \pi_{\theta_n}(A_n | S_n, \lambda) Q_{n, \lambda}^{\pi_\theta}(S_n, A_n) \right]
$$

The theoretical condition for almost sure feasibility (Theorem 4) is:

$$
\beta + \frac{M}{T_0} d(\mathcal{G}) \eta + \epsilon_{T_0} + \eta/2 < \delta_C
$$

where $d(\mathcal{G})$ is the graph diameter, $\eta$ is the step-size, $\beta$ represents representation error, and $\delta_C = (1 - c_{\max}) / \sqrt{M}$.

## Quantitative Results
The method was validated through several numerical experiments:
*   **Simulated Monitoring**: $N=2$ agents monitoring $M=4$ zones with requirements $c = [0.3, 0.3, 0.3, 0.3]$. Despite the total requirement (1.2) exceeding the capacity of a single agent (1.0), the system achieved feasibility.
*   **Training/Execution Parameters**: Policies were trained over 150,000 Actor-Critic episodes and executed for 200,000 time steps with a rollout horizon $T_0 = 1,000$ and a communication range of 2.5 meters.
*   **Realistic Environments**: 
    *   **Gazebo**: Implemented via ROS2 with two TurtleBot robots including collision avoidance.
    *   **Floorplan Patrol**: $N=2$ agents assigned to $M=3$ zones. The system successfully satisfied constraints $c = [0.25, 0.25, 0.25, 0.25]$ and $c = [0.5, 0.4, 0.3]$, which were unattainable by a single agent.

## Limitations
The theoretical guarantees of the protocol rely on three primary assumptions:
1.  **Estimation Consistency**: The truncation error $\epsilon_{T_0}$ must vanish as the horizon $T_0 \to \infty$.
2.  **No Repelling Forces**: The MDP must allow at least one feasible policy where an agent can be stationed at any particular zone $\mathcal{S}_m$ to ensure $r_m = 1$.
3.  **Representation Capacity**: The policy parameterization must be sufficiently dense (controlled by $\beta$) to approximate the optimal policies $\Pi(\lambda)$.
