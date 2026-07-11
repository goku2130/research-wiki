---
id: arxiv:1509.02971
type: paper
title: Continuous control with deep reinforcement learning
url: https://arxiv.org/abs/1509.02971
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

**Core Problem**
Deep Q-Learning (DQN) successfully handles high-dimensional observation spaces but cannot operate in continuous action domains. Discretizing continuous actions induces a curse of dimensionality, exponentially increasing the action space and rendering exploration intractable. Conversely, directly maximizing a continuous action-value function requires slow, iterative optimization at every timestep. The authors address the need for a model-free, off-policy algorithm capable of learning high-dimensional continuous control policies directly from raw sensory inputs without manual discretization or domain-specific tuning.

**Method/Recipe Step by Step**
The proposed Deep Deterministic Policy Gradient (DDPG) algorithm implements an actor-critic architecture with four stability mechanisms adapted from DQN:
1. **Actor-Critic Architecture:** A deterministic actor network $\mu(s|\theta^\mu)$ maps states to continuous actions, while a critic network $Q(s,a|\theta^Q)$ estimates the action-value function.
2. **Replay Buffer:** Transitions $(s_t, a_t, r_t, s_{t+1})$ are stored in a finite buffer $\mathcal{R}$ of size $10^6$. Minibatches are sampled uniformly to break temporal correlations and enable efficient hardware utilization.
3. **Target Networks:** Separate target networks $Q'$ and $\mu'$ compute stable training targets. Their weights are slowly tracked via soft updates rather than hard copies to prevent divergence.
4. **Batch Normalization:** Applied to state inputs and all network layers prior to action inputs to normalize feature scales across varying physical units, eliminating manual scaling.
5. **Exploration:** An Ornstein-Uhlenbeck noise process $\mathcal{N}$ is added to the actor's output to generate temporally correlated exploration, improving efficiency in physical systems with inertia.
Training proceeds by sampling a minibatch, computing bootstrapped targets, minimizing the critic's loss, updating the actor via the deterministic policy gradient, and performing soft target network updates.

**Key Formulas**
The critic is trained by minimizing the mean squared error:

$$
L = \frac{1}{N} \sum_i \left( y_i - Q(s_i, a_i|\theta^Q) \right)^2
$$

where the target values are computed as $y_i = r_i + \gamma Q'(s_{i+1}, \mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})$. The actor parameters are updated using the deterministic policy gradient:

$$
\nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_i \nabla_a Q(s, a|\theta^Q)\big|_{s=s_i, a=\mu(s_i)} \nabla_{\theta^\mu} \mu(s|\theta^\mu)\big|_{s_i}
$$

Target network weights are updated softly at each step: $\theta' \leftarrow \tau\theta + (1 - \tau)\theta'$ with $\tau \ll 1$. The exploration policy is defined as $\mu'(s_t) = \mu(s_t|\theta_t^\mu) + \mathcal{N}$.

**Key Quantitative Results and Numbers**
DDPG was evaluated across more than 20 simulated physics tasks (MuJoCo) and a racing simulator (Torcs), using both low-dimensional state vectors and raw 64x64 pixel inputs. Using identical architectures and hyperparameters (Adam optimizer, actor LR $10^{-4}$, critic LR $10^{-3}$, $\gamma=0.99$, $\tau=0.001$, OU noise $\theta=0.15, \sigma=0.2$), the algorithm robustly solved all tasks. Performance was normalized against a random policy (0) and an iLQG planning baseline (1). DDPG frequently matched or exceeded the planner, achieving normalized scores above 1.0 in multiple environments (e.g., $R_{best,loud}=1.990$ for hardCheetah and $R_{best,pix}=2.225$ for blockworld3da). The algorithm converged within at most 2.5 million experience steps, approximately 20 times fewer steps than required for DQN on Atari tasks. Network architectures contained approximately 130,000 parameters for low-dimensional inputs and 430,000 for pixel inputs.

**Stated Limitations**
The authors explicitly note that, like most model-free reinforcement learning methods, DDPG requires a substantial number of training episodes to converge, highlighting data inefficiency as a primary constraint. Furthermore, the use of non-linear function approximators eliminates theoretical convergence guarantees. Empirical analysis revealed that while Q-value estimates are accurate in simple domains, they become biased and less precise in complex tasks, though competent policies are still learned. Additionally, the exploration noise process requires environment-specific tuning, particularly when dealing with vastly different temporal scales across domains.
