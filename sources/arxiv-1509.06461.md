---
id: arxiv:1509.06461
type: paper
title: Human-level control through deep reinforcement learning
url: https://arxiv.org/abs/1509.06461
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

**Core Problem**
Q-learning suffers from systematic overestimation of action values due to the maximization operator selecting the maximum of inaccurate estimates. This upward bias arises whenever value estimates contain error, regardless of whether the error stems from environmental noise, function approximation, or non-stationarity. The authors demonstrate that Deep Q-Networks (DQN) exhibit substantial overestimations across all 49 tested Atari 2600 games. Because these overestimations are non-uniform and propagate through bootstrapping, they distort relative state values, degrade policy quality, and cause training instability.

**Method/Recipe Step by Step**
Double DQN generalizes Double Q-learning to deep function approximation by decoupling action selection from action evaluation, utilizing DQN's existing architecture without adding networks. The training procedure follows these steps:
1. Sample a transition $(S_t, A_t, R_{t+1}, S_{t+1})$ uniformly from the experience replay buffer.
2. Use the online network with parameters $\boldsymbol{\theta}_{t}$ to select the greedy action in the next state: $a^* = \underset{a}{\text{argmax}} Q(S_{t+1}, a; \boldsymbol{\theta}_{t})$.
3. Evaluate this selected action using the frozen target network with parameters $\boldsymbol{\theta}_{t}^{-}$ to compute the target value.
4. Update $\boldsymbol{\theta}_{t}$ via stochastic gradient descent to minimize the squared error between the current estimate and the target.
5. Periodically copy the online network parameters to the target network ($\boldsymbol{\theta}_{t}^{-} \leftarrow \boldsymbol{\theta}_{t}$) every $\tau$ steps, keeping the target network fixed between copies.

**Key Formulas**
The fundamental distinction lies in the target calculation. Standard Q-learning and DQN targets couple selection and evaluation:

$$
Y_t^Q \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t)
$$

$$
Y_t^{\text{DQN}} \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t^-)
$$

Double Q-learning decouples these using two independent weight sets:

$$
Y_t^{\text{DoubleQ}} \equiv R_{t+1} + \gamma Q(S_{t+1}, \underset{a}{\text{argmax}} Q(S_{t+1}, a; \theta_t); \theta_t')
$$

Double DQN replaces the second weight set with the target network:

$$
Y_t^{\mathrm{DoubleDQN}} \equiv R_{t+1} + \gamma Q(S_{t+1}, \underset{a}{\text{argmax}} Q(S_{t+1}, a; \boldsymbol{\theta}_{t}), \boldsymbol{\theta}_{t}^{-})
$$

The authors also derive a theoretical lower bound for Q-learning overestimation under unbiased but inaccurate estimates:

$$
\max_a Q_t(s, a) \ge V_*(s) + \sqrt{\frac{C}{m-1}}
$$

where $C$ is the mean squared estimation error and $m$ is the action count. Performance is normalized as:

$$
\text{score}_{\text{normalized}} = \frac{\text{score}_{\text{agent}} - \text{score}_{\text{random}}}{\text{score}_{\text{human}} - \text{score}_{\text{random}}}
$$

**Key Quantitative Results**
Double DQN consistently reduces overestimation and improves policy quality. Under a standard no-op evaluation (5 minutes), Double DQN achieves a median normalized score of 114.7% and a mean of 330.3%, surpassing DQN's 93.5% and 241.1%. Under a challenging human-start evaluation (30 minutes), Double DQN reaches a median of 88.4% and mean of 273.1%, while a tuned variant achieves 116.7% and 475.2%. Significant per-game improvements include Road Runner (233% to 617%), Asterix (70% to 180%), Zaxxon (54% to 111%), and Double Dunk (17% to 397%). The network uses three convolutional layers and one fully connected layer (~1.5M parameters), trained with $\gamma=0.99$, $\alpha=0.00025$, a 1M-transition replay buffer, batch size 32, and target updates every 10,000 steps.

**Stated Limitations**
The authors note several constraints. First, the primary evaluation uses hyperparameters originally tuned for DQN, making the comparison somewhat adversarial to Double DQN. Second, improvements are not universal; several games (e.g., Alien, Amidar, Asteroids) show marginal or negative changes, indicating context-dependent efficacy. Third, while the theoretical lower bound on overestimation decreases as the action count $m$ increases, empirical results show bias typically grows with $m$. Finally, the paper distinguishes this estimation-induced overoptimism from constructive "optimism in the face of uncertainty" exploration bonuses, emphasizing that the former is a harmful bias that propagates through bootstrapping and degrades learned policies.
