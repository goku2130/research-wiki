---
id: arxiv:1801.01290
type: paper
title: 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
  with a Stochastic Actor'
url: https://arxiv.org/abs/1801.01290
retrieved: '2026-07-12'
maturity: comprehensive
topic: policy-gradient-methods
---

The paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" by Haarnoja et al. (2018) introduces a novel model-free deep reinforcement learning algorithm called Soft Actor-Critic (SAC).

**Core Problem:**
Model-free deep reinforcement learning (RL) algorithms typically suffer from high sample complexity and brittle convergence properties, requiring meticulous hyperparameter tuning. These issues limit their applicability to complex, real-world domains, especially in continuous state and action spaces. Existing off-policy methods like DDPG are often unstable and sensitive to hyperparameters, while on-policy methods have poor sample efficiency. Prior maximum entropy RL methods formulated as Q-learning often require complex approximate inference procedures in continuous action spaces.

**Method/Recipe Step by Step:**
Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework. It aims to maximize expected reward while also maximizing the entropy of the policy, encouraging exploration and robustness.

1.  **Maximum Entropy Objective:** The algorithm optimizes a modified objective function that includes an entropy term:

$$
J(\pi) = \sum_{t=0}^{T} E_{(s_i,a_i) \sim \rho_\pi}[r(s_i,s_a) + \alpha H(\pi \cdot |s_i)].
$$

    For infinite horizon problems with discount factor $\gamma$:

$$
J(\pi) = \sum_{t=0}^{\infty} \mathbb{E}_{(s_t, a_t) \sim \rho_t} \left[ \sum_{t=1}^{\infty} \gamma^{l-t} \mathbb{E}_{s_t, a_t} + \alpha H(\pi(\cdot | s_t))| s_t, a_t| \right].
$$

    The temperature parameter $\alpha$ balances reward maximization and policy stochasticity.

2.  **Soft Policy Iteration (Theoretical Foundation):**
    *   **Soft Policy Evaluation:** Iteratively compute the soft Q-value for a fixed policy $\pi$ using a modified Bellman backup operator $T^\pi$:

$$
\mathcal{T}^\pi Q(s_i, a_t) \triangleq r(s_i, a_t) + \gamma E_{a_{t+1}}[V(s_i+1)].
$$

        where $V(s_i)$ is the soft state value function:

$$
V(s_i) = E_{a_{i-1}}[Q[s_i, a_t] - \log \pi(a_i|s_i)].
$$

    *   **Soft Policy Improvement:** Update the policy by minimizing the KL-divergence to an exponential of the Q-function, projecting into a tractable policy set $\Pi$:

$$
\pi_{\text{new}} = \arg \min_{\pi \in II} D_{\text{KL}}\left(\pi'(\cdot |s_i) \left\| \frac{\exp(Q^{\pi_{\text{old}}}(s_i, \cdot))}{Z^{\pi_{\text{old}}}(s_i)}\right).
$$

        The partition function $Z^{\pi_{\text{old}}}(s_i)$ normalizes the distribution and can be ignored for gradient calculation.

3.  **Soft Actor-Critic (Practical Algorithm):** Approximates soft policy iteration using function approximators (neural networks) for the state value function $V_\psi(s)$, soft Q-function $Q_\theta(s, a)$, and policy $\pi_\phi(a|s)$.
    *   **Value Function Update:** The state value function $V_\psi(s)$ is trained to minimize the squared residual error:

$$
J_V(\psi) = E_{u \sim D} \left[ \frac{1}{2} \left( V_\psi(s) - E_{a \sim \pi_\phi} \left( Q_\theta(s, a) - \log \pi_\phi(a|s) \right) \right)^2 \right].
$$

        The gradient is estimated as:

$$
\nabla_\psi J_V(\psi) = \nabla_\psi V_\psi(s)(V_\psi(s) - Q_\theta(s, a) + \log \pi_\phi(a|s)).
$$

    *   **Q-Function Update:** The soft Q-function $Q_\theta(s, a)$ is trained to minimize the soft Bellman residual using two Q-functions (for bias mitigation):

$$
J_Q(\theta) = E_{(s,a,r,s') \sim D} \left[ \frac{1}{2} \left( Q_\theta(s, a) - \hat{Q}(s, a) \right)^2 \right],
$$

        with the target $\hat{Q}(s, a) = r(s, a) + \gamma V_{\bar{\psi}}(s')$, where $V_{\bar{\psi}}$ is a target value network (exponentially moving average of $V_\psi$). The gradient is:

$$
\nabla_\theta J_Q(\theta) = \nabla_\theta Q_\theta(s, a)(Q_\theta(s, a) - r(s, a) - \gamma V_{\bar{\psi}}(s')).
$$

        The minimum of the two Q-functions is used for $V_\psi$ and policy updates.
    *   **Policy Update:** The policy $\pi_\phi(a|s)$ is updated by minimizing the expected KL-divergence, using the reparameterization trick $a = f_\phi(\epsilon, s)$:

$$
J_\pi(\phi) = E_{(s,\epsilon) \sim D} \left[ \log \pi_\phi(f_\phi(\epsilon, s)|s) - Q_\theta(s, f_\phi(\epsilon, s)) \right].
$$

        The gradient is approximated as:

$$
\nabla_\phi J_\pi(\phi) = \nabla_\phi \log \pi_\phi(a|s) + (\nabla_\phi \log \pi_\phi(a|s) - \nabla_\phi Q_\theta(s, a)) \nabla_\phi f_\phi(\epsilon, s).
$$

    *   **Action Bounding:** For continuous action spaces, an invertible squashing function (e.g., $\tanh$) is applied to Gaussian samples, and the change of variables formula is used to compute log-likelihoods:

$$
\log \pi(\mathbf{a}|s) = \log \mu(\mathbf{u}|s) - \sum_{i=1}^{D} \log \left(1 - \tanh^2(u_i)\right).
$$

**Algorithm 1 (Soft Actor-Critic):**
1.  Initialize parameter vectors $\psi, \bar{\psi}, \theta_1, \theta_2, \phi$.
2.  For each iteration:
    *   Collect environment steps: $a \sim \pi_\phi(a|s)$, $s' \sim p(s'|s, a)$, add $(s, a, r, s')$ to replay buffer $D$.
    *   For each gradient step:
        *   Update value function parameters $\psi$ using $\nabla_\psi J_V(\psi)$.
        *   Update Q-function parameters $\theta_1, \theta_2$ using $\nabla_{\theta_i} J_Q(\theta_i)$.
        *   Update policy parameters $\phi$ using $\nabla_\phi J_\pi(\phi)$.
        *   Update target value network parameters $\bar{\psi} \leftarrow \tau \psi + (1-\tau) \bar{\psi}$.

**Key Quantitative Results and Numbers:**
*   SAC achieves state-of-the-art performance on continuous control benchmark tasks (OpenAI Gym, rllab Humanoid).
*   Outperforms DDPG, PPO, and Soft Q-Learning (SQL) in terms of learning speed and final performance, especially on harder tasks like Ant-v1, Humanoid-v1, and Humanoid (rllab).
*   DDPG fails on Ant-v1, Humanoid-v1, and Humanoid (rllab).
*   SAC learns considerably faster than PPO.
*   SAC is faster and has better asymptotic performance than SQL.
*   **Hyperparameters (common):** Adam optimizer, learning rate $3 \cdot 10^{-4}$, discount $\gamma=0.99$, replay buffer size $10^6$, 2 hidden layers with 256 units, ReLU nonlinearity.
*   **SAC specific hyperparameters:** target smoothing coefficient $\tau=0.005$, target update interval 1, gradient steps 1.
*   **Reward Scale (environment specific):** Hopper-v1 (5), Walker2d-v1 (5), HalfCheetah-v1 (5), Ant-v1 (5), Humanoid-v1 (20), Humanoid (rllab) (10).
*   **Stability:** SAC shows significantly higher stability across different random seeds compared to a deterministic variant, exhibiting much lower variability in performance.

**Stated Limitations:**
*   **Reward Scale Sensitivity:** SAC is particularly sensitive to the scaling of the reward signal, which acts as the inverse temperature for the optimal policy's stochasticity. This hyperparameter needs to be tuned for each task.
*   **Computational Cost:** While SAC is sample-efficient, variants with hard target updates and multiple gradient steps between environment steps can increase computational cost.
*   **Theoretical Tractability:** The exact form of soft policy iteration is only tractable in the tabular case; for continuous domains, approximations with function approximators are necessary.
*   **Policy Parameterization:** The algorithm is agnostic to policy parameterization as long as it can be evaluated for any state-action tuple. The convergence proof holds for a given policy class.
