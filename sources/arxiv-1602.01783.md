---
id: arxiv:1602.01783
type: paper
title: Asynchronous Methods for Deep Reinforcement Learning
url: https://arxiv.org/abs/1602.01783
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

**Core Problem**
Deep reinforcement learning historically faced fundamental instability when combining simple online algorithms with deep neural networks, primarily due to non-stationary data streams and highly correlated temporal updates. Previous stabilization techniques relied on experience replay, which restricts algorithms to off-policy learning, increases memory and computational overhead, and necessitates specialized hardware like GPUs or distributed architectures. The authors identify the need for a lightweight, stable paradigm that enables on-policy and off-policy methods to train deep controllers efficiently without replay buffers or massive parallel infrastructure.

**Method/Recipe Step by Step**
The proposed asynchronous framework replaces experience replay with parallel actor-learner threads operating on a single multi-core CPU. The training cycle for each thread follows a deterministic sequence: (1) initialize a local parameter copy $\theta'$ and a step counter $t$; (2) interact with an independent environment copy, selecting actions via a thread-specific exploration policy; (3) receive rewards and next states, accumulating gradients over $t_{max}$ steps (or until episode termination); (4) asynchronously apply the accumulated gradients to the shared global parameter vector $\theta$ using lock-free, Hogwild!-style updates; and (5) periodically synchronize local parameters with the global model and update target networks at fixed intervals ($I_{target}$). The framework implements four variants: asynchronous one-step Q-learning, one-step Sarsa, n-step Q-learning, and asynchronous advantage actor-critic (A3C). A3C maintains a policy network and a value network with shared hidden layers, updating both using n-step returns in a forward-view manner. Optimization employs a shared RMSProp variant where moving averages of squared gradients are synchronized across threads to enhance stability. Entropy regularization is added to the policy objective to discourage premature convergence.

**Key Formulas**
The one-step Q-learning loss is defined as:
$$L_i(\theta_i) = \mathbb{E} \left( r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i) \right)^2$$
The shared RMSProp optimizer updates parameters elementwise using:
$$g = \alpha g + (1 - \alpha) \Delta \theta^2 \quad \text{and} \quad \theta \leftarrow \theta - \eta \frac{\Delta \theta}{\sqrt{g + \epsilon}}$$
For A3C, the policy gradient incorporates the advantage function $A(s_t, a_t) = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k}; \theta_v) - V(s_t; \theta_v)$, yielding the entropy-regularized update:
$$\nabla_{\theta'} \log \pi(a_t|s_t; \theta')(R_t - V(s_t; \theta_v)) + \beta \nabla_{\theta'} H(\pi(s_t; \theta'))$$
where $\beta$ controls the regularization strength and $H$ denotes policy entropy.

**Key Quantitative Results**
Evaluated across 57 Atari 2600 games, A3C trained on 16 CPU cores for four days achieved a mean human-normalized score of 623.0% and a median of 112.6%, surpassing Prioritized DQN (463.6% mean), Gorila (215.2% mean), and DQN (121.9% mean) while requiring half the training time and zero GPU usage. After just one day of training, A3C matched Dueling DQN’s mean score of 343.8%. The framework demonstrates strong scalability, with 16 parallel threads yielding approximately 24.1× speedup for one-step Q-learning and 12.5× for A3C. On the TORCS racing simulator, A3C reached 75–90% of human performance in roughly 12 hours. In continuous control tasks via MuJoCo, A3C solved domains in under 24 hours, and on a novel 3D maze navigation task (Labyrinth), an LSTM-based A3C agent achieved an average score of approximately 50 using only visual input.

**Stated Limitations**
The omission of experience replay inherently reduces data efficiency, as each environment interaction is consumed only once before discarding. The authors note that integrating replay buffers could significantly improve data reuse, particularly in computationally expensive domains like TORCS. Additionally, the n-step implementations operate in a forward-view rather than the more conventional backward-view with eligibility traces, which may limit compatibility with certain optimization techniques. While the single-machine CPU approach proves highly efficient and outperforms GPU-based baselines in training speed, the framework does not currently leverage distributed parameter servers or GPU acceleration. Finally, the stability of value-based methods without replay remains dependent on the diversity of exploration policies and shared RMSProp statistics, which may not generalize identically across all task distributions.
