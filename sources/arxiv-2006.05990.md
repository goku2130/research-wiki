---
id: arxiv:2006.05990
type: paper
title: Entropy Regularization in Deep Reinforcement Learning
url: https://arxiv.org/abs/2006.05990
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

**Core Problem**
On-policy reinforcement learning (RL) algorithms, particularly Proximal Policy Optimization (PPO), depend on numerous undocumented low- and high-level implementation decisions. These choices span loss functions, network architectures, normalization techniques, and hyperparameter configurations. The lack of transparent documentation obscures the true drivers of agent performance, severely hinders reproducibility, and impedes algorithmic progress by making it impossible to distinguish genuine methodological advances from implementation-level optimizations.

**Methodology and Experimental Recipe**
To systematically isolate the impact of these choices, the authors developed a unified on-policy RL framework based on SEED RL, exposing over 50 design decisions as explicit configuration options. The empirical study followed a structured recipe:
1. **Grouping:** Interacting choices were clustered into eight thematic categories (policy losses, network architecture, normalization/clipping, advantage estimation, training setup, timesteps, optimizers, regularization).
2. **Sampling:** Within each category, choices were randomly sampled from predefined ranges while all other settings were fixed to a competitive base configuration (a scaled PPOv2 setup with 256 parallel environments).
3. **Training:** More than 250,000 agents were trained across five continuous control environments (Hopper-v1, Walker2d-v1, HalfCheetah-v1, Ant-v1, Humanoid-v1) using MuJoCo 2.0.
4. **Evaluation:** Each configuration was trained with three independent seeds for one million steps (two million for Ant and Humanoid). Policies were evaluated every 100,000 steps by computing the average undiscounted return over 100 episodes. The final score was the median across seeds.
5. **Analysis:** Performance was assessed using two metrics: the conditional 95th percentile of performance with binomial confidence intervals, and the empirical distribution of choice values within the top 5% of configurations.

**Key Formulas**
The study formalizes core RL components used in the experiments. Advantage estimation relies on the N-step return:
$$\hat{V}_t^{(N)} = \sum_{i=t}^{t+N-1} \gamma^{i-t} r_i + \gamma^N V(s_{t+N})$$
and the Generalized Advantage Estimator (GAE):
$$\hat{V}_t^{\text{GAE}} = (1 - \lambda) \sum_{N>0} \lambda^{N-1} \hat{V}_t^{(N)}$$
The PPO policy loss enforces a trust region via clipping:
$$\mathcal{L}_{\text{PPO}}^\epsilon = -\min \left[ \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)} \hat{A}_t^\pi, \text{clip} \left( \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}, \frac{1}{1+\epsilon}, 1+\epsilon \right) \hat{A}_t^\pi \right]$$
Action distributions are parameterized as:
$$T_u(\mathcal{N}(x_\mu, T_\rho(x_\rho + c_\rho) + \epsilon_\rho))$$
where $T_u$ and $T_\rho$ denote transformations for bounding actions and ensuring non-negative standard deviations, respectively.

**Key Quantitative Results**
PPO outperformed alternative policy losses (PG, V-trace, AWR, V-MPO, RPA) on four of five environments, with a clipping threshold $\epsilon$ of 0.2 or 0.3 proving robust. Architecturally, separating value and policy networks improved performance on four environments. Initializing the final policy layer with weights scaled down by 100× increased Humanoid performance by 66%. An initial action standard deviation of 0.5 yielded optimal results across most tasks, and applying a $\tanh$ transformation to bounded actions improved HalfCheetah performance by 30%. Input normalization was critical for four environments, whereas value function normalization significantly aided HalfCheetah and Humanoid but degraded Walker2d. GAE with $\lambda = 0.9$ surpassed N-step returns, while PPO-style value loss clipping and Huber loss consistently reduced performance. Multi-pass data optimization was essential, with recomputing advantages at the start of each pass yielding the best sample efficiency. The discount factor $\gamma = 0.99$ and Adam optimizer with learning rate $3 \times 10^{-4}$ and momentum $0.9$ were highly robust. Notably, policy regularization (entropy bonuses or KL constraints) provided negligible benefits across environments except HalfCheetah.

**Stated Limitations**
The study is strictly confined to on-policy RL for continuous control tasks within OpenAI Gym v1, limiting generalizability to off-policy methods or discrete domains. Strong interactions between choices (e.g., learning rate and minibatch size) necessitate joint tuning, meaning random sampling within groups may obscure optimal configurations. Sub-optimal hyperparameter settings can cause complete training failure, potentially biasing performance distributions. Additionally, minor MuJoCo physics simulator version differences can subtly alter environment dynamics. The authors also note that the ineffectiveness of regularization may be specific to their experimental setup, as PPO’s inherent trust-region constraints and precise policy initialization likely render explicit entropy or KL penalties redundant.
