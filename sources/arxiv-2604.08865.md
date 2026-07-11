---
id: arxiv:2604.08865
type: paper
title: 'SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks'
url: https://arxiv.org/html/2604.08865v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

**Core Problem**
Standard token-level Proximal Policy Optimization (PPO) exhibits severe instability when aligning Large Language Models (LLMs) for long-horizon Chain-of-Thought (CoT) reasoning. The Generalized Advantage Estimation (GAE) mechanism struggles to propagate sparse terminal rewards across thousands of tokens, inducing high bias. Furthermore, token-level critics suffer from a "tail effect," where they overfit to semantic cues near the sequence end, causing advantage signals to vanish prematurely for correct trajectories or fail to penalize intermediate errors. While critic-free alternatives like Group Relative Policy Optimization (GRPO) bypass token-level noise by treating responses as atomic actions, they introduce high gradient variance and prohibitive computational latency, as they require sampling multiple responses ($N > 1$) per prompt to construct empirical baselines.

**Methodology and Recipe**
SPPO resolves this bias-variance trade-off by explicitly reformulating reasoning as a Sequence-Level Contextual Bandit (SL-CB). The optimization recipe proceeds as follows:
1. **Context-Action Mapping:** Collapse the temporal horizon to $H=1$. The prompt $s_p$ serves as the static context, and the entire generated response sequence $a_{seq}$ is treated as a single atomic action.
2. **Scalar Value Training:** Train a lightweight critic $V_\phi(s_p)$ to estimate the scalar probability of success (solvability) for a given prompt. The critic is optimized using a Binary Cross-Entropy (BCE) loss against binary rewards $R \in \{0,1\}$.
3. **Advantage Computation:** Derive a low-variance advantage signal by subtracting the scalar baseline from the outcome: $A(s_p, a) = R - V_\phi(s_p)$. This amplifies signals when the model is confident but wrong, and suppresses noise when uncertain.
4. **Uniform Broadcast:** Propagate the single sequence-level advantage $A(s_p, a)$ uniformly to all constituent tokens $t \in a$. This eliminates temporal credit assignment ambiguity by reinforcing or penalizing every step in the chain equally based on the final outcome.
5. **Policy Update:** Optimize the policy $\pi_\theta$ using PPO’s clipped surrogate objective, substituting the time-dependent advantage with the broadcasted sequence-level advantage. The algorithm operates with single-sample efficiency ($N=1$), enabling high-throughput updates.
6. **Decoupled Critic Architecture:** Optionally deploy a smaller critic (e.g., 1.5B parameters) to align a larger policy (e.g., 7B), leveraging the reduced complexity of scalar solvability estimation to minimize memory overhead.

**Key Formulas**
The mathematical core of SPPO is defined by:

$$
A(s_p, a) = R - V_\phi(s_p)
$$


$$
L_V(\phi) = -\mathbb{E}\left[R \log V_\phi(s_p) + (1 - R) \log(1 - V_\phi(s_p))\right]
$$


$$
J_{\mathrm{SPPO}}(\theta) = \mathbb{E}_{s_p, a, t}\left[\min\left(r_t(\theta)A(s_p, a), \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A(s_p, a)\right)\right]
$$

where $r_t(\theta)$ is the probability ratio between current and behavior policies, and $\epsilon$ is the clipping hyperparameter.

**Quantitative Results**
Evaluated on DeepSeek-R1-Distill-Qwen models across AIME24/25, AMC23, MATH500, and Minerva Math, SPPO consistently outperforms baselines. On the 1.5B scale, SPPO achieves an Average@16 accuracy of 48.06, surpassing GRPO ($N=8$) at 47.08. On the 7B scale, SPPO reaches 58.11, while a decoupled variant (7B policy + 1.5B critic) achieves 58.56. Computationally, SPPO converges to peak performance in approximately 22 hours, delivering a $5.9\times$ training speedup over GRPO. The decoupled critic architecture reduces peak VRAM usage by 12.8% compared to symmetric actor-critic setups. In controlled RLVR benchmarks (e.g., MountainCar, Hopper), SPPO successfully solves long-horizon sparse-reward tasks where standard PPO flatlines.

**Limitations**
The methodology is explicitly constrained to Reinforcement Learning with Verifiable Rewards (RLVR) tasks requiring objective ground-truth verifiers to estimate prompt solvability. Extending the sequence-level bandit formulation to open-ended generation tasks lacking verifiable outcomes remains a direction for future research. Additionally, applying this framework to subjective or ethical domains without robust reward modeling risks amplifying base model biases or generating plausible but factually incorrect reasoning chains.
