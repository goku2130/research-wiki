---
id: arxiv:2412.14135
type: paper
title: 'Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement
  Learning Perspective'
url: https://arxiv.org/abs/2412.14135
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

**Core Problem**
The paper addresses the challenge of reproducing OpenAI o1’s expert-level reasoning capabilities, which scale with both training and inference compute via reinforcement learning (RL). While recent efforts attempt to imitate o1’s reasoning style through knowledge distillation, these approaches are fundamentally constrained by the teacher model’s capability ceiling. The authors propose a structured RL roadmap to systematically achieve o1-like performance by leveraging the theoretically scalable power of search and learning, shifting AI paradigms from static supervised learning to dynamic, compute-scalable reasoning.

**Methodological Roadmap**
The proposed framework integrates four sequential, interdependent components:
1. **Policy Initialization:** Establishes a strong baseline through large-scale pre-training on web corpora, followed by instruction fine-tuning to transform next-token prediction into purposeful behavior. Crucially, it injects six human-like reasoning behaviors—problem analysis, task decomposition, task completion, alternative proposal, self-evaluation, and self-correction—via supervised fine-tuning or prompt engineering to enable systematic solution space exploration.
2. **Reward Design:** Provides dense, effective guidance for search and learning. It transitions from sparse outcome rewards to dense process rewards using reward shaping or learns reward models from preference/expert data. This component ensures the agent receives actionable feedback even when environmental signals are unavailable or overly coarse.
3. **Search:** Generates high-quality trajectories during both training and inference. It employs internal guidance (model uncertainty, self-evaluation) and external guidance (environmental feedback, heuristics) alongside tree search (Best-of-N, Beam Search, MCTS) or sequential revision strategies. Training-time search efficiently samples parallel candidates, while test-time search iteratively refines outputs through reflection and correction.
4. **Learning:** Updates the policy using search-generated data via policy gradients or behavior cloning. By learning from environment interactions rather than costly human annotations, this step enables continuous policy improvement and unlocks the potential for superhuman performance.

**Key Formulas**
The framework formalizes reward design and value estimation using two core equations. Potential-based reward shaping ensures policy invariance while transforming sparse outcome rewards into dense process rewards:

$$
F(s_t, a_t) = r(s_t, a_t) + \gamma\phi(s_{t+1}) - \phi(s_t)
$$

where $F$ and $r$ are reward functions, $\gamma$ is the discount factor, and $\phi$ is a potential function. Additionally, the value function quantifies long-term expected returns to guide search and policy updates:

$$
v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s \right]
$$

where $v_\pi(s)$ represents the state value under policy $\pi$, and $G_t$ is the discounted cumulative reward.

**Quantitative Context**
As a conceptual roadmap, the paper does not introduce original empirical experiments or novel quantitative results. It synthesizes scaling observations from prior literature, noting that Brown et al. (2024) observed increasing sample counts improve coverage, with small models potentially approaching near 100% pass@1 accuracy on MATH given sufficient scaling. Snell et al. (2024) further established that search computation scales effectively with model size and token sampling. The authors emphasize that o1’s performance consistently improves with increased RL and inference compute, though they caution that unmitigated search scaling can trigger inverse scaling phenomena.

**Stated Limitations**
The authors identify critical challenges across all roadmap components. Policy initialization struggles with balancing sampling efficiency against exploration diversity and ensuring domain-general reasoning behaviors beyond specialized tasks like math or coding. Reward design faces distribution shift when proxy reward models evaluate out-of-distribution trajectories, alongside difficulties in defining fine-grained rewards for variable action granularities and selecting appropriate feedback data for complex tasks. Search limitations include inverse scaling due to policy drift, the risk of computational waste on simple queries, unresolved trade-offs between tree search and sequential revisions under fixed compute budgets, and hardware efficiency bottlenecks in autoregressive generation and parallel search algorithms.
