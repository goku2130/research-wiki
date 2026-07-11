---
id: arxiv:2512.04302
type: paper
title: Towards better dense rewards in Reinforcement Learning Applications
url: https://arxiv.org/html/2512.04302v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

**Core Problem**
The source addresses the fundamental challenge of designing dense, accurate reward signals in reinforcement learning (RL). In goal-conditioned hierarchical RL (GCHRL), conventional methods rely on Euclidean distance between current states and assigned subgoals, which misaligns with true transition dynamics and yields misleading intermediate rewards. In reinforcement learning from human feedback (RLHF), reward sparsity—where feedback is exclusively provided upon sequence completion—severely hampers temporal credit assignment and learning efficiency. Additionally, continual RL agents struggle to transfer structured knowledge across environments with similar underlying dynamics.

**Graph-Guided subGoal Representation Generation (G4RL) Recipe**
1. **Graph Construction & Updating:** Maintain an undirected state graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with $N$ nodes. During exploration, if a new state feature $\phi(s_t)$ satisfies $\|\phi(s_t) - \phi(s_v)\|_2 > \epsilon_d$ for all existing nodes, add it as a new node and set edge weight $\boldsymbol{A}_{\phi(s_t), \phi(s_{t-1})} = 1$. If similar to existing nodes, relabel the closest node and increment the edge weight. When the graph is full, replace the oldest or least-connected node.
2. **Encoder-Decoder Training:** Train a feed-forward encoder $\mathbf{E}$ to map $\phi(s) \mapsto g(s)$ and a decoder $\mathbf{D}$ to compute dot-product similarity $g(s_u)^T g(s_v)$. Optimize via:
   $$\mathcal{L} = \sum_{\phi(s_u), \phi(s_v) \in \mathcal{V}} \left[ \mathbf{D}(g(s_u), g(s_v)) - \frac{\boldsymbol{A}_{\phi(s_u), \phi(s_v)}}{\max_{\phi(s_i), \phi(s_j)} \boldsymbol{A}_{\phi(s_i), \phi(s_j)}} \right]^2.$$
3. **Adaptive Schedule:** Track data changes with variable $c$ (increment by $N-1$ for node replacement, $1$ for edge updates). Trigger encoder-decoder gradient updates when $c > \beta(N^2 - N)$, then reset $c$.
4. **Hierarchical Reward Integration:** Replace Euclidean rewards with graph-based intrinsic signals:
   $$r_h = r_{\text{ext}} + \alpha_h \mathbf{D}(\mathbf{E}(\phi(s_t)), \mathbf{E}(g_t)), \quad r_l = -\|\phi(s_{t+1}) - g_t\|^2 + \alpha_l \mathbf{D}(\mathbf{E}(\phi(s_{t+1})), \mathbf{E}(g_t)).$$

**Shapley Credit Assignment Rewards (SCAR) Recipe**
1. **Cooperative Game Framing:** Segment generated text $y$ into $N$ contiguous units (tokens, spans, or sentences) acting as players $\mathcal{P}$.
2. **Characteristic Function:** Define coalition value using the reward model on partial sequences: $v(S) = r_\phi(x, y_S)$, where missing units are filled with empty spaces.
3. **Shapley Value Computation:** Calculate fair credit allocation:
   $$\mathrm{SV}_{u_i}(v) = \sum_{S \subseteq \mathcal{P} \setminus \{u_i\}} \frac{|S|!(N - |S| - 1)!}{N!} [v(S \cup \{u_i\}) - v(S)].$$
4. **Efficient Approximation:** Apply adaptive segmentation and Owen values with a predefined coalition structure $\mathcal{B}$ to reduce complexity from $O(2^N)$ to $O(N^2)$.
5. **Dense Reward Formulation:** Interpolate Shapley credit with terminal feedback:
   $$R_t(\alpha) = R_t^{\mathrm{KL}} + \alpha R_t^{\mathrm{shap}} + (1 - \alpha) \mathbb{I}(t = T) r_\phi(x, y).$$

**Quantitative Results**
G4RL-augmented baselines (HIRO, HRAC, HESS, HLPS) achieve significantly higher success rates and faster convergence across AntMaze, AntMaze-Sparse, AntGather, AntPush, and AntFall environments. High-level intrinsic rewards drive exploration, while low-level rewards refine execution. Computational overhead increases training time by approximately two-fold, but increasing the node-sampling interval to $t_c = 5$ or $10$ steps recovers efficiency with minimal performance degradation. SCAR consistently outperforms sparse RLHF, uniform distribution, and attention-based credit (ABC) baselines. On held-out test sets, SCAR achieves average rewards of 9.27 (IMDB sentiment), 4.35 (TL;DR summarization), and 7.31 (Anthropic HH instruction tuning), surpassing ABC scores of 8.48, 2.85, and 6.59, respectively. LLM-as-judge evaluations report win rates of 61.2% vs. sparse RLHF and 60.3% vs. ABC for summarization, and 56.3% vs. sparse RLHF and 54.9% vs. ABC for instruction tuning.

**Stated Limitations**
G4RL requires careful manual tuning of $\epsilon_d$, $\alpha_l$, and $\alpha_h$, and its theoretical foundation assumes symmetric, reversible dynamics, though empirical results demonstrate robustness to partial asymmetry. Computational bottlenecks stem from graph construction and node comparisons. SCAR's approximation overhead remains non-trivial, and its validity depends on reward models capable of meaningfully scoring partial sequences, limiting applicability to rule-based or final-answer-only evaluators. An ongoing continual RL proposal leverages graph Laplacian spectral matching to transfer state-value knowledge across structurally similar environments, though rigorous baselines, similarity metrics, and appropriate test environments require further development.
