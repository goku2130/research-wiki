Reinforcement Learning with Verifiable Rewards (RLVR) is a paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs) by utilizing objective, binary feedback—typically based on final answer correctness in mathematics or code—to guide policy optimization. Unlike traditional RLHF, which relies on subjective human preferences or proxy reward models, RLVR leverages the ground-truth verifiability of the output to incentivize the discovery of correct reasoning trajectories [source:arxiv:2506.14245][source:arxiv:2509.24981].

## Theoretical Foundation and MDP Formulation

RLVR treats mathematical and coding reasoning as a specialized finite-horizon Markov Decision Process (MDP) [source:arxiv:2509.24981]. In this formulation, the state space consists of the prompt and the sequence of generated tokens, and the transitions are deterministic and tree-structured [source:arxiv:2509.24981]. The reward is sparse and binary, provided only at the terminal state based on the correctness of the final answer $\mathcal{I}_{\mathrm{Ans}}(a_i)$ [source:arxiv:2506.14245].

A central theoretical pillar in analyzing RLVR is the **Logic Prior assumption**, which posits that a correct Chain-of-Thought (CoT) is significantly more likely to lead to a correct answer than an incorrect CoT:
\[
P(\mathcal{I}_{\mathrm{Ans}}=1 \mid \mathcal{I}_{\mathrm{CoT}}=1) = \alpha > P(\mathcal{I}_{\mathrm{Ans}}=1 \mid \mathcal{I}_{\mathrm{CoT}}=0) = \beta
\]
Under this assumption, it is proven that the expected advantage for correct CoTs is positive ($\mathbb{E}[\hat{A}(y_i) \mid \mathcal{I}_{\mathrm{CoT}}=1] > 0$), while it is negative for incorrect ones, theoretically guaranteeing that the probability of generating correct reasoning paths increases monotonically during training [source:arxiv:2506.14245].

## Optimization Frameworks

### Group Relative Policy Optimization (GRPO) and PPO
GRPO is a prevalent framework in RLVR that eliminates the need for a separate value network by using group-relative advantages. For a prompt $q$, $G$ responses are sampled, and the advantage $\hat{A}(y_i)$ for response $y_i$ is computed as:
\[
\hat{A}(y_i) = \frac{R(y_i) - \mu_{\mathbf{Y}}}{\sigma_{\mathbf{Y}}}, \quad \mu_{\mathbf{Y}} = \frac{1}{G} \sum_{j=1}^G R(y_j), \quad \sigma_{\mathbf{Y}} = \sqrt{\frac{1}{G} \sum_{j=1}^G (R(y_j) - \mu_{\mathbf{Y}})^2}
\]
To prevent overly large updates and ensure training stability, GRPO utilizes a clipped surrogate objective [source:arxiv:2512.16912]:
\[
J(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{|\mathbf{y}^{(i)}|} \min \{ r_t^{(i)}(\theta) A_i, \text{clip}(r_t^{(i)}(\theta), 1-\varepsilon, 1+\varepsilon) A_i \} \right]
\]
where $r_t^{(i)}(\theta)$ is the probability ratio $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ and $\varepsilon$ is the clipping ratio [source:arxiv:2512.16912].

### Random Policy Valuation (ROVER)
ROVER challenges the necessity of Generalized Policy Iteration (GPI) loops used in PPO and GRPO, arguing they are unnecessarily complex for the deterministic, tree-structured MDP of reasoning [source:arxiv:2509.24981]. Instead, ROVER recovers optimal actions directly from the Q-function of a fixed uniform policy $\pi_u$:
\[
\hat{Q}^{\pi_u}(s, a) \leftarrow r(s, a) + \frac{1}{|A|} \sum_{a' \in A} \hat{Q}^{\pi_u}(s', a')
\]
To stabilize training, ROVER uses an intrinsic relative Q-parameterization: $Q(s_t, a_t) = \rho \big( \log \pi_\theta(a_t | s_t) - \log \pi_{\theta_{\text{old}}}(a_t | s_t) \big)$ [source:arxiv:2509.24981].

### Rewards-Aware Policy Optimization (RAPO)
RAPO addresses the "mode-seeking" limitation of reverse KL divergence used in GRPO [source:arxiv:2510.03865]. It replaces reverse KL with a forward KL objective and entropy maximization:
\[
\mathcal{J}_{\mathrm{FKL}}(\theta) = \mathbb{E}_{x \sim P(x), y \sim \pi_{\theta}(y|x)}[r(x, y)] - \alpha \mathbb{D}_{\mathrm{KL}}(\pi_{\mathrm{ref}}||\pi_{\theta}) + \beta H(\pi_{\theta})
\]
RAPO further employs a reweighted reference policy $\hat{\pi}_{\text{ref}}(y|x) = \pi_{\text{ref}}^{\phi(r(x,y))}(y|x)/Z$, where $\phi(r)$ is a monotonically increasing function that pushes the policy toward uniformity when rewards are low to encourage exploration [source:arxiv:2510.03865].

## The Exploration-Exploitation Dilemma

A critical debate in RLVR is whether the process merely reweights existing trajectories or extends the model's reasoning boundary. While some hypothesized that all correct paths exist in the base model, empirical evidence using the **CoT-Pass@K** metric (requiring both CoT and answer to be correct) demonstrates that RLVR genuinely extends the capability boundary on benchmarks like AIME [source:arxiv:2506.14245].

### Mechanisms to Overcome the Exploration Ceiling
Several methods have been proposed to prevent the model from collapsing into narrow, over-optimized behaviors:

| Method | Mechanism | Target Problem | Result |
| :--- | :--- | :--- | :--- |
| **PSN-RLVR** | Additive Gaussian noise in MLP/FFN layers [source:arxiv:2602.02555] | Token-level jitter and lack of global coherence [source:arxiv:2602.02555] | AIME 2024 Pass@256 increased from 62.8% $\to$ 65.5% (Qwen3-4B) [source:arxiv:2602.02555] |
| **Entropy Shaping** | Augmented advantage $\psi(\mathcal{H}_t) = \min(\alpha \mathcal{H}_t^{\text{detach}}, \frac{|A_t|}{\kappa})$ [source:arxiv:2506.14758] | Exploitation bias and entropy collapse [source:arxiv:2506.14758] | AIME24 Pass@256 rose from 46.7 $\to$ 56.7 (Qwen2.5-Base + GRPO) [source:arxiv:2506.14758] |
| **Forward KL (RAPO)** | $\mathbb{D}_{\mathrm{KL}}(\pi_{\mathrm{ref}}||\pi_{\theta})$ instead of $\mathbb{D}_{\mathrm{KL}}(\pi_{\theta}||\pi_{\mathrm{ref}})$ [source:arxiv:2510.03865] | Mode-seeking behavior trapping policy in base model support [source:arxiv:2510.03865] | AIME2024 Full accuracy of 0.809 (Qwen2.5-7B) [source:arxiv:2510.03865] |

## Disagreements and Paradoxes in RLVR

### The Entropy Paradox
There is a fundamental disagreement regarding the role of policy entropy in reasoning performance:
*   **Entropy as a Catalyst:** [source:arxiv:2506.14758] argues that high token entropy correlates with pivotal logical tokens and reflective actions, and thus should be explicitly incentivized via advantage shaping to prevent performance plateaus.
*   **Entropy Minimization as a Benefit:** [source:arxiv:2512.16912] reports a paradox where entropy minimization (induced by clipping in GRPO) can actually improve mathematical reasoning. They find that clipping acts as an implicit entropy regularizer that forces monotonic entropy decay, which can be beneficial in certain regimes.

### The Impact of Spurious Rewards
Classical RL suggests that noise in the reward signal hinders learning. However, [source:arxiv:2512.16912] finds that "spurious rewards" (random Bernoulli(1/2) feedback) can actually lead to performance gains in stronger models (e.g., R1-Distill-Llama-8B), though they cause degradation in weaker models (e.g., Qwen2.5-Math-1.5B).

## Current Status and Trajectory

RLVR appears to be a rising and dominant technique for developing reasoning-heavy LLMs, as evidenced by a shift toward complex RL loops for "O1-style" reasoning [source:arxiv:2506.14245][source:arxiv:2509.24981]. 

The trajectory suggests a movement away from general-purpose RL algorithms (PPO) toward **domain-specific optimizations** (ROVER) and **advanced exploration strategies** (PSN, RAPO). However, the field remains fragmented regarding the optimal approach to exploration: some favor parameter-space noise [source:arxiv:2602.02555], others favor divergence-based objectives [source:arxiv:2510.03865], and others favor entropy-based advantage shaping [source:arxiv:2506.14758]. It is not yet widely established whether these methods are orthogonal or mutually exclusive, though PSN-RLVR is claimed to be orthogonal to $\text{Pass@K}$ training [source:arxiv:2602.02555].

## Key Takeaways
*   **Capability Extension:** RLVR does more than reweight; it expands the model's reasoning boundary, as verified by CoT-Pass@K metrics [source:arxiv:2506.14245].
*   **Exploration Ceiling:** Standard RLVR (Reverse KL) can trap models in the base model's support region [source:arxiv:2510.03865]. Parameter-space noise (PSN) and Forward KL (RAPO) are primary methods to break this ceiling [source:arxiv:2602.02555][source:arxiv:2510.03865].
*   **Algorithmic Simplification:** ROVER demonstrates that for the specific MDP of math reasoning, complex GPI loops can be replaced by random policy valuation [source:arxiv:2509.24981].
*   **Entropy Ambivalence:** While entropy is generally seen as a tool for exploration [source:arxiv:2506.14758], implicit entropy reduction via clipping can also yield gains in specific model regimes [source:arxiv:2512.16912].

## Related Topics
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)