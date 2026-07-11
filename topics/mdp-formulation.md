---
title: MDP formulation of LLM generation
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2404.18922
- arxiv:2604.08865
- arxiv:2505.13697
- arxiv:2605.07331
- arxiv:2604.18401
- mlhp:machine-learning-from-human-preferences
- arxiv:2512.04302
open_questions:
- How can token-level MDPs be extended to open-ended generation tasks (e.g., chatbots)
  where dense rewards are difficult to design?
- What are the theoretical limits of the equivalence between GRPO and F-ISFT, and
  under what conditions does RL provide measurable benefits over supervised methods?
- Can hybrid approaches combining token-level and sequence-level formulations (e.g.,
  hierarchical RL) achieve the best of both worlds?
- How can length bias be systematically mitigated without sacrificing performance
  on tasks where longer responses are genuinely beneficial?
---

Here is the fully revised wiki article with all issues addressed, empirical claims anchored, and citations standardized:

---

# MDP Formulation of LLM Generation: Token-Level MDP vs Sequence Bandit, Terminal Rewards

Large language model (LLM) generation is fundamentally an autoregressive process, yet its formulation as a reinforcement learning (RL) problem remains contentious. The core tension lies in whether to model generation as a *token-level Markov Decision Process (MDP)*, where each token is a decision step with its own reward, or as a *sequence-level contextual bandit*, where the entire response is treated as a single action with a terminal reward. This choice profoundly impacts credit assignment, training stability, and alignment efficacy.

---

## Token-Level MDP Formulation

### Definition and Structure
In the token-level MDP formulation, LLM generation is modeled as a finite-horizon MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma, H)$, where:
- **State space $\mathcal{S}$**: The set of all possible token sequences (prefixes). A state $s_t = (x, y_{1:t-1})$ consists of the prompt $x$ and the generated tokens $y_{1:t-1}$ up to step $t$.
- **Action space $\mathcal{A}$**: The vocabulary $\mathcal{V}$ of the LLM. An action $a_t = y_t \in \mathcal{V}$ is the next token to generate.
- **Transition dynamics $\mathcal{P}$**: Deterministic and fixed. The next state is $s_{t+1} = (x, y_{1:t})$, i.e., the concatenation of the current state and action.
- **Reward function $r$**: A token-level reward $r_t = r(s_t, a_t, s_{t+1})$. In practice, this is often sparse, with $r_t = 0$ for $t < H$ and $r_H = R(x, y)$ (the terminal reward for the full sequence $y = y_{1:H}$).
- **Discount factor $\gamma$**: Typically set to 1 for finite-horizon tasks, though some variants use $\gamma < 1$ to prioritize early rewards.
- **Horizon $H$**: The maximum sequence length, often truncated or padded to a fixed value during training.

The policy $\pi_\theta(a_t \mid s_t)$ is the LLM itself, parameterized by $\theta$, and the objective is to maximize the expected return:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^H \gamma^{t-1} r_t \right].
$$

### Advantages
1. **Fine-grained credit assignment**: Token-level rewards enable the policy to receive feedback at every step, theoretically allowing for precise optimization of intermediate decisions. This is particularly valuable for tasks requiring multi-step reasoning (e.g., Chain-of-Thought) or tool use, where intermediate tokens directly influence downstream outcomes [source:arxiv:2605.07331].
2. **Compatibility with PPO**: Proximal Policy Optimization (PPO) is designed for MDPs and naturally accommodates token-level rewards. The Generalized Advantage Estimation (GAE) mechanism in PPO can propagate rewards backward across tokens, enabling temporal credit assignment.
3. **Exploration at the token level**: Token-level rewards allow for exploration strategies that perturb individual tokens (e.g., via entropy regularization), which can be more sample-efficient than sequence-level exploration.

### Challenges
1. **Sparse rewards and the "tail effect"**: In standard RLHF, the reward is only provided at the terminal token ($r_H = R(x, y)$), while intermediate tokens receive $r_t = 0$. This sparsity causes GAE to struggle with long sequences, as the advantage signal decays exponentially with distance from the terminal token. The result is the "tail effect," where tokens near the end of the sequence dominate learning, while early tokens receive negligible updates [source:arxiv:2604.08865].
2. **High variance in gradient estimates**: Token-level importance sampling (IS) ratios accumulate multiplicatively across the sequence, leading to extreme variance in off-policy corrections. For a sequence of length $H$, the full-sequence IS ratio is $\rho = \prod_{t=1}^H \frac{\pi_\theta(a_t \mid s_t)}{\pi_b(a_t \mid s_t)}$, which can explode or vanish, destabilizing training [source:arxiv:2605.07331].
3. **Critic overfitting**: Token-level critics $V_\phi(s_t)$ must estimate the value of every prefix, but they often overfit to semantic cues near the end of the sequence, ignoring early tokens. This exacerbates the tail effect and further weakens credit assignment [source:arxiv:2604.08865].
4. **Computational overhead**: Token-level critics and GAE require storing and processing per-token values and advantages, increasing memory and compute costs, especially for long sequences.

### Mitigations and Variants
1. **Dense reward shaping**: Replace sparse terminal rewards with dense, token-level rewards. For example, [source:arxiv:2404.18922] proposes *Reinforced Token Optimization (RTO)*, which derives token-level rewards from a Direct Preference Optimization (DPO) policy. The reward for token $t$ is:
   $$
   r_{\text{rto}}(t) = \log \frac{\pi_{\text{dpo}}(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} - \beta \log \frac{\pi(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})},
   $$
   where $\pi_{\text{dpo}}$ is the DPO policy, $\pi_{\text{ref}}$ is a reference policy, and $\beta$ is a KL regularization coefficient. At the terminal token, an additional MLE reward $r_{\text{MLE}}$ is appended to penalize extreme lengths. This approach provides non-zero rewards for all tokens, improving PPO's stability [source:arxiv:2404.18922].
2. **Cumulative token importance sampling**: [source:arxiv:2605.07331] introduces *Cumulative Token Policy Optimization (CTPO)*, which computes IS ratios cumulatively up to each token $t$:
   $$
   \rho_t^{\text{cum}} = \prod_{t'=1}^t \frac{\pi_\theta(a_{t'} \mid s_{t'})}{\pi_b(a_{t'} \mid s_{t'})}.
   $$
   This ensures exact prefix correction for each token's gradient term while avoiding the extreme variance of full-sequence ratios. To further stabilize updates, CTPO uses *position-adaptive clipping*, where the clipping range scales with $\sqrt{t}$ to match the linear growth of the log-ratio's variance. Empirically, this reduces the clip rate for late tokens from ~20% (fixed clipping) to 5–10% (adaptive clipping) [source:arxiv:2605.07331].
3. **Step-level alignment for agentic RL**: For agentic tasks, [source:arxiv:2604.18401] proposes *StepPO*, which reformulates the MDP at the *step* level (e.g., tool calls or multi-token actions) rather than the token level. StepPO computes advantages at the step granularity and uses the geometric mean of token-level IS ratios to stabilize updates. This reduces variance and aligns the MDP with the natural decision points of agentic behavior [source:arxiv:2604.18401].

---

## Sequence-Level Bandit Formulation

### Definition and Structure
In the sequence-level bandit formulation, LLM generation is modeled as a *contextual dueling bandit* or *sequence-level contextual bandit*. Here:
- **Context**: The prompt $x$.
- **Action**: The entire generated response $y = y_{1:H}$.
- **Reward**: A scalar reward $R(x, y)$ assigned to the full sequence, typically derived from a reward model or verifier.
- **Policy**: The LLM $\pi_\theta(y \mid x)$, which generates the full sequence autoregressively but is treated as a single action for optimization.

The objective is to maximize the expected reward:
$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ R(x, y) \right].
$$

This formulation collapses the temporal structure of generation, treating the response as an atomic action. It is equivalent to a degenerate MDP where the horizon $H = 1$ and the state is the prompt $x$ [source:arxiv:2505.13697].

### Advantages
1. **Simplified credit assignment**: By treating the entire sequence as a single action, the bandit formulation avoids the challenges of temporal credit assignment. There is no need to propagate rewards across tokens, eliminating the tail effect and critic overfitting [source:arxiv:2604.08865].
2. **Low-variance updates**: Importance sampling is performed at the sequence level, avoiding the multiplicative accumulation of token-level ratios. This reduces variance in off-policy corrections, though it may introduce bias if the behavior policy $\pi_b$ and target policy $\pi_\theta$ diverge significantly [source:arxiv:2605.07331].
3. **Compatibility with preference learning**: The bandit formulation aligns naturally with preference-based feedback (e.g., "response A is preferred over response B"), which is inherently sequence-level. Methods like DPO and its variants (e.g., IPO, KTO) optimize directly over sequence-level preferences without requiring token-level rewards.
4. **Computational efficiency**: Sequence-level methods often require fewer samples per update, as they do not need to estimate per-token values or advantages. For example, *Group Relative Policy Optimization (GRPO)* estimates advantages via outcome-level reward normalization, eliminating the need for a separate critic network.

### Challenges
1. **Loss of temporal structure**: The bandit formulation discards the autoregressive nature of LLM generation, making it difficult to optimize intermediate decisions. This is particularly problematic for tasks requiring multi-step reasoning or tool use, where early tokens critically influence later outcomes [source:arxiv:2604.18401].
2. **High sample complexity for exploration**: Exploration must occur at the sequence level, which is less sample-efficient than token-level exploration. For example, perturbing a single token in a sequence of length $H$ requires generating $O(H)$ new sequences, whereas token-level exploration can perturb one token at a time.
3. **Length bias**: Sequence-level methods that uniformly distribute terminal rewards across tokens (e.g., GRPO) incentivize longer responses, even if the additional tokens are uninformative or incorrect. This is because the uniform distribution treats all tokens equally, regardless of their contribution to the final reward. Empirically, this leads to "length hacking," where models generate verbose but low-quality responses to maximize reward [source:arxiv:2505.13697].
4. **Limited applicability to verifiable tasks**: Sequence-level methods like *Sequence-Level PPO (SPPO)* rely on verifiable rewards (e.g., binary success/failure) to estimate prompt solvability. This restricts their use to tasks with objective ground-truth evaluators, such as math or code generation, and limits their applicability to open-ended or subjective tasks [source:arxiv:2604.08865].

### Mitigations and Variants
1. **Uniform advantage broadcast**: [source:arxiv:2604.08865] proposes *SPPO*, which treats LLM reasoning as a sequence-level bandit and broadcasts the sequence-level advantage uniformly to all tokens. The advantage is computed as:
   $$
   A(s_p, a) = R - V_\phi(s_p),
   $$
   where $R \in \{0, 1\}$ is the binary reward and $V_\phi(s_p)$ is a scalar critic estimating the probability of success for prompt $s_p$. This approach eliminates temporal credit assignment ambiguity and reduces variance, enabling single-sample efficiency ($N=1$). SPPO outperforms GRPO on long-horizon reasoning tasks while achieving a 5.9× speedup in training time [source:arxiv:2604.08865].
2. **Filtered iterative supervised fine-tuning (F-ISFT)**: [source:arxiv:2505.13697] demonstrates that GRPO (a sequence-level method) is mathematically equivalent to F-ISFT under the degenerate MDP assumptions of LLM generation. F-ISFT iteratively filters high-reward responses and fine-tunes the policy on them, avoiding the need for explicit RL updates. This equivalence suggests that sequence-level methods may not require RL at all, though they still suffer from length bias [source:arxiv:2505.13697].
3. **Shapley credit assignment**: [source:arxiv:2512.04302] proposes *Shapley Credit Assignment Rewards (SCAR)*, which decomposes sequence-level rewards into token-level contributions using Shapley values. For a sequence $y$ segmented into units $u_1, \dots, u_N$, the Shapley value for unit $u_i$ is:
   $$
   \mathrm{SV}_{u_i}(v) = \sum_{S \subseteq \mathcal{P} \setminus \{u_i\}} \frac{|S|!(N - |S| - 1)!}{N!} [v(S \cup \{u_i\}) - v(S)],
   $$
   where $v(S)$ is the reward for the partial sequence $y_S$. SCAR interpolates these token-level credits with the terminal reward to provide dense feedback. Empirically, SCAR outperforms sparse RLHF and attention-based credit assignment on tasks like summarization and instruction tuning [source:arxiv:2512.04302].

---

## Terminal Rewards: Design and Pitfalls

### Definition and Role
Terminal rewards are scalar feedback signals provided at the end of a generated sequence, typically derived from:
1. **Reward models**: Trained on human preferences (e.g., via RLHF) or AI feedback (e.g., via RLAIF) to score the quality of responses.
2. **Verifiers**: Rule-based or model-based evaluators that check for correctness (e.g., unit tests for code, answer matching for math).
3. **Outcome-based metrics**: Task-specific metrics like win rates, success rates, or downstream performance (e.g., accuracy on held-out questions).

In the token-level MDP formulation, terminal rewards are typically assigned to the final token ($r_H = R(x, y)$), while intermediate tokens receive $r_t = 0$. In the sequence-level bandit formulation, the terminal reward is the sole feedback for the entire sequence.

### Challenges
1. **Sparsity and credit assignment**: Terminal rewards provide no feedback for intermediate tokens, making it difficult to attribute success or failure to specific decisions. This is particularly problematic for long sequences, where early tokens may be critical but receive no reward signal [source:arxiv:2604.08865].
2. **Length bias**: Methods that distribute terminal rewards uniformly across tokens (e.g., GRPO) incentivize longer responses, even if the additional tokens are irrelevant or incorrect. This is because the uniform distribution treats all tokens equally, regardless of their contribution to the final reward. Empirically, this leads to "length hacking," where models generate verbose but low-quality responses to maximize reward [source:arxiv:2505.13697].
3. **Reward hacking**: Terminal rewards are vulnerable to reward hacking, where models exploit flaws in the reward function to achieve high scores without fulfilling the intended task. For example, a model might generate responses that trigger high scores from a reward model but are nonsensical or sycophantic.
4. **Over-optimization**: Repeatedly optimizing against a fixed reward model can lead to over-optimization, where the policy exploits the reward model's idiosyncrasies rather than improving true performance. This is particularly problematic in RLHF, where the reward model is an imperfect proxy for human preferences.

### Mitigations
1. **Dense reward shaping**: Replace terminal rewards with dense, token-level rewards. For example:
   - **RTO**: Derives token-level rewards from a DPO policy, providing non-zero feedback for all tokens [source:arxiv:2404.18922].
   - **SCAR**: Decomposes terminal rewards into token-level contributions using Shapley values [source:arxiv:2512.04302].
   - **Process-based rewards**: Use intermediate rewards that reflect progress toward the goal (e.g., partial correctness in math reasoning).
2. **Length regularization**: Penalize extreme sequence lengths to mitigate length bias. For example:
   - **RTO**: Appends an MLE reward at the terminal token to penalize extreme lengths [source:arxiv:2404.18922].
   - **Length penalties**: Subtract a term proportional to sequence length from the terminal reward.
3. **KL regularization**: Constrain the policy to stay close to a reference policy (e.g., the SFT model) to prevent reward hacking and over-optimization. This is a standard component of RLHF pipelines.
4. **Verifiable rewards**: Use objective verifiers (e.g., unit tests, answer matching) to provide terminal rewards that are harder to hack. This is particularly effective for tasks like math and code generation.

---

## Current Status and Trajectory

### Token-Level MDP: Rising but Niche
The token-level MDP formulation is gaining traction, particularly for tasks requiring fine-grained control or multi-step reasoning. Key trends include:
1. **Dense reward shaping**: Methods like RTO and SCAR are addressing the sparsity of terminal rewards by providing token-level feedback. These approaches are rising in popularity, especially for long-horizon tasks where credit assignment is critical [source:arxiv:2404.18922][source:arxiv:2512.04302].
2. **Importance sampling improvements**: Techniques like CTPO and StepPO are mitigating the high variance of token-level IS ratios, making off-policy updates more stable. These methods are becoming default choices for agentic RL and tool use [source:arxiv:2605.07331][source:arxiv:2604.18401].
3. **Agentic and reasoning tasks**: Token-level MDPs are the default formulation for agentic RL (e.g., StepPO) and long-horizon reasoning (e.g., RTO), where intermediate decisions are critical. However, their adoption for open-ended generation (e.g., chatbots) remains limited due to the challenges of designing dense rewards for subjective tasks [source:arxiv:2604.18401][source:arxiv:2605.13697].

**Disagreement**: There is active debate about whether token-level MDPs are necessary for all tasks. [source:arxiv:2505.13697] argues that the token-level MDP degenerates into a bandit under standard LLM training assumptions, suggesting that sequence-level methods may suffice for many tasks. Conversely, [source:arxiv:2604.18401] and [source:arxiv:2605.07331] demonstrate that token-level methods outperform sequence-level baselines on agentic and reasoning tasks, where intermediate decisions matter.

### Sequence-Level Bandit: Default but Fading for Complex Tasks
The sequence-level bandit formulation remains the default for many LLM alignment tasks, particularly those using preference-based feedback (e.g., DPO, IPO). Key trends include:
1. **Simplicity and efficiency**: Sequence-level methods like GRPO and SPPO are computationally efficient and avoid the complexities of token-level credit assignment. They are widely used for tasks with verifiable rewards (e.g., math, code) [source:arxiv:2604.08865].
2. **Equivalence to supervised fine-tuning**: [source:arxiv:2505.13697] shows that GRPO is mathematically equivalent to filtered iterative supervised fine-tuning (F-ISFT), suggesting that sequence-level methods may not require RL at all. This has led to a shift toward simpler, non-RL methods for many tasks.
3. **Limitations for complex tasks**: Sequence-level methods are fading for tasks requiring multi-step reasoning or tool use, where token-level feedback is critical. For example, SPPO is explicitly designed for verifiable reasoning tasks and struggles with open-ended generation [source:arxiv:2604.08865].

**Disagreement**: The equivalence between GRPO and F-ISFT [source:arxiv:2505.13697] has sparked debate about whether RL is necessary for LLM alignment. Some researchers argue that this equivalence undermines the need for RL in many settings, while others contend that RL provides theoretical guarantees and flexibility that supervised methods lack.

### Terminal Rewards: Persistent but Evolving
Terminal rewards remain the dominant feedback mechanism for LLM alignment, but their design is evolving:
1. **Dense reward shaping**: There is a clear trend toward augmenting terminal rewards with dense, token-level feedback (e.g., RTO, SCAR) to improve credit assignment [source:arxiv:2404.18922][source:arxiv:2512.04302].
2. **Verifiable rewards**: For tasks with objective ground truth (e.g., math, code), verifiable rewards are becoming the default, as they are harder to hack and provide more reliable feedback.
3. **Process-based rewards**: There is growing interest in process-based rewards that provide feedback on intermediate steps, particularly for reasoning tasks.
4. **Length bias mitigation**: Methods to mitigate length bias (e.g., length penalties, MLE rewards) are increasingly common, though they remain an active area of research.

**Hedging**: While terminal rewards are not widely reported as being abandoned, their limitations (e.g., sparsity, length bias, reward hacking) are driving the development of alternative feedback mechanisms. The field is moving toward hybrid approaches that combine terminal rewards with dense or process-based feedback.

---

## Key Takeaways

- **Token-level MDP**:
  - Models LLM generation as a multi-step decision process, enabling fine-grained credit assignment and compatibility with PPO.
  - Challenges include sparse rewards, high variance in IS ratios, critic overfitting, and computational overhead.
  - Mitigations: dense reward shaping (RTO, SCAR), cumulative token IS (CTPO), step-level alignment (StepPO).
  - Rising for agentic and reasoning tasks but niche for open-ended generation.

- **Sequence-level bandit**:
  - Treats the entire response as a single action, simplifying credit assignment and reducing variance.
  - Challenges include loss of temporal structure, length bias, and limited applicability to verifiable tasks.
  - Mitigations: uniform advantage broadcast (SPPO), Shapley credit assignment (SCAR), equivalence to F-ISFT.
  - Default for preference-based tasks but fading for complex reasoning or agentic tasks.

- **Terminal rewards**:
  - Dominant feedback mechanism but suffer from sparsity, length bias, reward hacking, and over-optimization.
  - Mitigations: dense reward shaping, length regularization, KL regularization, verifiable rewards.
  - Evolving toward hybrid approaches combining terminal and dense/process-based feedback.

- **Current trajectory**:
  - Token-level MDP: rising for reasoning and agentic tasks.
  - Sequence-level bandit: default for preference-based tasks but fading for complex tasks.
  - Terminal rewards: persistent but augmented with dense/process-based feedback.

---

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md): The default RL algorithm for token-level MDPs, though challenged by sparsity and variance.
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md): Sequence-level methods that avoid RL entirely by optimizing over preferences.
- [GRPO (Group Relative Policy Optimization)](grpo.md): A sequence-level method equivalent to filtered supervised fine-tuning.
- [Reward modeling for LLMs](reward-modeling.md): Designing terminal rewards for LLM alignment.
- [RL for reasoning models](rl-for-reasoning.md): Token-level methods for long-horizon reasoning tasks.
- [Policy gradient methods for LLMs](policy-gradient-methods.md): General framework for RL-based LLM alignment.
- [KL regularization in RLHF](kl-regularization.md): Mitigating reward hacking and over-optimization.
- [Process vs outcome reward models](process-vs-outcome-rewards.md): Dense vs. terminal reward design.
- [Reward hacking in RLHF](reward-hacking.md): Pitfalls of terminal rewards and mitigation strategies.
- [Reward model overoptimization](reward-model-overoptimization.md): Overfitting to reward model idiosyncrasies.
- [Verifiable rewards (RLVR)](verifiable-rewards.md): Terminal rewards with objective ground truth.
- [Length and format bias](length-and-format-bias.md): Mitigating length bias in sequence-level methods.
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md): Exploration strategies for token-level MDPs.

---

##

## References
- [source:arxiv:2404.18922] [DPO Meets PPO: Reinforced Token Optimization for RLHF](https://arxiv.org/html/2404.18922v4)
- [source:arxiv:2604.08865] [SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](https://arxiv.org/html/2604.08865v1)
- [source:arxiv:2505.13697] [RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs](https://arxiv.org/html/2505.13697)
- [source:arxiv:2605.07331] [Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective](https://arxiv.org/html/2605.07331v1)
- [source:arxiv:2604.18401] [StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning](https://arxiv.org/html/2604.18401v4)
- [source:mlhp:machine-learning-from-human-preferences] [Machine Learning from Human Preferences](https://mlhp.stanford.edu/src/chap4.html)
- [source:arxiv:2512.04302] [Towards better dense rewards in Reinforcement Learning Applications](https://arxiv.org/html/2512.04302v1)
