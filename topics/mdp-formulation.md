---
title: MDP formulation of LLM generation
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2505.13697
- ojs:beyond-prompt-engineering-a-reinforced-t
- linkedin:rl-formulations-for-llms-mdp-vs-bandit-l
- x:there-are-two-common-rl-training-formula
- emergentmind:token-level-markov-decision-process-emer
- ojs:pdf-beyond-prompt-engineering-a-reinforc
open_questions:
- Can learned token-level state representations (MDL/DBN) improve credit assignment
  in process-supervised generation, or is the fixed history-as-state convention sufficient
  when per-step rewards exist?
- Does the bandit formulation's equivalence to F-ISFT hold for *non-binary*, *dense*
  outcome rewards (e.g., continuous scores from a reward model), or does the uniform
  advantage assumption break?
- What is the minimal per-step reward density (e.g., PRM every $k$ tokens) needed
  to prevent MDP degeneration and recover true sequential credit assignment?
- How do token-level MDPs for *agentic* tasks (tool use, multi-turn) differ structurally
  from generation MDPs, and do they avoid the bandit collapse?
---

Large Language Model (LLM) generation is naturally framed as sequential decision-making, but the choice between modeling each token as an MDP step versus treating the full completion as a single bandit action fundamentally shapes credit assignment, algorithm design, and the inductive biases of post-training. This article dissects the token-level MDP formulation, its degeneration into a contextual bandit under standard structural assumptions, the role of terminal rewards, and alternative token-level RL applications.

## Token-level MDP formulation for generation

In the canonical token-level MDP, the state $s_t$ at step $t$ is the concatenation of the prompt $q$ and all previously generated tokens $o_{<t}$, i.e., $s_t = (q, o_{<t})$ [source:arxiv:2505.13697]. The action $a_t$ is the selection of the next token $o_t$ from the vocabulary, and the policy $\pi_\theta(a_t|s_t)$ is the LLM's conditional distribution $\pi_\theta(o_t|q, o_{<t})$ [source:linkedin:rl-formulations-for-llms-mdp-vs-bandit-l]. Transitions are deterministic: $s_{t+1} = (s_t, a_t)$ [source:x:there-are-two-common-rl-training-formula]. The episode terminates when an end-of-sequence token $\langle \text{eos} \rangle$ is emitted, yielding a trajectory $\tau = (s_1, a_1, \dots, s_T, a_T)$ [source:linkedin:rl-formulations-for-llms-mdp-vs-bandit-l].

This formulation mirrors the autoregressive generation process exactly. However, the reward structure in practice almost always deviates from the MDP ideal of per-step rewards. Under **outcome supervision**, a scalar reward $R(\tau)$ is observed only at the terminal state $s_{T+1}$ (after $\langle \text{eos} \rangle$), and $r_t = 0$ for all $t < T$ [source:arxiv:2505.13697; x:there-are-two-common-rl-training-formula]. This creates a severe **temporal credit assignment problem**: the algorithm must attribute the terminal outcome to individual token decisions across a variable-length horizon.

## Structural assumptions that collapse the MDP to a bandit

[source:arxiv:2505.13697] identifies two structural assumptions prevalent in LLM post-training (e.g., DeepSeek-R1/GRPO) that effectively reduce the token-level MDP to a contextual bandit:

1. **States encode full action history**: Because $s_t = (q, o_{<t})$, the state is a sufficient statistic for the entire history of actions. The process of generating a sequence becomes equivalent to constructing a single "macro-action" (the full response $o_{1:T}$) conditioned on the context $q$ [source:arxiv:2505.13697].
2. **Terminal reward with uniform credit assignment**: The binary/categorical reward $R \in \{0,1\}$ (or a scalar from a verifier) is assigned only at $T$. In algorithms like GRPO, the advantage $\hat{A}_{i,t}$ is computed per response $i$ and then **uniformly distributed across all tokens** in that response: $\hat{A}_{i,t} = \hat{A}_i$ for all $t$ [source:arxiv:2505.13697].

Under these assumptions, the return-to-go from any state $s_t$ is identical for all $t$ within a given trajectory (it equals the terminal reward). The policy gradient therefore reduces to weighting the *entire sequence's* log-probability by the same scalar, which is mathematically equivalent to a contextual bandit where the context is $q$, the action is the full completion $o_{1:T}$, and the reward is $R$ [source:arxiv:2505.13697].

**Disagreement**: [source:linkedin:rl-formulations-for-llms-mdp-vs-bandit-l] and [source:x:there-are-two-common-rl-training-formula] present the MDP and bandit formulations as *distinct, co-existing design choices* — PPO uses MDP, REINFORCE/RLOO use bandit — and note the bandit formulation "has been argued to perform similarly despite this simplicity." [source:arxiv:2505.13697] contradicts this framing: it argues that **the MDP formulation *as commonly implemented* (with terminal rewards and uniform advantages) *is* a bandit in disguise**, so the empirical similarity is not a coincidence but a mathematical necessity. The bandit formulation is not an alternative; it is the *effective* formulation of the token-MDP under standard assumptions.

## GRPO as Filtered Iterative SFT (F-ISFT)

[source:arxiv:2505.13697] proves that under the above assumptions, the GRPO objective simplifies to an on-policy variant of **Filtered Iterative Supervised Fine-Tuning (F-ISFT)**. Starting from the GRPO objective with clipping and KL penalty:

$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \text{ISR}_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(\text{ISR}_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right) - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right]
$$

where $\text{ISR}_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}$ [source:arxiv:2505.13697]. Assuming (a) KL penalty is negligible due to clipping, (b) ISR stays within clip range, and (c) $\hat{A}_{i,t} = \hat{A}_i$ (uniform advantage), the objective becomes:

$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{\hat{A}_i}{|o_i|} \sum_{t=1}^{|o_i|} \text{ISR}_{i,t}(\theta) \right]
$$

Splitting responses into positive ($\mathcal{G}^+$) and negative ($\mathcal{G}^-$) sets with advantages $\hat{A}_q^+$ and $\hat{A}_q^-$, the gradient is:

$$
\nabla_\theta \mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \left( A^+ \sum_{i \in \mathcal{G}^+} \sum_{t} \text{ISR}_{i,t} \nabla_\theta \log \pi_\theta(o_{i,t}|q,o_{i,<t}) + A^- \sum_{i \in \mathcal{G}^-} \sum_{t} \text{ISR}_{i,t} \nabla_\theta \log \pi_\theta(o_{i,t}|q,o_{i,<t}) \right) \right]
$$

where $A^+ = \frac{\hat{A}_q^+}{|o_i|}$, $A^- = \frac{\hat{A}_q^-}{|o_i|}$ [source:arxiv:2505.13697]. This is a **weighted maximum likelihood update**: increase log-prob of positive completions, decrease log-prob of negative completions, with weights proportional to advantage and inversely proportional to length. Empirically, F-ISFT+ (positive only) and F-ISFT+- (both) match GRPO across GSM8K and Countdown on Qwen, Llama, DeepSeek-Math, and Qwen3 models (e.g., GSM8K Qwen-2.5-1.5B: GRPO 78.24%, F-ISFT+- 76.37%; DeepSeek-Math-7B: GRPO 82.4%, F-ISFT+- 83.7%) [source:arxiv:2505.13697].

**Implication**: The "RL" in GRPO is largely nominal; the algorithm performs iterative supervised learning on filtered model generations. The observed length increase in RL-trained models (correct responses slightly longer, incorrect responses *much* longer) is a side effect of the $1/|o_i|$ scaling and training on negative samples, not improved reasoning [source:arxiv:2505.13697].

## Sequence bandit formulation

The bandit formulation explicitly treats the full completion as a single action [source:linkedin:rl-formulations-for-llms-mdp-vs-bandit-l; x:there-are-two-common-rl-training-formula]:
- **Context**: prompt $q$
- **Action**: full completion $o_{1:T} \sim \pi_\theta(\cdot|q)$
- **Reward**: $R(q, o_{1:T})$ observed after generation
- **Episode**: one-step (context $\to$ action $\to$ reward $\to$ done)

Algorithms: **REINFORCE** (policy gradient with baseline), **RLOO** (Reinforcement Learning with Outcome Optimization) [source:x:there-are-two-common-rl-training-formula]. This formulation *avoids* the credit assignment problem by construction — there is no per-token credit to assign. The gradient estimator is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} \left[ (R(q,o) - b(q)) \nabla_\theta \log \pi_\theta(o|q) \right]
$$

where $b(q)$ is a baseline (e.g., average reward over a group). This is mathematically identical to the F-ISFT gradient derived above when the advantage is uniform and the baseline is the group mean [source:arxiv:2505.13697]. The bandit formulation is therefore not an approximation — it is the *correct* abstraction for outcome-supervised LLM post-training.

## Terminal rewards and credit assignment mechanisms

| Mechanism | Formulation | Credit Assignment | Algorithms |
|-----------|-------------|-------------------|------------|
| **Outcome supervision** (terminal only) | Token-MDP (degenerate) / Bandit (native) | Uniform across tokens (MDP) / None needed (Bandit) | GRPO, PPO (with terminal reward), REINFORCE, RLOO |
| **Process supervision** (per-step) | Token-MDP (non-degenerate) | Learned value function / PRM | PPO with PRM, Process-RL |
| **Token-level RL for input** | Token-MDP (non-generative) | Q-learning with immediate + terminal rewards | RTLIR [source:ojs:beyond-prompt-engineering-a-reinforced-t; ojs:pdf-beyond-prompt-engineering-a-reinforc] |

**RTLIR** [source:ojs:beyond-prompt-engineering-a-reinforced-t; ojs:pdf-beyond-prompt-engineering-a-reinforc] demonstrates a *different* token-level MDP: **input refinement**, not generation. The state $s_i = \text{CONCAT}[v_1, a_1, \dots, v_i, a_i]$ includes embeddings $v_j$ and binary keep/delete actions $a_j \in \{0,1\}$ for each input token. The agent traverses the input sequence once, deciding per token. Rewards combine:
- **Immediate**: cosine similarity of kept token to target embedding
- **Terminal**: log-prob ratio of refined vs original vs optimal input under a downstream LLM

Policy learning uses Q-learning with value decomposition ($Q = V + \mathcal{A} - \text{mean}(\mathcal{A})$) and prioritized experience replay [source:ojs:pdf-beyond-prompt-engineering-a-reinforc]. This is a *bona fide* token-level MDP with per-step decisions and non-uniform credits — but it operates on *input* tokens, not generated output tokens.

## MDL-based state representation for token-level MDPs

[source:emergentmind:token-level-markov-decision-process-emer] addresses a foundational question: **what constitutes a state** when observations are discrete tokens? The Token-Level MDP framework uses **Minimum Description Length (MDL)** to learn a mapping $\phi: h \to s$ from history $h$ (token sequence) to state $s$ that balances:
- Succinctness of state sequence given actions: $CL(S_{1:n} | a_{1:n})$
- Predictiveness of rewards given states/actions: $CL(r_{1:n} | S_{1:n}, a_{1:n})$

Cost function:

$$
\text{Cost}(\phi | h) = CL(S_{1:n} | a_{1:n}) + CL(r_{1:n} | S_{1:n}, a_{1:n})
$$

Optimal mapping $\phi^* = \arg\min_\phi \text{Cost}(\phi | h)$ [source:emergentmind:token-level-markov-decision-process-emer]. For i.i.d. sequences, $CL(x_{1:n}) = n H(\hat{\theta}) + \sum_{i=1}^{m'} \frac{1}{2} \log n$. The framework extends to Dynamic Bayesian Networks for inter-token dependencies. This is a *representation learning* approach to token-level MDPs, distinct from the fixed "history-as-state" convention in LLM generation.

## Current status and trajectory

The **token-level MDP for generation is fading as a *distinct* formulation** for outcome-supervised post-training. [source:arxiv:2505.13697] demonstrates it structurally degenerates to a bandit; the community is converging on the bandit view (RLOO, REINFORCE, DPO-as-bandit) for outcome rewards. However, **process supervision** (per-step rewards from PRMs) and **token-level RL for non-generation tasks** (RTLIR, tool use, reasoning steps) keep token-level MDPs alive in niche but growing directions. The MDL-based state representation [source:emergentmind:token-level-markov-decision-process-emer] remains a research curiosity, not widely adopted in LLM post-training. **Not widely reported**: any large-scale adoption of learned token-level state representations in place of the fixed history-as-state convention for generation.

## Key takeaways

- The standard token-level MDP for LLM generation (state = prompt + prior tokens, action = next token, reward = terminal only) **mathematically collapses to a contextual bandit** because the state encodes full history and advantages are uniformly distributed [source:arxiv:2505.13697].
- **GRPO reduces to Filtered Iterative SFT (F-ISFT)** under these assumptions; empirical parity across models/datasets confirms the degeneration [source:arxiv:2505.13697].
- The **bandit formulation (full completion as action) is the correct abstraction** for outcome-supervised post-training; algorithms like REINFORCE and RLOO implement it natively [source:x:there-are-two-common-rl-training-formula; arxiv:2505.13697].
- **Process supervision** (per-step rewards) is required to make the token-level MDP non-degenerate for generation [source:linkedin:rl-formulations-for-llms-mdp-vs-bandit-l; arxiv:2505.13697].
- **Token-level MDPs thrive outside generation**: RTLIR uses a genuine token-MDP with per-step Q-learning for input refinement [source:ojs:beyond-prompt-engineering-a-reinforced-t].
- **State representation learning** (MDL/DBN) for token sequences exists but is not integrated into mainstream LLM RL pipelines [source:emergentmind:token-level-markov-decision-process-emer].

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [KL regularization in RLHF](kl-regularization.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Length and format bias](length-and-format-bias.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)

## References
- [source:arxiv:2505.13697] [RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs](https://arxiv.org/html/2505.13697v4)
- [source:ojs:beyond-prompt-engineering-a-reinforced-t] [Beyond Prompt Engineering: A Reinforced Token-Level Input Refinement for Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/34586/36741)
- [source:linkedin:rl-formulations-for-llms-mdp-vs-bandit-l] [RL formulations for LLMs: MDP vs Bandit - LinkedIn](https://www.linkedin.com/posts/cwolferesearch_there-are-two-common-rl-training-formulations-activity-7380972876264497153-J2gt)
- [source:x:there-are-two-common-rl-training-formula] [There are two common RL training formulations for LLMs: Markov ...](https://x.com/cwolferesearch/status/1975209587453440344)
- [source:emergentmind:token-level-markov-decision-process-emer] [Token-Level Markov Decision Process - Emergent Mind](https://www.emergentmind.com/topics/token-level-markov-decision-process-mdp)
- [source:ojs:pdf-beyond-prompt-engineering-a-reinforc] [[PDF] Beyond Prompt Engineering: A Reinforced Token-Level Input ...](https://ojs.aaai.org/index.php/AAAI/article/view/34586/36741)
