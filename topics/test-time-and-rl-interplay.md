---
title: Test-time compute and RL interplay
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2203.14465
- arxiv:2403.09629
- nature:olympiad-level-formal-mathematical-reaso
- arxiv:2304.01132
- arxiv:2501.02497
- wellecks:l1-controlling-how-long-a-reasoning-mode
- arxiv:2510.09988
- deepmind:alphageometry-2-new-geometry-reasoning-s
open_questions:
- What is the functional form of the test-time scaling law relating model size, search
  budget (tokens/rollouts), verifier quality, and task difficulty—and can it be derived
  from first principles?
- How can prompt-space search (optimizing reasoning structure $p$) be made tractable,
  and does it yield transferable "reasoning algorithms" beyond per-instance answer
  search?
- Does the RL-search flywheel converge to a fixed point, or does it exhibit cyclic
  dynamics / reward hacking as the policy overfits to the search distribution?
- Can intrinsic self-correction be made reliable without extrinsic tools, or is the
  "verifier gap" fundamental for open-ended domains lacking formal verification?
---

Test-time compute scaling has emerged as the primary lever for eliciting System-2 reasoning capabilities from LLMs, shifting the bottleneck from parameter count to inference-time search guided by learned reward signals. The interplay between reinforcement learning (RL) and test-time search creates a dual optimization loop where RL internalizes search strategies into policy parameters while search externalizes computation to solve specific instances beyond the policy's zero-shot capability.

## Test-Time Compute Paradigms: System-1 Adaptation vs System-2 Reasoning

The literature distinguishes two fundamentally different uses of test-time compute [source:arxiv:2501.02497]. **Test-Time Adaptation (TTA)** targets System-1 models—fast, pattern-based predictors—by updating parameters (Test-Time Training, FTTA), optimizing in-context demonstrations, editing representations via steering vectors, or calibrating outputs with $k$-NN retrieval. These methods aim at robustness to distribution shift but do not induce deliberate reasoning. **Test-Time Reasoning** targets System-2 models by structuring generation into explicit, multi-step reasoning chains with feedback and search. This paradigm decomposes into three components: *Feedback Modeling* (ORMs, PRMs, generative critics), *Search Strategies* (repeated sampling, self-correction, tree search), and *Improvement Training* (distilling search traces back into the policy) [source:arxiv:2501.02497]. The boundary is porous: STaR [source:arxiv:2203.14465] and Quiet-STaR [source:arxiv:2403.09629] use self-generated rationales filtered by correctness (a form of outcome verification) to fine-tune the base model, effectively converting test-time search successes into training data.

## Search Strategies at Test Time

### Repeated Sampling: Best-of-N and Self-Consistency

The simplest search generates $N$ independent completions $y^{(1)},\dots,y^{(N)} \sim \pi_\theta(\cdot|x)$ and aggregates. **Best-of-N (BoN)** selects $\arg\max_i V(y^{(i)})$ using a verifier $V$ (ORM or PRM) [source:arxiv:2501.02497; wellecks:l1-controlling-how-long-a-reasoning-mode]. **Self-Consistency (voting)** selects the most frequent answer $\arg\max_a \sum_i \mathbb{I}[a_i = a]$, optionally weighted by $V$ [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. The voting accuracy converges as $N\to\infty$ to a limit determined by the generator $g$ and verifier $v$ [source:wellecks:l1-controlling-how-long-a-reasoning-mode]:

$$
\frac{1}{M}\sum_{i=1}^{M}{\mathbb{I}}\left[a_{i}^{*}=\arg\max_{a}\sum_{z}v(x,z,a)g(z,a|x)\right]
$$

where $z$ is the solution path. Self-consistency CoT improves math reasoning by $\sim 18\%$ over vanilla CoT [source:arxiv:2501.02497]. However, BoN suffers from **over-optimization** when $V$ is imperfect: as $N$ grows, the selected output exploits reward model errors rather than true quality [source:wellecks:l1-controlling-how-long-a-reasoning-mode; arxiv:2510.09988].

### Self-Correction and Refinement

Refinement iterates $y^{(i+1)} = g(x, y^{(i)}, F(y^{(i)}))$ where $F$ provides feedback [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. **Extrinsic feedback** (code execution, theorem provers, retrievers) works reliably because it injects ground-truth signals and localizes errors [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. **Intrinsic feedback** (self-critique) yields mixed results in math: models often fail to locate their own errors even when capable of correcting them [source:arxiv:2501.02497; wellecks:l1-controlling-how-long-a-reasoning-mode]. AlphaProof uses Lean verification as extrinsic feedback at every tactic step, making refinement the core of its search [source:nature:olympiad-level-formal-mathematical-reaso].

### Tree Search: MCTS and Structured Exploration

Tree search explores a reasoning tree where nodes are partial states $s$ (e.g., prefix of CoT), actions $a$ are next-step generations, and transitions $s\to s'$ append tokens. **Monte Carlo Tree Search (MCTS)** operates in four phases [source:arxiv:2510.09988]:
1. **Selection**: UCT policy $a = \arg\max_{a\in A(s)} \left( Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}} \right)$
2. **Expansion**: Sample new child nodes via LLM policy $\pi_\theta$
3. **Simulation**: Rollout to terminal state (or heuristic evaluation)
4. **Backpropagation**: Update $Q(s,a)$ and visit counts $N(s,a)$

**Tree-of-Thought (ToT)** uses BFS/DFS with a value function; **Reward Balanced Search (Rebase)** allocates a compute budget $P$ to balance exploration/exploitation [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. AlphaProof employs an AND-OR tree search over Lean tactic states with a learned value function, where the return for decomposed subgoals is $G_t = \min(\text{returns of subgoals})$ to enforce all subgoals must be solved [source:nature:olympiad-level-formal-mathematical-reaso]. The **unified framework** of [source:arxiv:2510.09988] formalizes test-time scaling as:

$$
p^* = \arg\max_{p \in \mathcal{A}(\pi, Q, \mathcal{C}_{\text{infer}})} V(p)
$$

where $\mathcal{A}$ is the search algorithm, $\mathcal{C}_{\text{infer}}$ the compute budget, and $V$ the value function.

| Search Method | Compute Profile | Feedback Type | Typical Use Case |
|---------------|-----------------|---------------|------------------|
| Best-of-N / Voting | Parallel, $N\times$ | Outcome (ORM) / None | Quick quality boost |
| Self-Correction | Sequential, iterative | Extrinsic (tools) / Intrinsic | Code, formal proof |
| MCTS / ToT | Tree, adaptive | Process (PRM) / Value net | Hard reasoning, planning |

## Reward Modeling for Search

Search requires a reward signal $R(s,a)$ or $V(s)$. Three families dominate:

1. **Outcome Reward Models (ORMs)**: Score final answer correctness. Cheap but sparse; cannot guide intermediate steps.
2. **Process Reward Models (PRMs)**: Score each reasoning step. ThinkPRM achieves comparable performance with $1\%$ of process supervision data vs discriminative PRMs [source:arxiv:2501.02497]. Generative PRMs (LLM-as-judge) can be trained with $\sim 40$K SFT+DPO samples [source:arxiv:2501.02497].
3. **Generative Critics**: LLMs produce natural language critiques. Training-free (prompting) or SFT/DPO-trained. Used in self-correction loops [source:arxiv:2501.02497].

AlphaProof uses **Lean verification as perfect binary reward** at the tactic level ($r_t = -1$ per tactic to incentivize brevity) [source:nature:olympiad-level-formal-mathematical-reaso]. The unified framework treats reward as a **unified signal** serving two roles [source:arxiv:2510.09988]:
- **Internalization (RL)**: $\theta^* = \arg\max_\theta \mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)] - \lambda \int_{s\in\tau} D_{\text{KL}}(\pi_\theta(\cdot|s)\|\pi_{\mathcal{P}}(\cdot|s)) ds$
- **Externalization (Search)**: $p^* = \arg\max_{p\in\mathcal{P}_{\text{plan}}} \left[ \sum_{t=0}^{T-1} \gamma^t R_{\text{ext}}(s_t,a_t) + \mathcal{H}_\theta(s_T,p) \right]$

**Disagreement**: [source:arxiv:2501.02497] emphasizes PRMs as critical for step-wise search guidance; [source:nature:olympiad-level-formal-mathematical-reaso] shows AlphaProof succeeds with only sparse outcome rewards (Lean proof completion) plus a learned value function for intermediate states, suggesting PRMs may be unnecessary when a perfect verifier exists. [source:arxiv:2510.09988] argues the reward design must match the search algorithm: MCTS needs a value function $Q(s,a)$, while BoN only needs a terminal score.

## RL and Search Interplay: The Dual Optimization Paradigm

The central insight is that **RL and search are two optimizers for one objective** [source:arxiv:2510.09988]. RL *internalizes* search behavior into policy weights $\theta$ (training-time scaling in latent space $\Theta$). Search *externalizes* computation at test time to find $p^*$ for a specific problem $Q$ (test-time scaling in objective space $\mathcal{P}(Q)$). This creates a flywheel:

1. **Search generates high-quality traces** $\tau^*$ for hard problems (using current policy $\pi_\theta$ as proposal + reward guidance).
2. **RL trains on $\tau^*$** (SFT or policy gradient) to improve $\pi_\theta$, reducing the need for search at future test time.
3. **Improved $\pi_\theta$ enables more efficient search** (better proposals, better value estimates).

**STaR** [source:arxiv:2203.14465] implements this loop: generate rationales $\to$ filter by correctness $\to$ fine-tune on correct rationales $\to$ iterate. The objective is $\nabla J = \sum_i \mathbb{E}[\mathbb{1}(\hat{y}_i=y_i) \nabla \log p_M(\hat{y}_i,\hat{r}_i|x_i)]$—a policy gradient with binary reward. **Rationalization** (conditioning on ground-truth answer to generate rationale) prevents plateauing on unsolved problems. **Quiet-STaR** [source:arxiv:2403.09629] generalizes to *every token position*: generate parallel rationales $\to$ reward by log-likelihood improvement on future tokens $\to$ REINFORCE update. The reward is $r_j = \log p_{j:j+n_{\text{true}}}^{\text{talk}}(X_{j+1:j+n_{\text{true}}+1}) - \log \overline{p}_{j:j+n_{\text{true}}}^{\text{talk}}(\cdots)$.

**ReST-MCTS** [source:arxiv:2510.09988] uses MCTS to generate training data for iterative SFT/RL, achieving SOTA on MATH, GPQA, CEval. **AlphaProof** [source:nature:olympiad-level-formal-mathematical-reaso] runs a massive RL loop: Gemini auto-formalizes 1M NL problems $\to$ 80M Lean problems $\to$ distributed actors use MCTS to prove/disprove $\to$ Lean-verified outcomes update proof network (policy + value). At **test time**, AlphaProof performs *Test-Time RL (TTRL)*: generates millions of problem variants $\to$ runs focused RL on them to adapt to the target problem structure.

**Joint Space Optimization** [source:arxiv:2510.09988] extends the duality: search should optimize over both **Prompt Space $\mathcal{P}$** (reasoning structure/template) and **Answer Space $\mathcal{S}$** (solution trace given template): $s^* = \arg\max_{p\in\mathcal{P}, s\in\mathcal{S}_p} V(s)$. Most current methods fix $p$ (single prompt template), searching only $\mathcal{S}$—a key limitation.

## Inference Scaling Laws and Compute-Optimal Tradeoffs

Empirical evidence suggests **smaller models with more test-time tokens can outperform larger models with fewer tokens** [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. This inverts the training-time scaling law (where larger models are more compute-efficient). The compute-optimal inference strategy depends on the task difficulty distribution: easy tasks need only System-1; hard tasks justify System-2 search. No universal test-time scaling law exists yet [source:arxiv:2501.02497]. DeepSeek-R1-style long CoT models regularly exceed 10,000 tokens/response [source:wellecks:l1-controlling-how-long-a-reasoning-mode], making token budget a first-order cost factor.

## Length Control and Budget-Aware Reasoning (L1)

Unbounded CoT length causes cost explosion. **L1** [source:wellecks:l1-controlling-how-long-a-reasoning-mode] uses RL to adhere to length constraints in the prompt (e.g., "use up to 1000 tokens") by adding a length penalty to the reward. The model learns to express uncertainty ("Wait..."), backtrack ("Alternatively..."), and self-verify within the budget. This shifts the control from *implicit* (model decides when to stop) to *explicit* (user specifies budget). AlphaProof's $r_t=-1$ per tactic is a hard-coded length penalty; L1 makes it a controllable hyperparameter.

## Case Studies: AlphaProof and AlphaGeometry 2

| Component | AlphaProof | AlphaGeometry 2 |
|-----------|------------|-----------------|
| **Domain** | Algebra, Number Theory | Geometry |
| **Formalism** | Lean 4 | Custom symbolic engine + NL |
| **Policy** | 3B encoder-decoder (proof network) | Gemini-based LM (trained from scratch) |
| **Search** | AND-OR MCTS with learned value net | Neuro-symbolic: LM proposes constructions, engine verifies |
| **Reward** | $r_t=-1$ per tactic; $G_t=\min(\text{subgoal returns})$ | Symbolic engine verification (binary) |
| **Training** | 300B tokens pretrain $\to$ 300K SFT $\to$ RL on 80M problems | 10$\times$ synthetic data vs AlphaGeometry 1 |
| **Test-Time RL** | Generates problem variants, runs focused RL | Knowledge-sharing across search trees |
| **IMO 2024** | 3/5 non-geometry (incl. hardest) | 1/1 geometry in 19 sec |
| **Combined** | **28/42 (Silver)** | |

AlphaGeometry 2's symbolic engine is **2 orders of magnitude faster** than v1, enabling deeper search [source:deepmind:alphageometry-2-new-geometry-reasoning-s]. Both systems required **manual formalization** of IMO problems—a major deployment bottleneck. Combinatorics problems remain unsolved by either system [source:deepmind:alphageometry-2-new-geometry-reasoning-s; nature:olympiad-level-formal-mathematical-reaso].

## Current Status and Trajectory

**Rising**: Test-time search (MCTS, BoN, self-correction with tools) is the default for high-stakes reasoning (math, code, formal proof). The RL-search flywheel (ReST-MCTS, AlphaProof) is the dominant paradigm for pushing frontier capabilities. **Default**: BoN/self-consistency with ORMs is standard for API-based deployment; PRMs are adopted where step-wise control matters (e.g., process supervision for math). **Fading/Unsettled**: Pure intrinsic self-correction (no tools) shows inconsistent gains and is not widely reported as reliable for hard reasoning [source:arxiv:2501.02497; wellecks:l1-controlling-how-long-a-reasoning-mode]. Prompt-space search (optimizing reasoning structure $p$) is largely unexplored beyond fixed templates [source:arxiv:2510.09988]. **Critical gap**: No universal inference scaling law exists to predict compute-optimal search configuration for a given task/model/verifier triplet [source:arxiv:2501.02497]. System-2 models struggle to generalize to non-symbolic reasoning (e.g., open-ended analysis, creative writing) [source:arxiv:2501.02497].

## Key Takeaways

- Test-time compute splits into **TTA (System-1 robustness)** and **Test-Time Reasoning (System-2 search)**; only the latter induces deliberate multi-step reasoning.
- **Search strategies form a hierarchy**: repeated sampling (parallel, easy) $\to$ self-correction (sequential, needs extrinsic feedback) $\to$ tree search (structured, needs value function).
- **Reward design must match search algorithm**: BoN needs terminal score; MCTS needs step-wise $Q(s,a)$; RL needs differentiable/low-variance reward.
- **RL and search are dual optimizers**: RL internalizes, search externalizes. The flywheel (search $\to$ training data $\to$ better policy $\to$ better search) drives frontier progress (STaR, ReST-MCTS, AlphaProof).
- **Length control is becoming explicit**: L1-style budget-aware RL replaces implicit EOS prediction; AlphaProof's per-step penalty is a special case.
- **Formal verification (Lean) provides perfect rewards** but requires manual formalization and massive compute (AlphaProof: days per problem).
- **Inference scaling laws are unknown**: compute-optimal tradeoff between model size, token budget, and search width/depth is empirical, not predicted.

## Related Topics

- [RL for reasoning models](rl-for-reasoning.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)

## References
- [source:arxiv:2203.14465] [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- [source:arxiv:2403.09629] [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
- [source:nature:olympiad-level-formal-mathematical-reaso] [Olympiad-level formal mathematical reasoning with reinforcement learning (AlphaProof)](https://www.nature.com/articles/s41586-025-09833-y)
- [source:arxiv:2304.01132] [Search-in-the-Chain: Interactively Enhancing Large Language Models with Search](https://arxiv.org/abs/2304.01132)
- [source:arxiv:2501.02497] [Test-Time Compute: from System-1 Thinking to System-2 Reasoning](https://arxiv.org/html/2501.02497v2)
- [source:wellecks:l1-controlling-how-long-a-reasoning-mode] [L1: Controlling how long a reasoning model thinks with reinforcement learning](https://wellecks.com/data/welleck2025scifm_tutorial.pdf)
- [source:arxiv:2510.09988] [Unifying Tree Search Algorithm and Reward Design for LLM Reasoning](https://arxiv.org/abs/2510.09988)
- [source:deepmind:alphageometry-2-new-geometry-reasoning-s] [AlphaGeometry 2: New geometry reasoning system](https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/)
