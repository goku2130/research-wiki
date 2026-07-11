---
title: Agentic and tool-use RL
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2510.01132
- arxiv:2509.02547
- arxiv:2604.09459
- alphaxiv:triage-role-typed-credit-assignment-for-
- github:awesome-rl-based-agentic-search-papers
- research:advancing-search-augmented-language-mode
- bespokelabs:improving-multi-turn-tool-use-with-reinf
open_questions:
- No large-scale, controlled comparison exists between PPO, GRPO, and newer credit-assignment
  methods (GiGPO, CARL, IGPO) on identical agentic benchmarks with matched compute
  budgets.
- The conditions under which sparse binary rewards suffice vs. dense verified rewards
  are required remain empirically unmapped across task families.
- Whether turn-level credit assignment (AgentPRM, CARL) can be combined with hierarchical
  critics (ArCHer) without prohibitive compute overhead is untested.
- The "echo trap" in GRPO is documented but its mitigation via token-level IS (Perplexity)
  has not been ablated against turn-level CA methods.
---

Agentic RL reframes LLM fine-tuning as sequential decision-making in partially observable environments, replacing the single-turn preference optimization of RLHF with multi-turn tool use, search, and reasoning loops. The field has converged on a POMDP formalism where credit assignment across 100K–1M token trajectories—not token-level advantage estimation—is the central algorithmic bottleneck.

## Formalization: From Degenerate MDP to Agentic POMDP

The landscape survey formalizes the paradigm shift by contrasting the underlying decision processes [source:arxiv:2509.02547]. **Preference-Based RL Fine-Tuning (PBRFT)** collapses to a single-step MDP with horizon $T=1$ and $\gamma=1$:

$$
\langle \mathcal{S}_{\text{trad}}, \mathcal{A}_{\text{trad}}, P_{\text{trad}}, R_{\text{trad}}, T=1, \gamma=1 \rangle, \quad J_{\text{trad}}(\theta) = \mathbb{E}_{a \sim \pi_\theta}[r(a)]
$$

**Agentic RL** lifts this to a Partially Observable MDP where the agent receives observations $o_t = O(s_t)$, the action space splits into natural language and structured tool invocations $\mathcal{A}_{\text{agent}} = \mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{action}}$, and the objective maximizes discounted cumulative reward over horizon $T$:

$$
\langle \mathcal{S}_{\text{agent}}, \mathcal{A}_{\text{agent}}, P_{\text{agent}}, R_{\text{agent}}, \gamma, O \rangle, \quad J_{\text{agent}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \gamma^t R_{\text{agent}}(s_t, a_t) \right], \quad 0 < \gamma < 1
$$

This formalism makes explicit five compounding difficulties absent in PBRFT [source:arxiv:2604.09459]:
1. **Stochastic environment transitions** (non-deterministic API/tool responses)
2. **Partial observability** (agent sees only $o_t$, not $s_t$)
3. **Long horizons** (10–100+ turns, 100K–1M tokens)
4. **Action heterogeneity** (mixing planning, tool calls, formatting)
5. **Non-verifiable intermediate states** (no ground truth for intermediate turns)

The survey identifies six optimizable capability modules enabled by this formalism: Planning, Tool Use, Memory, Self-Improvement, Reasoning (System 1 vs System 2), and Perception [source:arxiv:2509.02547].

## Multi-Turn Tool Use: Environments, Rewards, and Algorithm Selection

### Environment Design and Curriculum

The Practitioner's Guide establishes that **curriculum learning across environment complexity** is necessary for skill acquisition [source:arxiv:2510.01132]. Agents trained on simpler TextWorld configurations (lower spatial/object complexity) develop reusable state-tracking and exploration skills that generalize to harder settings. On TextWorld (w2-o3-q4), Qwen-1.5B improved from 17% base to 88% with PPO, but performance dropped to 51% on the most complex setting (w8-o12-q4). Larger models handle complexity better: Qwen-7B achieved 72% on w4-o6-q8. **Cross-domain SFT initialization (e.g., ALFWorld → TextWorld) causes rapid policy collapse** due to conflicting behavioral biases [source:arxiv:2510.01132].

### Reward Granularity: Dense Verified vs. Sparse vs. Model-Based

Reward design is the highest-leverage design choice. On SWE-Gym software engineering tasks [source:arxiv:2510.01132]:
- **Sparse binary reward**: 4.2% success
- **Dense, execution-verified ratio rewards** (e.g., unit test pass ratio): 22% success
- **Model-based judges**: CodeRM-8B 7.2%, GPT-4.1 9.3%

The guide concludes model-based judges are **not yet a viable substitute for execution-based verifiers** in complex reasoning. However, the Perplexity search-agent pipeline uses a **gated aggregation** reward that combines binary correctness with a learned preference score [source:research:advancing-search-augmented-language-mode]:

$$
R(\tau_i) = r_{\text{base}}(\tau_i) \cdot s(\tau_i) - \text{pen}_{\text{eff}}(\tau_i)
$$

where $r_{\text{base}} \in \{0,1\}$ is verifiable correctness, $s(\tau_i) \in [0,1]$ is a Bradley-Terry preference score, and $\text{pen}_{\text{eff}}$ penalizes excess tool calls and length via group-relative anchored penalties. This design prevents reward hacking by making correctness a prerequisite for preference credit.

**DISAGREEMENT**: The Practitioner's Guide reports dense verified rewards dramatically outperform sparse binary (22% vs 4.2%) and model-based judges (7–9%) [source:arxiv:2510.01132]. Yet Bespoke Labs achieved 55%→78% on BFCL multi-turn tool use using **only a sparse binary reward** (pass/fail BFCL eval) with GRPO [source:bespokelabs:improving-multi-turn-tool-use-with-reinf]. Perplexity's gated design explicitly avoids sparse-reward variance by requiring $r_{\text{base}}=1$ before applying $s(\tau)$ [source:research:advancing-search-augmented-language-mode]. The discrepancy likely stems from task structure: BFCL evaluates discrete API-call correctness (verifiable per-turn), while SWE-Gym requires extended code-editing trajectories where intermediate progress signals are essential.

### Algorithm Selection: PPO vs. GRPO vs. RLOO

The Practitioner's Guide reports a striking algorithm hierarchy on TextWorld w2-o3-q4 (Qwen-1.5B) [source:arxiv:2510.01132]:
- **PPO**: 88% success
- **RLOO**: 51%
- **Reinforce++ / GRPO**: ~18% (negligible gains)

The clipped PPO objective used in the study follows the standard formulation with distinct sample and time indices:

$$
L^{\text{CLIP}}(\theta) = \frac{1}{N} \sum_{j=1}^{N} \sum_{t=0}^{T_j} \min\Bigl( r_{j,t}(\theta) \hat{A}_{j,t},\; \text{clip}\bigl(r_{j,t}(\theta), 1-\epsilon, 1+\epsilon\bigr) \hat{A}_{j,t} \Bigr)
$$

or equivalently $\mathbb{E}_{j \sim [N],\, t \sim [T_j]}\bigl[ \min( r_{j,t}(\theta) \hat{A}_{j,t},\; \text{clip}(r_{j,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{j,t} ) \bigr]$, where $r_{j,t}(\theta) = \frac{\pi_\theta(a_{j,t} \mid h_{j,t})}{\pi_{\theta_{\text{old}}}(a_{j,t} \mid h_{j,t})}$ is the probability ratio for token $t$ in trajectory $j$.

**DISAGREEMENT**: This result contradicts the broader adoption of GRPO in agentic search (Perplexity [source:research:advancing-search-augmented-language-mode], Bespoke Labs [source:bespokelabs:improving-multi-turn-tool-use-with-reinf]) and its listing as a primary algorithmic family in the landscape survey [source:arxiv:2509.02547]. The credit assignment survey notes GRPO assigns **identical credit to all actions in a trajectory**, leading to an "echo trap" where models converge to repetitive safe behaviors [source:arxiv:2604.09459]. The Practitioner's Guide's negative GRPO result may reflect its specific hyperparameters or the TextWorld environment's need for fine-grained credit; Perplexity mitigates GRPO's variance with token-level importance sampling and a two-stage SFT→RL pipeline [source:research:advancing-search-augmented-language-mode].

### SFT–RL Budget Allocation

For a fixed compute budget, a **hybrid SFT→RL schedule dominates pure SFT or pure RL** [source:arxiv:2510.01132]. With 60 SFT demonstrations + 400 RL episodes: 85% in-domain success, 59% generalization to complex tasks. Pure SFT (same budget): 55% generalization. Pure RL: 11% generalization. The guide recommends **strong SFT priors from minimal demonstrations** to accelerate convergence and stabilize policy initialization.

## Search-Augmented RL: Agentic Search Systems

### Problem Formulation

Agentic search reframes information-seeking as a dynamic decision loop where the agent must learn [source:github:awesome-rl-based-agentic-search-papers]:
1. **When** to initiate search
2. **How intensively** to search (depth, query volume)
3. **How to integrate** retrieved evidence
4. **How to revise** queries based on intermediate results

Standard RAG and prompt engineering are insufficient for autonomous research-grade seeking.

### Optimization Strategies

The awesome-rl-based-agentic-search-papers taxonomy categorizes the design space [source:github:awesome-rl-based-agentic-search-papers]:
- **Reward modeling**: Outcome Reward Models (ORM, final answer) vs Process Reward Models (PRM, intermediate steps)
- **Reward function**: Rule-based (EM, F1) vs LLM-based (judge)
- **Algorithms**: GRPO (dominant), PPO, DPO, SFT (cold-start), Meta-RL
- **Targeted capabilities**: Search Efficiency, Retrieval-Search Interaction, Context Memory, Structural Navigation
- **Environments**: Simulated repos, live web, synthetic scientific corpora

Key benchmarks: Multi-hop QA (HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle), General Web (NQ, TriviaQA, PopQA, GAIA), Specialized (SWE-bench Lite, LongBench v2, HLE-Bio/Chem-Gold).

### Perplexity's Two-Stage Pipeline

Perplexity's production system disentangles deployment guardrails from search optimization via [source:research:advancing-search-augmented-language-mode]:

**Stage 1 (SFT)**: Qwen3-family model trained on:
- Preference-oriented data (tone, clarity, formatting, language consistency)
- Production-format tool-use trajectories (single/multi-turn, annotated with tool harness)

**Stage 2 (On-policy GRPO with token-level IS)**: Heterogeneous mixture:
- 90% **Verifiable Search-Agent QA**: Synthetic multi-hop via seed selection → chain construction (2–4 hops) → name-free synthesis → independent verification
- 10% **Rubric-based Chat**: Atomic objective rubrics calibrated via pass@4 filter

**Results**: Qwen3.5-Large-SFT-RL matches/exceeds `gpt-5.4` on FRAMES and Facts Open. Internal preference metric `pplx-sbs-search` improved 0.602 → 0.742. Training stable: gradient norms controlled, train-inference KL ~$10^{-3}$. Tool-call frequency and length penalties decreased monotonically.

**Limitations noted**: Joint optimization in single RL stage fails; RL-only underperforms guardrails; SFT alone compromises search; small reward models fail fine-grained preferences and reinforce bad behaviors [source:research:advancing-search-augmented-language-mode].

## Credit Assignment in Agentic RL: Taxonomy and Methods

The credit assignment survey organizes 47 methods across **Granularity** (token, segment, step/turn, multi-agent) × **Methodology** (Monte Carlo, Temporal Difference, Model-based/LLM-as-Critic, Game-theoretic, Information-theoretic) [source:arxiv:2604.09459].

### Reasoning RL Recipes (Single-Turn, 500–30K tokens)

| Granularity | Method | Mechanism | Key Formula |
|-------------|--------|-----------|-------------|
| **Token** | VinePPO | MC: fork $K$ continuations ("vines") per token to estimate value without critic | $V(s_t) \approx \frac{1}{K}\sum_{k=1}^K R(\tau^{(k)})$ |
| **Token** | RED | Linear regression probes on reward model hidden states | — |
| **Segment/Step** | SPO | Partition chain into semantic segments | — |
| **Segment/Step** | PRM | Step-level supervision via process reward model | — |
| **Segment/Step** | PURE | "Min-form" credit to prevent hacking | $V(s_t) = \mathbb{E}[\min_{t' \ge t} r_{t'}]$ |
| **Self-Supervised** | SPRO | Leave-one-out "Masked Step Advantage" | $c_i = P(\text{correct}\|\text{full}) - P(\text{correct}\|\text{without step } i)$ |

**Quantitative**: SPRO achieves $3.4\times$ training efficiency over GRPO [source:arxiv:2604.09459].

### Agentic RL Recipes (Multi-Turn, 100K–1M tokens)

| Granularity | Method | Mechanism | Key Formula |
|-------------|--------|-----------|-------------|
| **Turn** | AgentPRM | TD+GAE replaces expensive MC labeling | $V(s_t) \leftarrow V(s_t) + \alpha[r_t + \gamma V(s_{t+1}) - V(s_t)]$ |
| **Turn** | HCAPO | LLM critic "generative verification" in imagination | — |
| **Turn** | C3 / CCPO | Leave-one-out or structural causal model for ATE of a turn | — |
| **Hierarchical** | ArCHer | High-level turn critic + low-level token actor | — |
| **Sparse** | CARL | Identify bifurcation points via action entropy $H(\pi(\cdot\|s_t))$; update only critical actions | — |
| **Info-Theoretic** | IGPO | Credit = information gain about success | $c_t = \log P(\text{success}\|h_{1:t}) - \log P(\text{success}\|h_{1:t-1})$ |

**Quantitative**: AgentPRM $8\times$ sample efficiency vs MC-PRM; CARL reduces gradient updates 72% no degradation; GiGPO +12% ALFWorld, +9% WebShop over GRPO [source:arxiv:2604.09459].

### Multi-Agent Credit Assignment

| Method | Mechanism |
|--------|-----------|
| M-GRPO | Decompose inter-agent (team) + intra-agent (individual) credit |
| SHARP | Shapley value decomposition across agents |
| MAPPA | AI feedback for per-action process rewards |

**Quantitative**: SHARP +23.7% over single-agent, +14.1% over multi-agent baselines; MAPPA +5.0–17.5pp AIME, +7.8–17.2pp AMC [source:arxiv:2604.09459].

### Limitations of Current CA Methods

- **MC computational overhead**: VinePPO scales $O(K \cdot L)$, prohibitive for long sequences [source:arxiv:2604.09459]
- **LLM-as-Critic bias**: HCAPO, CAPO susceptible to self-evaluation bias [source:arxiv:2604.09459]
- **Evaluation fragmentation**: No shared benchmarks for systematic CA comparison [source:arxiv:2604.09459]
- **Environment constraints**: Reasoning RL methods assume deterministic transitions and verifiable intermediates—violated in agentic settings [source:arxiv:2604.09459]

## Training Recipes and Practical Stabilization

### Bespoke Labs: Minimal-Data Multi-Turn GRPO

Using only 100 training samples from BFCL v3 multi-turn-base, Qwen2.5-7B-Instruct improved 55% → 78% (23% absolute) [source:bespokelabs:improving-multi-turn-tool-use-with-reinf]. Key hyperparameters:
- 4× H200 (3 train, 1 vLLM inference)
- Batch 1, grad accum 4, LR $10^{-6}$ constant, 20 warmup
- $\text{max\_grad\_norm}=0.2$, KL $\beta=0.001$, ref model updated every 100 steps
- $\mu=2$ (1 on-policy + 1 off-policy step/batch)
- **Overlong filtering**: mask loss for rollouts exceeding max completion length

### Critical Failure Modes and Fixes

**Completion length blowup**: Agent produces increasingly long gibberish → accuracy collapse [source:bespokelabs:improving-multi-turn-tool-use-with-reinf].
- **Failed**: Dr.GRPO, gibberish penalties (GPT-4o-mini), removing KL entirely
- **Worked**: Overlong filtering + small KL ($\beta=0.001$)

**Reward hacking**: Complex rewards (tool counts, format checks) worsened stability. **"Less is more"**: binary correctness-only reward yielded best performance [source:bespokelabs:improving-multi-turn-tool-use-with-reinf].

**Reference model dynamics**: Updating reference every 100 steps boosted performance. Hypothesis: stronger policy needs stronger reference to prevent KL from pulling toward initial low-performing state [source:bespokelabs:improving-multi-turn-tool-use-with-reinf].

### Exploration Horizon Effects

- TextWorld: Performance saturates beyond **8 exploration steps** (for 4-step optimal solutions) [source:arxiv:2510.01132]
- SWE-Gym: Increasing tool calls from 10 → 40 increased success 3% → 22% [source:arxiv:2510.01132]

## Current Status and Trajectory

Agentic RL is **rising rapidly but remains pre-standardization**. The POMDP formalism and credit assignment taxonomy are widely cited [source:arxiv:2509.02547][source:arxiv:2604.09459], and production systems (Perplexity [source:research:advancing-search-augmented-language-mode], OpenAI Deep Research [source:arxiv:2509.02547]) demonstrate competitive performance on benchmarks like BrowseComp (51.5% pass@1 for Deep Research) and FRAMES/Facts Open. However, **no single algorithmic recipe has emerged as default**: PPO dominates the Practitioner's Guide's controlled study [source:arxiv:2510.01132], while GRPO powers most search-agent deployments [source:research:advancing-search-augmented-language-mode][source:bespokelabs:improving-multi-turn-tool-use-with-reinf]. The field is fragmented by **evaluation fragmentation**—no shared agentic benchmarks enable systematic comparison [source:arxiv:2604.09459]—and **environment constraints** (API costs, stochasticity, partial observability) that prevent clean ablation. Credit assignment research is shifting from token-level (reasoning RL) to turn-level and hierarchical methods (agentic RL), but **computational overhead of MC methods and bias of LLM critics remain unresolved**. Not widely reported: any large-scale comparison of PPO vs GRPO vs newer methods (GiGPO, CARL, IGPO) on identical agentic benchmarks with identical compute budgets.

## Key Takeaways

- **Formalism shift**: Agentic RL = POMDP with split action space ($\mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{action}}$), discounted cumulative reward, partial observability—fundamentally distinct from PBRFT's $T=1$ MDP [source:arxiv:2509.02547].
- **Credit assignment is the core bottleneck**: Trajectories of 100K–1M tokens with stochastic transitions, non-verifiable intermediates, and action heterogeneity make episode-level methods (GRPO) prone to "echo trap" [source:arxiv:2604.09459].
- **Reward density matters critically**: Execution-verified dense rewards (22%) dramatically outperform sparse binary (4.2%) and model-based judges (7–9%) on complex coding tasks [source:arxiv:2510.01132], but binary rewards suffice for discrete API-calling benchmarks [source:bespokelabs:improving-multi-turn-tool-use-with-reinf].
- **Curriculum + SFT initialization + stabilized PPO** is the Practitioner's Guide's winning recipe for TextWorld/ALFWorld/SWE-Gym; **SFT→GRPO with gated rewards + token-level IS** is Perplexity's production recipe for web search [source:arxiv:2510.01132][source:research:advancing-search-augmented-language-mode].
- **Turn-level CA methods (AgentPRM, CARL, IGPO) show large efficiency gains** (8× sample efficiency, 72% fewer updates) over token-level reasoning RL methods when transferred to agentic settings [source:arxiv:2604.09459].
- **Practical stability requires**: overlong filtering, small KL with periodic reference updates, minimal reward complexity, and synthetic verifiable data pipelines [source:bespokelabs:improving-multi-turn-tool-use-with-reinf][source:research:advancing-search-augmented-language-mode].

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)

## References
- [source:arxiv:2510.01132] [A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning](https://arxiv.org/html/2510.01132v1)
- [source:arxiv:2509.02547] [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547)
- [source:arxiv:2604.09459] [From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models](https://arxiv.org/html/2604.09459v1)
- [source:alphaxiv:triage-role-typed-credit-assignment-for-] [TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning](https://www.alphaxiv.org/abs/2606.32017)
- [source:github:awesome-rl-based-agentic-search-papers] [Awesome-RL-based-Agentic-Search-Papers](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers)
- [source:research:advancing-search-augmented-language-mode] [Advancing Search-Augmented Language Models](https://research.perplexity.ai/articles/advancing-search-augmented-language-models)
- [source:bespokelabs:improving-multi-turn-tool-use-with-reinf] [Improving Multi-Turn Tool Use with Reinforcement Learning](https://bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)
