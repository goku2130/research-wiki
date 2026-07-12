---
title: Self-improvement and self-play RL
maturity: comprehensive
updated: '2026-07-12'
sources:
- openai:iterated-distillation-and-amplification-
- arxiv:2212.08073
- arxiv:1810.08575
- arxiv:2505.03335
- arxiv:2308.08998
- arxiv:2203.14465
- arxiv:2402.04683
- arxiv:2212.10560
- openai:iterated-amplification-supervising-stron
- semanticscholar:machine-learning-projects-for-iterated-d
- aclanthology:re-rest-reflection-reinforced-self-train
open_questions:
- Can STaR-style rationale bootstrapping work without any ground-truth answers, using
  only a verifier (e.g., code execution) as in Absolute Zero?
- Will ReST's growing-batch offline RL paradigm re-emerge as the dominant LLM post-training
  recipe under a different name (e.g., iterative DPO, online DPO)?
- Can Re-ReST's reflector mechanism be combined with Absolute Zero's proposer/solver
  to create a self-play loop with reflection on environmental feedback?
- Does IDA's recursive decomposition have a practical implementation for LLMs that
  doesn't require human decomposers (e.g., using the model itself as $H'$)?
---

Self-improvement and self-play RL methods enable language models to enhance their reasoning and alignment capabilities by generating their own training data, rewards, or task decompositions, reducing reliance on human annotation. These techniques span from rationale bootstrapping (STaR) and reward-model-guided self-training (ReST) to fully autonomous task proposal and solving (Absolute Zero), recursive task decomposition (Iterated Amplification), and reflection-reinforced self-training (Re-ReST).

## STaR: Self-Taught Reasoner

STaR (Self-Taught Reasoner) bootstraps chain-of-thought reasoning by iteratively generating rationales for problems with known answers, filtering for correctness, and fine-tuning on the resulting data [source:arxiv:2203.14465]. Given a pretrained model $M$, a dataset $\mathcal{D} = \{(x_i, y_i)\}$, and a small prompt set $\mathcal{P}$ with rationales, STaR repeats: (1) generate rationale $\hat{r}_i$ and answer $\hat{y}_i$ for each $x_i$; (2) keep only samples where $\hat{y}_i = y_i$; (3) for failures, use "rationalization" — prompt with the correct answer $y_i$ to generate a justification $\hat{r}_i^{\text{rat}}$; (4) fine-tune the *original* model $M$ on the combined correct rationales; (5) iterate until plateau [source:arxiv:2203.14465].

The method approximates a policy gradient objective where the reward is the indicator of answer correctness. The expected reward is $J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot|x_i)} [\mathbb{1}(\hat{y}_i = y_i)]$, and its gradient uses the log-derivative trick: $\nabla J = \sum_i \mathbb{E}[\mathbb{1}(\hat{y}_i = y_i) \nabla \log p_M(\hat{y}_i, \hat{r}_i | x_i)]$ [source:arxiv:2203.14465]. Filtering implements the indicator, discarding gradients from incorrect rationales.

Key results on GPT-J (6B): CommonsenseQA accuracy reached **72.5%** (vs. 60.0% direct fine-tuning, 36.6% few-shot CoT), approaching GPT-3 30$\times$ larger at 73.0% [source:arxiv:2203.14465]. On $n$-digit addition, 16 iterations yielded **89.5%** overall (vs. 76.3% without rationales); Rationalization allowed the model to learn multiple digit lengths simultaneously [source:arxiv:2203.14465]. GSM8K reached **10.7%** (vs. 5.8% direct, 3.0% few-shot) [source:arxiv:2203.14465].

Limitations: requires initial few-shot performance above chance (GPT-2 failed on arithmetic); high chance-performance settings (e.g., binary) admit spurious rationales; rationalization hint formatting can be non-trivial [source:arxiv:2203.14465].

## ReST: Reinforced Self-Training

ReST (Reinforced Self-Training) frames self-improvement as a "growing batch" RL algorithm alternating between a **Grow** step (sampling from current policy to augment data) and an **Improve** step (offline RL on the augmented set filtered by a reward model) [source:arxiv:2308.08998]. This decouples exploration (Grow) from exploitation (Improve), aiming to combine online RL's exploration with offline RL's efficiency.

**Initialization**: Supervised fine-tuning on dataset $\mathcal{D}$ with NLL loss $\mathcal{L}_{\text{NLL}}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}}[\sum_t \log \pi_\theta(y_t|y_{<t}, x)]$ [source:arxiv:2308.08998].

**Grow step**: Sample $N_g$ outputs per context $x \sim \mathcal{D}$ from current policy $\pi_\theta$, forming augmented dataset $\mathcal{D}_g = \{(x^i, y^i)\} \cup \mathcal{D}$ [source:arxiv:2308.08998].

**Improve step**: Score $\mathcal{D}_g$ with reward model $R(x,y)$; filter by threshold $\tau$: $F(x,y;\tau) = \mathbb{1}_{R(x,y) > \tau}$; optimize with reward-weighted loss $J(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_g}[F(x,y;\tau) \mathcal{L}(x,y;\theta)]$ [source:arxiv:2308.08998]. Repeat Improve with increasing thresholds $\tau_1 < \tau_2 < \dots$ and decreasing learning rates. Then cycle back to Grow with the new policy.

On machine translation (IWSLT 2014 De-En, WMT 2020 Zh-En, internal Web En-Zh) with MetricX reward: ReST (G=2, I=3) achieved average reward **83.1** vs. supervised baseline **70.9** and online RL **71.6** [source:arxiv:2308.08998]. Second Grow step added **+5.3** on IWSLT and **+0.8** on Web. Best-of-N with $N<10$ matched supervised BC with $N=200$ [source:arxiv:2308.08998]. Simple BC loss outperformed offline RL losses (GOLD, BVMPO, OAC) within ReST [source:arxiv:2308.08998].

Limitations: Reward model is an imperfect proxy; human evals diverge from reward rankings as policy drifts; repeated cycles risk overfitting to reward model; "delusions" observed (e.g., reward increases for repetitive translations) [source:arxiv:2308.08998].

## Re-ReST: Reflection-Reinforced Self-Training for Language Agents

Re-ReST (Reflection-Reinforced Self-Training) addresses the quality gap in self-training for language agents by introducing a "reflector" mechanism that refines inferior model-generated samples using environmental feedback [source:aclanthology:re-rest-reflection-reinforced-self-train].

**Core Problem**: Finetuning language agents using reasoning-action trajectories is highly effective, but acquiring these trajectories is impractical or expensive via human annotations or stronger model demonstrations. Self-training—where the agent generates its own supervision—is limited by the quality of model-generated samples, which are often low-quality in challenging agent tasks.

**Method**: Re-ReST follows a six-step process:
1. **Sample Generation**: The language agent generates initial reasoning-action trajectories for target tasks.
2. **Environmental Feedback**: The agent's outputs are submitted to an external environment to receive feedback (e.g., unit test results for code generation).
3. **Refinement via Reflector**: A reflector takes the agent's original output and the environmental feedback to produce an improved, higher-quality version of the sample.
4. **Dataset Enrichment**: These refined, high-quality samples enrich the self-training dataset, replacing or augmenting the low-quality initial attempts.
5. **Self-Training (Finetuning)**: The agent is finetuned on this enriched dataset of high-quality trajectories.
6. **Inference Application**: Re-ReST employs a method to utilize reflection during inference *without* requiring ground-truth feedback, overcoming a limitation of prior reflection methods.

**Quantitative Results**: Evaluated across sequential decision-making, multi-hop QA, visual QA, text-to-image generation, and code generation:
- **HotpotQA**: Standard self-training improved baseline by **7.6%**; Re-ReST added a further **2.0%** boost.
- **AlfWorld**: Standard self-training improved baseline by **28.4%**; Re-ReST added a further **14.1%** boost.

The reflector efficiently generates high-quality samples that significantly enhance self-training compared to standard self-training alone [source:aclanthology:re-rest-reflection-reinforced-self-train].

**Limitations**: Prior reflection work required ground-truth feedback during inference; Re-ReST resolves this by demonstrating inference-time reflection without such feedback. No other specific limitations of Re-ReST itself are detailed in the source.

## Absolute Zero: Reinforced Self-Play Reasoning with Zero Data

Absolute Zero (AZ) eliminates *all* human-curated question-answer pairs: a single model $\pi_\theta$ acts as both **proposer** ($\pi_\theta^{\text{propose}}$) and **solver** ($\pi_\theta^{\text{solve}}$), generating tasks optimized for its own learnability and solving them with verifiable feedback from a code executor [source:arxiv:2505.03335].

**Loop**: (1) Proposer generates task $\tau$ conditioned on task type and $K$ reference examples from a buffer. (2) Python executor validates: syntax, safety (no `os`/`sys`), determinism (identical outputs across runs). (3) Valid tasks become one of three reasoning modes: **Deduction** (given program $p$ and input $i$, predict output $o$), **Abduction** (given $p$ and $o$, infer $i$), **Induction** (given I/O pairs and description $m$, synthesize $p$) [source:arxiv:2505.03335]. (4) Solver generates answer $y$. (5) Rewards: solver gets binary $r_{\text{solve}} = \mathbb{I}_{(y=y^*)}$; proposer gets learnability reward $r_{\text{propose}} = 0$ if $\bar{r}_{\text{solve}}=0$ else $1 - \bar{r}_{\text{solve}}$ where $\bar{r}_{\text{solve}}$ is average solver success over $G$ rollouts [source:arxiv:2505.03335]. (6) Optimization via **Task-Relative REINFORCE++ (TRR++)** with separate baselines per task-role configuration: normalized advantage $A_{\text{task,role}}^{\text{norm}} = (r - \mu_{\text{task,role}})/\sigma_{\text{task,role}}$ [source:arxiv:2505.03335].

Overall objective:

$$
\mathcal{J}(\theta) := \max_\theta \mathbb{E}_{z\sim p(z)} \left[ \mathbb{E}_{(x,y^*)\sim f_e(\cdot|\tau), \tau\sim \pi_\theta^{\text{propose}}(\cdot|z)} \left[ \lambda r_e^{\text{propose}}(\tau, \pi_\theta) + \mathbb{E}_{y\sim \pi_\theta^{\text{solve}}(\cdot|x)}[ r_e^{\text{solve}}(y, y^*) ] \right] \right]
$$

[source:arxiv:2505.03335].

Results on Qwen2.5 (3B, 7B, 14B) and Llama-3.1-8B: AZR-Coder-7B surpassed "zero" models trained on tens of thousands of human examples by **+1.8 pp** overall average [source:arxiv:2505.03335]. Cross-domain transfer: training on self-proposed code tasks improved math average by **+10.9** (Base-7B) and **+15.2** (Coder-7B) pp [source:arxiv:2505.03335]. Scaling: 3B/7B/14B Coder gains **+5.7/+10.2/+13.2** pp overall [source:arxiv:2505.03335]. Ablation: removing induction or abduction dropped math performance significantly [source:arxiv:2505.03335].

Limitations: Safety "uh-oh moments" (Llama-3.1-8B CoT mentioned "outsmarting" humans); verifier restricted to deterministic programs [source:arxiv:2505.03335].

## Iterated Distillation and Amplification (IDA)

IDA addresses tasks beyond human scale by recursively decomposing them into human-solvable subtasks, using the model to solve subcomponents, and distilling the composite solution back into the model [source:openai:iterated-distillation-and-amplification-] [source:arxiv:1810.08575] [source:openai:iterated-amplification-supervising-stron]. The core assumption: humans cannot solve the full task but *can* decompose it and solve atomic subproblems.

**Process** (Christiano et al. formalization): Four parallel processes [source:arxiv:1810.08575]:
1. **Data Collection**: Human expert $H$ decomposes question $Q$ into subquestions $Q_1,\dots,Q_k$; agent $X$ answers each; $H$ produces final answer $A$; record transcript $\tau = (Q, Q_1, A_1, \dots, Q_k, A_k, A)$.
2. **Human Modeling**: Train predictor $H'$ to imitate $H$'s decomposition and final answer decisions.
3. **Target Generation**: Sample $Q\sim\mathcal{D}$, run $\text{Amplify}^{H'}(X)$ (using $H'$ to decompose, $X$ to answer subquestions) to get $(Q, A)$ pairs.
4. **Agent Training**: Supervised learning on $(Q, A)$ pairs.

$X$ is a Transformer encoder-decoder with pointer network for symbol copying; layer: $y \leftarrow \text{LayerNorm } x + \text{Attention } x$, $z \leftarrow \text{LayerNorm } y + \text{BatchNorm MLP } y$ [source:arxiv:1810.08575].

**Iterated Amplification Recipe** (OpenAI formulation): The process follows a recursive "recipe" to amplify a weak expert (the human) into a strong learner (the AI) [source:openai:iterated-amplification-supervising-stron]:
1. **Initial Training**: Sample small sub-tasks that humans can solve. Train the AI to perform these using human demonstrations.
2. **Task Decomposition**: Sample slightly larger tasks. A human breaks the task down into the smaller components from Step 1.
3. **Composite Solving**: The AI systems from Step 1 solve these smaller components. The human coordinates assembly of these solutions to solve the larger task.
4. **Direct Training**: The solutions from this human-assisted decomposition process become a new training signal. The AI is trained to solve these second-level tasks directly, without human help.
5. **Iteration**: Repeat, using the AI's ability to solve increasingly complex composite tasks to generate training signals for even larger tasks.

Toy algorithmic tasks (permutation powering, sequential assignments, wildcard search, shortest path, union find, expression evaluation): IDA matched supervised learning from ground truth with "modest slowdown" [source:arxiv:1810.08575] [source:openai:iterated-amplification-supervising-stron]. Sample efficiency: only **6k–24k** oracle calls to $H$ (vs. tens of millions for supervised) [source:arxiv:1810.08575]. Compute overhead: ~2$\times$ standard supervised due to running $\text{Amplify}^{H'}(X)$ for targets [source:arxiv:1810.08575].

**Follow-up Work**: Evans, Saunders, and Stuhlmüller (2019) proposed three research projects extending IDA to high school mathematics problems, investigating whether learning to decompose problems via IDA outperforms standard supervised learning [source:semanticscholar:machine-learning-projects-for-iterated-d]. This work aimed to test IDA on more naturalistic reasoning domains beyond algorithmic toys, though the provided source is a proposal/review without reported quantitative results.

Limitations: Experiments used algorithmic oracle instead of real humans; domains limited to combinatorial tasks with ground truth; $X$ trained via SL, but real-world may need RL from amplified reward; success depends on question distribution $\mathcal{D}$ covering all needed subquestions [source:arxiv:1810.08575] [source:openai:iterated-distillation-and-amplification-] [source:openai:iterated-amplification-supervising-stron]. The OpenAI blog notes the technique is in preliminary stages, tested only on simple toy domains with algorithmic signals simulating human decomposition [source:openai:iterated-amplification-supervising-stron].

## Comparative Analysis: Self-Improvement vs. Self-Play vs. Amplification vs. Reflection

| Dimension | STaR | ReST | Re-ReST | Absolute Zero | IDA |
|-----------|------|------|---------|---------------|-----|
| **External supervision** | Ground-truth answers $y_i$; few-shot rationales $\mathcal{P}$ | Reward model $R(x,y)$ (trained on human prefs) | Environmental feedback (unit tests, etc.); reflector model | None (code executor verifies); seed triplet only | Human expert $H$ (or oracle) for decomposition & atomic answers |
| **Self-generated signal** | Rationales filtered by answer correctness | Samples filtered by reward threshold | Refined trajectories via reflector + env. feedback | Tasks proposed for learnability; solutions verified by executor | Subtask answers from current model; decomposition from $H'$ |
| **Optimization** | Offline SL on filtered rationales (iteratively fine-tunes across outer loops) | Offline RL (BC loss) on filtered samples; growing batch | Offline SL on reflector-enriched dataset | Online RL (TRR++) on proposer+solver joint objective | SL on amplified targets; iterative distillation |
| **Reasoning mode** | Chain-of-thought (forward) | General LM generation (translation) | Reasoning-action trajectories (agentic) | Deduction, abduction, induction (code-based) | Algorithmic decomposition (recursive) |
| **Scalability bottleneck** | Needs initial CoT ability; ground-truth answers required | Reward model generalization & overfitting | Quality of reflector & environmental feedback | Code executor scope (deterministic only); safety | Human decomposition capacity; distribution coverage |
| **Key theoretical framing** | Policy gradient with indicator reward | Growing-batch offline RL | Reflection + self-training loop | Self-play with learnability reward | HCH (Human Consulting HCH) approximation |

**Disagreements / tensions**: 
- STaR and ReST both use filtering but STaR's filter is *ground-truth correctness* (exact, sparse) while ReST's is a *learned reward model* (dense, proxy) [source:arxiv:2203.14465] [source:arxiv:2308.08998]. ReST explicitly argues offline RL losses (GOLD, BVMPO) underperform simple BC within its framework [source:arxiv:2308.08998]; STaR's gradient derivation suggests REINFORCE with baseline could be used but they implement SL on filtered data [source:arxiv:2203.14465].
- Absolute Zero claims "zero data" but relies on a *code executor* as an external verifier — a form of environment supervision distinct from human labels [source:arxiv:2505.03335]. IDA requires human decomposition, which Christiano et al. acknowledge is untested on real-world "messy" tasks [source:arxiv:1810.08575].
- STaR iteratively fine-tunes the model across outer loops [source:arxiv:2203.14465]; ReST continues training the same policy across Grow/Improve cycles with decreasing LR [source:arxiv:2308.08998]; Absolute Zero uses TRR++ with per-task baselines to stabilize online RL [source:arxiv:2505.03335]; Re-ReST finetunes on a reflector-enriched dataset in a single-stage offline loop [source:aclanthology:re-rest-reflection-reinforced-self-train].
- On generalization: STaR shows cross-task transfer within domain (arithmetic iterations help larger digits) [source:arxiv:2203.14465]; Absolute Zero demonstrates *cross-domain* transfer (code $\to$ math +10–15 pp) [source:arxiv:2505.03335]; ReST evaluates only within translation; IDA only on algorithmic toys; Re-ReST shows gains on diverse agent tasks (HotpotQA, AlfWorld, VQA, code) but cross-domain transfer not explicitly measured [source:aclanthology:re-rest-reflection-reinforced-self-train].
- **New tension**: Re-ReST introduces a *reflector* as a separate component that uses environmental feedback to refine samples, whereas Absolute Zero's proposer/solver and ReST's reward model operate without an explicit reflection step. Re-ReST's inference-time reflection without ground-truth feedback [source:aclanthology:re-rest-reflection-reinforced-self-train] contrasts with prior reflection methods that required such feedback — a potential disagreement with earlier reflection literature not captured in the existing sources.

## Current status and trajectory

**STaR**: Rising as a foundational rationale-distillation technique; its filtering+rationalization loop is widely cited and adapted (e.g., in RLVR pipelines). However, dependence on ground-truth answers limits "zero" settings. Not widely reported as a standalone SOTA method for frontier models, but its components (self-generated CoT, correctness filtering) are default in many post-training recipes.

**ReST**: Fading as a named algorithm; the Grow/Improve paradigm persists in spirit (e.g., iterative DPO, online DPO, RL with periodic data refresh) but the specific ReST formulation (BC loss on reward-thresholded samples, increasing thresholds) is less common than PPO/GRPO/DPO variants. The finding that simple BC outperforms complex offline RL losses on filtered data remains influential [source:arxiv:2308.08998].

**Re-ReST**: Emerging as a promising direction for language agent self-training; the reflector + environmental feedback loop addresses a key bottleneck (low-quality self-generated samples) and demonstrates strong gains on agent benchmarks (AlfWorld +14.1%, HotpotQA +2.0% over standard self-training). The inference-time reflection without ground-truth feedback is a notable advance. Not yet widely adopted in frontier model post-training but conceptually aligned with "self-correction" and "critique-revision" trends [source:aclanthology:re-rest-reflection-reinforced-self-train].

**Absolute Zero**: Rising rapidly; represents the frontier of "zero human data" reasoning. The self-play proposer/solver with code verification and learnability reward is an active research direction (e.g., follow-ups on task diversity, safety, stochastic programs). Scaling trends (+13 pp at 14B) suggest continued investment. Safety "uh-oh moments" and deterministic-only verifier are open constraints [source:arxiv:2505.03335].

**IDA**: Fading as a practical training recipe for LLMs; the recursive decomposition ideal remains conceptually central to alignment theory (HCH, scalable oversight) but the algorithmic implementation (Transformer pointer nets, algorithmic oracles, SL distillation) has not translated to LLM post-training at scale. Current "scalable oversight" work (e.g., debate, recursive reward modeling) inherits the motivation but not the method. The Evans et al. (2019) proposal to apply IDA to mathematics [source:semanticscholar:machine-learning-projects-for-iterated-d] did not yield widely reported follow-up results in the LLM era. Not widely reported in recent LLM RL literature.

**Overall trajectory**: The field is converging on *verifiable-reward self-play* (Absolute Zero style) and *iterative self-training with learned rewards* (ReST spirit via DPO/GRPO) for reasoning, while *reflection-reinforced self-training* (Re-ReST) emerges for agentic settings with environmental feedback. IDA's decomposition paradigm lives on in theoretical alignment research rather than engineering practice. STaR's rationale bootstrapping is a standard building block.

## Key takeaways

- **STaR** proves that filtering model-generated CoT by ground-truth correctness, augmented with rationalization, can bootstrap strong reasoning from minimal seeds — but requires answer labels and initial CoT competence [source:arxiv:2203.14465].
- **ReST** demonstrates that a growing-batch offline RL loop (sample → filter by reward model → BC) beats online PPO and static offline RL on translation, with simple BC loss sufficing on filtered data [source:arxiv:2308.08998].
- **Re-ReST** shows that a reflector using environmental feedback (e.g., unit tests) can significantly boost self-training for language agents (+14.1% on AlfWorld, +2.0% on HotpotQA over standard self-training), and enables inference-time reflection without ground-truth feedback [source:aclanthology:re-rest-reflection-reinforced-self-train].
- **Absolute Zero** achieves SOTA "zero-data" reasoning by self-play proposer/solver with code execution verification and a learnability reward ($1 - \text{solve rate}$), showing strong cross-domain transfer (code $\to$ math) and scaling with model size [source:arxiv:2505.03335].
- **IDA** establishes the theoretical framework for amplifying weak supervision via recursive decomposition and distillation, but practical LLM adoption is blocked by reliance on human decomposers and algorithmic toy domains [source:arxiv:1810.08575] [source:openai:iterated-distillation-and-amplification-] [source:openai:iterated-amplification-supervising-stron]. Follow-up proposals to extend to mathematics [source:semanticscholar:machine-learning-projects-for-iterated-d] remain unverified at scale.
- **Unifying thread**: All five methods replace human annotation with model-generated signals (rationales, samples, tasks, subtask answers, reflected trajectories) filtered or verified by an external oracle (ground truth, reward model, code executor, human decomposer, environment + reflector). The trend is toward *weaker oracles* (code executor > reward model > human decomposer > ground truth) and *stronger self-generation* (proposer/solver > sampler > rationale generator > subproblem solver > reflector-refined agent).

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [KL regularization in RLHF](kl-regularization.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [DPO variants deep-dive](dpo-variants.md)
- [RLAIF (RL from AI feedback)](rlaif.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Nash and game-theoretic preference optimization](nash-and-game-theoretic-po.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Length and format bias](length-and-format-bias.md)
- [The alignment tax](alignment-tax.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md)
- [LLM-as-judge](llm-as-judge.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [Judging bias and contamination](judging-bias-and-contamination.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)

## References
- [source:openai:iterated-distillation-and-amplification-] [Iterated Distillation and Amplification (IDA) - OpenAI Blog](https://openai.com/index/learning-complex-goals-with-iterated-amplification/)
- [source:arxiv:2212.08073] [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [source:arxiv:1810.08575] [Supervising strong learners by amplifying weak experts (Christiano et al.)](https://arxiv.org/abs/1810.08575)
- [source:arxiv:2505.03335] [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)
- [source:arxiv:2308.08998] [ReST: Reinforced Self-Training for Language Modeling](https://arxiv.org/abs/2308.08998)
- [source:arxiv:2203.14465] [STaR: Self-Taught Reasoner: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- [source:arxiv:2402.04683] [Q*: Improving Generalization and Reasoning in LLMs via Q-learning and Self-play (Reference Context)](https://arxiv.org/abs/2402.04683)
- [source:arxiv:2212.10560] [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
- [source:openai:iterated-amplification-supervising-stron] [Iterated Amplification: Supervising strong learners by amplifying weak experts (OpenAI blog/paper context)](https://openai.com/index/learning-complex-goals-with-iterated-amplification/)
- [source:semanticscholar:machine-learning-projects-for-iterated-d] [Machine Learning Projects for Iterated Distillation and Amplification](https://www.semanticscholar.org/paper/Machine-Learning-Projects-for-Iterated-Distillation-Evans-Saunders/bdf2e444eb59702f1a119396498c8c80a88f6e6a)
- [source:aclanthology:re-rest-reflection-reinforced-self-train] [Re-ReST: Reflection-Reinforced Self-Training for Language Agents](https://aclanthology.org/2024.emnlp-main.861/)
