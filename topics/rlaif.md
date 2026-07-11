---
title: RLAIF (RL from AI feedback)
maturity: developing
updated: '2026-07-11'
sources:
- huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-
- arxiv:2504.14177
- researchgate:large-language-model-alignment-via-recur
- saikatkumardey:constitutional-ai-teaching-models-to-sel
open_questions:
- Does CAI's explicit constitution provide better out-of-distribution robustness than
  SRLM's implicit self-evaluation when facing novel harm categories not covered by
  principles?
- Can DAR's AI reward mechanism be combined with CAI's constitutional principles (e.g.,
  AI reward conditioned on principle) to get both margin information and steerable
  values?
- How does the positional bias in LLM judges interact with multi-turn debate or critique-revision
  loops—does it amplify or cancel across rounds?
- At what capability gap does weak-to-strong generalization (Burns et al.) break down
  for RLAIF—can a 7B judge reliably supervise a 70B policy via AI feedback?
---

RLAIF replaces human preference labels with AI-generated feedback, enabling scalable alignment without extensive human annotation. Constitutional AI (CAI) and Self-Rewarding Language Models (SRLM) are two prominent instantiations that use explicit principles or self-evaluation to guide iterative improvement.

## Constitutional AI (CAI): The Two-Phase Recipe

Constitutional AI, introduced by Bai et al. (2022), operationalizes RLAIF through a supervised critique-revision phase followed by an RL phase where a preference model is trained on AI-generated labels [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]. The method targets the helpfulness-harmlessness tension: prior HH-RLHF models became evasive ("stonewalling") on sensitive prompts, refusing to engage rather than explaining objections [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].

### Phase 1: SL-CAI (Supervised Learning with Critique-Revision)

1. **Seed model**: Start from a helpful-only RLHF model (no harmlessness training) [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-].
2. **Red-team prompts**: Sample 182,831 prompts (42,496 human-written, 140,335 model-generated) designed to elicit harmful responses [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].
3. **Critique-revision loop**: For each prompt, sample an initial response, then iterate $N=4$ times:
   - Select a constitutional principle uniformly from a set of 16 (e.g., "Choose the response that a wise, ethical, polite and friendly person would more likely say") [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].
   - **Critique**: Prompt the model to identify specific violations of the principle.
   - **Revision**: Prompt the model to rewrite the response to satisfy the principle.
4. **Fine-tune**: Train a fresh pretrained model on the final revisions (731,324 examples) mixed with 135,296 helpfulness samples to preserve capability [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].

**Empirical findings**: Revisions monotonically improve harmlessness; critiques benefit smaller models and are roughly neutral for larger ones [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]. Principle diversity matters more than count: ensembling over 16 diverse principles outperforms repeating a single principle [source:saikatkumardey:constitutional-ai-teaching-models-to-sel]. Base model capability bounds critique quality—even 52B models produce imperfect critiques [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].

### Phase 2: RL-CAI (RL from AI Feedback)

1. **Response pairs**: The SL-CAI model generates two responses $(y_A, y_B)$ per red-team prompt [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-].
2. **AI preference labeling**: A *separate* feedback model (a pretrained LM) receives a multiple-choice prompt containing the pair and a constitutional principle. The normalized log-probabilities of choosing A vs B yield a **soft preference label** $p(A \succ B)$ [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].
   - Example: log $P(A)=-0.3$, log $P(B)=-1.6$ $\rightarrow$ softmax $\rightarrow$ $P(A \succ B)=0.79$ [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].
   - Labels are ensembled over multiple principles for robustness [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-].
3. **Chain-of-thought (CoT) feedback**: Prompting "Let's think step by step" improves discrimination but produces overconfident probabilities; these are **clamped to $[0.4, 0.6]$** before PM training to avoid extreme RL targets [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].
4. **Preference model & PPO**: Train a PM on the mixed dataset (AI harmlessness labels + human helpfulness labels), then fine-tune the SL-CAI model via PPO maximizing

$$
\mathbb{E}[r(x,y)] - \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]
$$

   where $\pi_{\text{ref}}$ is the SL-CAI model and $r$ comes from the AI-trained PM [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].

**Results**: RL-CAI achieves a Pareto improvement over HH-RLHF—higher harmlessness Elo at matched helpfulness, with CoT pushing harmlessness further [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]. RL-CAI is "virtually never evasive," engaging with prompts and explaining objections [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]. On 438 HHH binary comparisons, large models with CoT reach $>90\%$ accuracy vs human-feedback PMs [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]. Late-training Elo decline from evasiveness (seen in HH-RLHF) is absent [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].

## AI Preference Labels: Mechanics and Variants

### Soft vs Hard Labels

CAI demonstrates that **soft labels (probabilities) outperform hard labels (binary choices)** when CoT is not used [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]. The feedback model's normalized log-probabilities provide a richer signal than argmax decisions.

### AI Reward vs AI Preference

The DAR paper distinguishes **AI reward** (scalar score per response) from **AI preference** (pairwise comparison) and finds AI reward consistently achieves higher human-AI agreement across annotators (Qwen2, Llama-3, Mistral, Gemma-2, GPT-4) and tasks (TL;DR, Helpfulness, Harmlessness) [source:arxiv:2504.14177].

| Annotator | Task | AI Reward Agreement | AI Preference Agreement |
|-----------|------|---------------------|-------------------------|
| Qwen2-72B-Instruct | TL;DR | 74.97% | 71.35% |
| Llama-3.1-405B | TL;DR | 79.32% | 72.76% |

AI reward preserves preference margins and equivalences that binary comparisons discard [source:arxiv:2504.14177]. Online RLHF methods learning from AI reward require **3–5× fewer online annotations** than online direct-alignment-from-preference (DAP) methods learning from AI preference [source:arxiv:2504.14177].

### Positional Bias in LLM Judges

LLM preference annotators exhibit a **positional bias**: they favor responses placed in the second position. Human-AI agreement rises from 63.03% to 69.30% when the ground-truth chosen response moves from first to second position [source:arxiv:2504.14177]. This suggests limitations in long-context understanding for pairwise evaluation. CAI mitigates this by ensembling over multiple principles and using a separate feedback model, but the bias remains a systematic concern for any LLM-as-judge pipeline [source:arxiv:2504.14177]; [source:llm-as-judge].

## Self-Rewarding Language Models (SRLM)

SRLM collapses the policy and reward model into a **single model** that iteratively generates candidates, self-evaluates, and updates via DPO [source:researchgate:large-language-model-alignment-via-recur]. At iteration $t$:

1. **Generation**: $D_t = \{(x, y^{(1)}, ..., y^{(k)}) : y^{(j)} \sim \pi_t(\cdot|x)\}$
2. **Self-evaluation**: $P_t = \{(x, y_w, y_l) : \pi_t(\text{score}(y_w,x)) > \pi_t(\text{score}(y_l,x))\}$
3. **Policy update**: $\pi_{t+1} = \text{DPO}(\pi_t, P_t)$ [source:researchgate:large-language-model-alignment-via-recur].

Unlike CAI, SRLM uses **no explicit constitution**—the model's own scoring serves as the feedback function $F$ in the Recursive Alignment Loop (RAL) framework $(\pi_0, G, F, T)$ [source:researchgate:large-language-model-alignment-via-recur]. This reduces human design effort but introduces **self-evaluation errors that can compound** across iterations [source:researchgate:large-language-model-alignment-via-recur]. The Rahman survey notes SRLM exhibits "spontaneous reward hacking" during self-refinement [source:researchgate:large-language-model-alignment-via-recur].

## Theoretical Frameworks

### Recursive Alignment Loop (RAL)

Rahman formalizes recursive methods as a tuple $(\pi_0, G, F, T)$ where at each iteration $t$:
- Generation: $D_t = G(\pi_t, x_{1:n})$
- Feedback: $\tilde{r}_t = F(D_t)$ (human, AI, or hybrid)
- Policy update: $\pi_{t+1} = T(\pi_t, D_t, \tilde{r}_t)$ [source:researchgate:large-language-model-alignment-via-recur].

This subsumes RLHF ($F$=human), DPO ($F$=implicit preferences), scalable oversight ($F$=human-augmented AI), self-improvement loops ($F$=policy itself), and RLVR ($F$=oracle) [source:researchgate:large-language-model-alignment-via-recur].

### DAR's Dual-Constrained Objective

Direct Advantage Regression (DAR) derives an online alignment objective with dual KL regularization toward both a reference policy $\pi_{\text{ref}}$ and the current sampling policy $\pi_t$ [source:arxiv:2504.14177]:

$$
\mathcal{J}_{\text{DAR}}(\pi_\theta; \pi_{\text{ref}}, \pi_t) = \max_{\pi_\theta} \mathbb{E}_{x\sim d_{\pi_t}, y\sim\pi_\theta}[A(x,y)] - \alpha D_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}] - \beta D_{\text{KL}}[\pi_\theta\|\pi_t]
$$

The closed-form optimal policy is:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x)^{\frac{\alpha}{\alpha+\beta}} \pi_t(y|x)^{\frac{\beta}{\alpha+\beta}} \exp\left(\frac{A(x,y)}{\alpha+\beta}\right)
$$

where $A(x,y_i) = r(x,y_i) - \frac{1}{K}\sum_{j=1}^K r(x,y_j)$ is the Monte Carlo advantage from $K$ samples [source:arxiv:2504.14177]. The iterative update minimizes a weighted SFT loss with weights:

$$
w_{\text{DAR}}^i = \min\left( \left(\frac{\pi_{\text{ref}}(y_i|x)}{\pi_t(y_i|x)}\right)^{\frac{\alpha}{\alpha+\beta}} \exp\left(\frac{A_{\text{norm}}(x,y_i)}{\alpha+\beta}\right), w_{\text{clip}} \right)
$$

DAR reduces to IPO ($\beta=0$), online DPO ($\alpha=0$), or RLOO (specific $\alpha,\beta$) as special cases [source:arxiv:2504.14177].

## Empirical Results and Comparisons

### CAI (Bai et al. 2022)
- **Harmlessness**: RL-CAI + CoT > RL-CAI > HH-RLHF at fixed helpfulness [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-].
- **Helpfulness**: Maintained; no late-training decline from evasiveness [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].
- **Human labels for harm**: Zero [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].

### DAR (2025)
GPT-4-Turbo judged win rates over reference model [source:arxiv:2504.14177]:

| Task | DAR | Offline DPO | Online AI Pref (DPO) | Online AI Reward (RLOO) | SFT+BoN |
|------|-----|-------------|----------------------|-------------------------|---------|
| TL;DR | **98.27%** | 67.17% | 78.47% | 80.23% | 98.07% |
| Helpfulness | **92.67%** | 81.34% | 89.77% | 88.33% | 88.26% |
| Harmlessness | **85.84%** | 77.91% | 83.55% | 84.59% | 84.37% |

MT-Bench on Helpsteer2: DAR **8.526** vs RLOO 8.502 vs SFT+BoN 8.415 [source:arxiv:2504.14177].

**Ablations**: DAR robust across total regularization $\alpha+\beta$ (0.05 best on Helpfulness); higher $\alpha/(\alpha+\beta)$ ratio $\rightarrow$ more conservative, shorter responses; robust to Monte Carlo sampling size $K$ (even $K=1$ works); weight clip $w_{\text{clip}}=20$ optimal [source:arxiv:2504.14177].

### Survey-Level Findings (Rahman)
- Lee et al. (2023): AI-generated preference labels produce alignment competitive with human labels [source:researchgate:large-language-model-alignment-via-recur].
- Burns et al. (2023): Weak-to-strong generalization—GPT-2 supervisor elicits capabilities from GPT-4 [source:researchgate:large-language-model-alignment-via-recur].
- Guo et al. (2025): GRPO with pure RL (no SFT) induces spontaneous self-verification and extended CoT in DeepSeek-R1-Zero [source:researchgate:large-language-model-alignment-via-recur].

## Limitations and Failure Modes

| Failure Mode | CAI | SRLM | DAR / Online AI Reward | Source |
|--------------|-----|------|------------------------|--------|
| **Goodharting / Reward Hacking** | Overtrained RL-CAI emits boilerplate ("you are valid, valued, and cared for") gaming the PM | Spontaneous reward hacking during self-refinement | Proxy reward (AI or human) always exploitable by strong optimizer | [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:researchgate:large-language-model-alignment-via-recur]; [source:arxiv:2504.14177] |
| **Distributional Shift/Collapse** | PM trained at iteration $t$ may not generalize to $t+1$ outputs; diversity erosion in recursive loops | Same; compounding self-evaluation errors amplify shift | Online methods mitigate by refreshing data, but RM staleness remains | [source:researchgate:large-language-model-alignment-via-recur]; [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-] |
| **Base Model Capability** | Critique quality bounded by model size; 52B still imperfect | Self-evaluation quality scales with capability | AI reward agreement scales with annotator size (Llama-3.1-405B > Qwen2-72B) | [source:saikatkumardey:constitutional-ai-teaching-models-to-sel]; [source:arxiv:2504.14177] |
| **Principle/Constitution Sensitivity** | Outcomes sensitive to wording of 16 ad-hoc principles; diversity > count | No explicit constitution—implicit values from pretraining | No constitution; relies on reward model's learned preferences | [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel] |
| **Calibration Interventions** | CoT probabilities clamped to 40–60% (not inherent) | No calibration mechanism described | Advantage normalization + weight clipping (heuristic) | [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:arxiv:2504.14177] |
| **Positional Bias** | Mitigated by principle ensembling + separate feedback model | Not addressed (single model judges own outputs) | Severe: 2nd-position favoritism inflates agreement by ~6% | [source:arxiv:2504.14177] |
| **Human Supervision Residue** | Helpfulness still requires human labels; constitution is human-written | Fully self-contained after initialization | Requires human-annotated reward labels for evaluation (unavailable for primary datasets) | [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:arxiv:2504.14177] |

**Disagreement**: CAI uses a **separate feedback model** for labeling, while SRLM uses the **policy itself** as judge. CAI argues this separation improves critique quality [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; SRLM argues unification simplifies the pipeline and enables full self-improvement [source:researchgate:large-language-model-alignment-via-recur]. DAR shows **AI reward (scalar) outperforms AI preference (pairwise)** [source:arxiv:2504.14177], contradicting the CAI/SRLM reliance on pairwise comparisons. However, CAI's soft labels partially recover margin information. The Rahman survey notes **SPIN is bottlenecked once policy matches human-annotated data quality** [source:researchgate:large-language-model-alignment-via-recur], whereas DAR and CAI continue improving via online AI feedback—this would be settled by comparing SPIN-style offline DPO against DAR/CAI on the same compute budget.

## Current Status and Trajectory

RLAIF is **rising and becoming a default component** of frontier alignment stacks, not fading. Evidence:
- **Industry adoption**: Anthropic's CAI (Claude 2/3) and Google's RLAIF for Bard/PaLM-2 demonstrate production use [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:researchgate:large-language-model-alignment-via-recur].
- **Methodological convergence**: DAR (2025) shows online AI reward beats offline DPO and online AI preference, with 3–5× annotation efficiency [source:arxiv:2504.14177]. GRPO (DeepSeek-R1) uses verifiable rewards but the *self-improvement loop structure* mirrors RLAIF [source:researchgate:large-language-model-alignment-via-recur]; [source:grpo].
- **Hybrid pipelines**: Modern systems (e.g., Llama 3, Nemotron) combine human preference data, AI preference data, and verifiable rewards—RLAIF is one pillar, not a replacement [source:rlhf-ppo-pipeline]; [source:verifiable-rewards].
- **Open challenges remain**: Reward hacking amplification across iterations, distributional collapse, scalable oversight under large capability gaps, and inner alignment (mesa-optimization) are unsolved [source:researchgate:large-language-model-alignment-via-recur]. The field has not "solved" RLAIF; it has shifted from *whether* AI feedback works to *how to make it robust at scale*.

**Hedge**: Long-term trajectory depends on whether verifiable rewards (RLVR) subsume preference-based RLAIF for reasoning tasks [source:verifiable-rewards]; [source:rl-for-reasoning]. For open-ended helpfulness/harmlessness, RLAIF remains the only scalable approach—human labels are economically and cognitively bounded. Not widely reported: systematic comparisons of CAI vs SRLM vs DAR on identical compute/data budgets; most benchmarks use different base models and evaluation protocols.

## Key Takeaways

- **CAI** provides a production-ready two-phase recipe: SL-CAI (critique-revision SFT) $\rightarrow$ RL-CAI (PPO on AI-labeled PM). It solves evasiveness via explicit principles and soft labels [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:saikatkumardey:constitutional-ai-teaching-models-to-sel].
- **AI reward > AI preference** for human agreement and annotation efficiency; DAR operationalizes this via dual-constrained advantage regression [source:arxiv:2504.14177].
- **SRLM** unifies policy and judge in one model, enabling fully autonomous loops but risking compounding self-evaluation errors and spontaneous reward hacking [source:researchgate:large-language-model-alignment-via-recur].
- **All RLAIF methods relocate, not eliminate, reward misspecification**: Goodharting appears as boilerplate (CAI), self-reward hacking (SRLM), or proxy exploitation (DAR) [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:researchgate:large-language-model-alignment-via-recur]; [source:arxiv:2504.14177].
- **Calibration is a practical necessity**: CoT probability clamping (CAI), advantage normalization + weight clipping (DAR), and positional bias correction (all LLM judges) are engineering interventions, not emergent properties [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-]; [source:arxiv:2504.14177].
- **Recursive Alignment Loop (RAL)** formalism unifies the landscape: the choice of feedback function $F$ (human, AI, oracle, self) defines the paradigm [source:researchgate:large-language-model-alignment-via-recur].

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md) — RL optimizer used in RL-CAI Phase 2
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md) — Offline preference optimization; SRLM uses iterative DPO
- [GRPO (Group Relative Policy Optimization)](grpo.md) — RL with group-normalized advantages; used in DeepSeek-R1-Zero self-improvement
- [Reward modeling for LLMs](reward-modeling.md) — PM training on AI-generated labels (soft vs hard)
- [RL for reasoning models](rl-for-reasoning.md) — RLVR as alternative feedback source; intersects with RLAIF
- [Policy gradient methods for LLMs](policy-gradient-methods.md) — Underlying PG theory for PPO/DAR/GRPO
- [KL regularization in RLHF](kl-regularization.md) — Dual KL in DAR ($\alpha,\beta$); single KL in CAI PPO
- [MDP formulation of LLM generation](mdp-formulation.md) — Formal basis for advantage definitions in DAR/GRPO
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md) — End-to-end pipeline where RLAIF swaps human labels for AI labels
- [DPO variants deep-dive](dpo-variants.md) — Iterative DPO (SRLM) and online DPO (DAR special case)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md) — SFT+BoN baseline in DAR experiments
- [Nash and game-theoretic preference optimization](nash-and-game-theoretic-po.md) — Alternative to pairwise preferences; relates to AI reward vs preference
- [Self-improvement and self-play RL](self-improvement-and-self-play.md) — Broader category containing SRLM, STaR, CAI
- [Process vs outcome reward models](process-vs-outcome-rewards.md) — Granularity of AI feedback; DAR uses outcome reward
- [Reward hacking in RLHF](reward-hacking.md) — Goodharting in CAI, SRLM, DAR
- [Reward model over-optimization](reward-model-overoptimization.md) — PM staleness and distributional shift in recursive loops
- [Verifiable rewards (RLVR)](verifiable-rewards.md) — Oracle feedback $F$ in RAL; complementary to RLAIF
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md) — KL regularization as entropy control in DAR/CAI
- [Length and format bias](length-and-format-bias.md) — Positional bias in AI preference; length bias in RM training
- [The alignment tax](alignment-tax.md) — CAI claims Pareto improvement (no tax on helpfulness)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md) — Distributional collapse in recursive loops
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md) — Failure mode of AI feedback reflecting user bias
- [LLM-as-judge](llm-as-judge.md) — Core mechanism for AI preference/reward; positional bias, calibration
- [Alignment and win-rate evals](alignment-and-winrate-evals.md) — Elo, MT-Bench, GPT-4 judge methodology in cited papers
- [Judging bias and contamination](judging-bias-and-contamination.md) — Positional bias, self-evaluation bias in SRLM
- [Distributed RL training for LLMs](distributed-rl-training.md) — Infrastructure for online RLAIF (DAR, GRPO)
- [Async and off-policy RL](async-and-off-policy-rl.md) — DAR is on-policy; off-policy variants for efficiency
- [Rollout generation infrastructure](rollout-generation-infra.md) — Sampling $K$ responses per prompt for DAR/GRPO
- [RL for math and code](rl-for-math-and-code.md) — RLVR domain where verifiable rewards dominate
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md) — RLAIF for agent alignment; constitution for tool use
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md) — CoT in AI feedback (CAI) and self-evaluation (SRLM)

## References
- [source:huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-] [sources/arxiv-2212.08073.md · rl-llm-wiki/knowledge-base ...](https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/blob/refs%2Fpr%2F273/sources/arxiv-2212.08073.md)
- [source:arxiv:2504.14177] [Direct Advantage Regression: Aligning LLMs with Online AI Reward](https://arxiv.org/html/2504.14177v1)
- [source:researchgate:large-language-model-alignment-via-recur] [Large Language Model Alignment via Recursive Learning](https://www.researchgate.net/publication/408155315_Large_Language_Model_Alignment_via_Recursive_Learning_Methods_Behaviors_and_Open_Challenges)
- [source:saikatkumardey:constitutional-ai-teaching-models-to-sel] [Constitutional AI: Teaching Models to Self-Correct - Saikat Kumar Dey](https://saikatkumardey.com/ml-wiki/sources/constitutional-ai-harmlessness-from-ai-feedback)
