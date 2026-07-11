---
title: Verifiable rewards (RLVR)
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2509.21882
- promptfoo:reinforcement-learning-with-verifiable-r
- arxiv:2605.29275
- arxiv:2503.23829
- github:awesome-rlvr-reinforcement-learning-with
- cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin
- lucek:reinforcement-learning-with-verifiable-r
open_questions:
- How much of the RLVR tax (attempt inflation, miscalibration) is **intrinsic to binary
  reward optimization** versus an artifact of current training recipes (e.g., lack
  of refusal rewards, KL scheduling)?
- Can **generative verifiers** (soft rewards) achieve the same unhackability guarantees
  as binary verifiers, or do they inevitably re-introduce reward-model-like vulnerabilities
  at scale?
- 'What is the **true capability ceiling of search compression**: does there exist
  a pass@k saturation point beyond which no amount of probability-mass concentration
  yields further pass@1 gains, and how does it vary by task/domain?'
- How should **contamination screens** (partial-prompt probes, held-out fresh benchmarks)
  be standardized and mandated in RLVR publications to make reported numbers comparable?
---

Reinforcement Learning with Verifiable Rewards (RLVR) replaces subjective preference models with deterministic, programmatic verifiers—unit tests, math normalizers, or executable checkers—to supply binary or near-binary reward signals for LLM policy optimization. While RLVR has driven striking gains on math and coding benchmarks, recent work shows that reported improvements are often inflated by evaluation budget mismatches, a constellation of side effects termed the "RLVR tax," and data contamination, and that the method may primarily compress existing search capability rather than expand the reasoning frontier.

## Definition and core loop

RLVR frames LLM generation as an MDP where the reward function $r(s, a)$ is computed by an external, deterministic verifier rather than a learned reward model [source:github:awesome-rlvr-reinforcement-learning-with]. The canonical loop iterates: (1) sample $K$ completions $\{a\}_{1..K} \sim \pi_\theta(\cdot|s)$; (2) pass each through a verification function $r(s, a) \in \{\gamma, 0\}$ (or a soft variant); (3) update $\pi_\theta$ with a policy-gradient algorithm (PPO, GRPO, REINFORCE++) to maximize expected reward; (4) optionally refine the verifier to cover new edge cases [source:github:awesome-rlvr-reinforcement-learning-with][source:lucek:reinforcement-learning-with-verifiable-r]. The verifier must be "tamper-proof" and objectively groundable—e.g., a compiler, a unit-test suite, a formal proof checker, or a string-normalization routine for math [source:github:awesome-rlvr-reinforcement-learning-with][source:lucek:reinforcement-learning-with-verifiable-r].

## Rule-based verifiers: code and math

In code generation, verifiers execute candidate programs against hidden unit tests (e.g., BIRD Text2SQL benchmark) or check syntax/schema compliance [source:promptfoo:reinforcement-learning-with-verifiable-r][source:lucek:reinforcement-learning-with-verifiable-r]. In mathematics, verifiers normalize model output and reference answers (e.g., stripping LaTeX, evaluating symbolic equivalence) and return exact-match or numeric-equivalence signals [source:arxiv:2509.21882][source:promptfoo:reinforcement-learning-with-verifiable-r]. These binary rewards are attractive because they are **unhackable in principle**—the model cannot "persuade" the verifier—and they enable **auditable, reproducible** training [source:github:awesome-rlvr-reinforcement-learning-with][source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin].

However, rule-based verification is **domain-limited**. Only ~60.3% of math problems and ~45.4% of complex multi-domain queries admit single-term answers amenable to exact matching [source:arxiv:2503.23829]. For everything else—creative writing, long-form QA, scientific reasoning—binary verifiers do not exist, forcing a choice between heuristic proxies (format rewards, partial matching) and learned/LLM-based rewards that re-introduce subjectivity and hackability [source:lucek:reinforcement-learning-with-verifiable-r][source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin].

## The RLVR tax: attempt inflation, calibration, and safety

[source:arxiv:2509.21882] identifies a systematic "RLVR tax"—unintended side effects that accompany pass@1 gains:

| Tax component | Mechanism | Evidence |
|---------------|-----------|----------|
| **Attempt inflation** | Models convert abstentions ("I don't know") into confident but incorrect answers. | Qwen2.5-14B-Instruct → R1-Distill-Qwen-14B: "Not attempted" dropped 1,136 → 102, yet **shared accuracy** (on items both attempted) fell 12.5% → 10.5% [source:arxiv:2509.21882]. |
| **Miscalibration** | Confidence estimates diverge from true accuracy. | Expected Calibration Error (ECE) worsened 0.598 → 0.692 in the same comparison [source:arxiv:2509.21882]. |
| **Instruction-fidelity erosion** | Models over-optimize for the verifier at the expense of user constraints (format, style, safety). | Not directly quantified in sources; noted as a risk from longer reasoning traces increasing privacy/safety exposure [source:arxiv:2509.21882]. |
| **Safety/privacy risk** | Longer CoT traces may leak PII or reveal chain-of-thought that aids adversarial attacks. | Highlighted as a consequence of "attempt inflation" and longer generations [source:arxiv:2509.21882]. |

The paper proposes a **tax-aware reporting standard**: matched sampling budgets (saturation curves $acc(k)$), refusal rates, shared accuracy, ECE, judge-robustness stress tests, and explicit contamination screens [source:arxiv:2509.21882].

## Search compression vs. capability expansion

[source:promptfoo:reinforcement-learning-with-verifiable-r] argues that RLVR predominantly achieves **search compression**: the policy learns to concentrate probability mass on reasoning paths it could already sample at test time, rather than acquiring new reasoning capabilities. Evidence: in a representative run, pass@1 rose +25 pp while pass@8 rose only +2 pp, implying a compression ratio of $25/35 \approx 71\%$ [source:promptfoo:reinforcement-learning-with-verifiable-r]. [source:arxiv:2509.21882] corroborates that **budget mismatch**—comparing RLVR models evaluated at high $k$ (pass@k) against baselines at pass@1—conflates search with capability. Their standardized SoberScore (avg@32, matched prompts/verifiers) slashed reported gains: Open-RS3-1.5B 46.70% → 30.94% ($\Delta=+15.76$), STILL-3-1.5B 39.33% → 31.46% ($\Delta=+7.87$), while DAPO-Qwen-32B actually *lost* 1.56 pp [source:arxiv:2509.21882].

**Disagreement:** [source:arxiv:2503.23829] reports **consistent OOD generalization gains** with a generative verifier (RM-7B): NaturalReasoning 39.8% vs 29.4%, WebInstruct 44.0% vs 33.9%, and stable scaling to 100k samples where rule-based rewards degraded. This suggests that *soft, model-based verifiers* may expand capability more than binary rule-based ones, or that the OOD benchmarks differ in nature from the in-distribution math/code tasks where compression dominates. [source:lucek:reinforcement-learning-with-verifiable-r] also cites "emergent reasoning" behaviors in AppWorld (tool-use persistence, API-doc reading) driven only by a binary task-completion reward, which [source:promptfoo:reinforcement-learning-with-verifiable-r] would classify as search compression. The discrepancy may hinge on **verifier richness**: binary verifiers on narrow tasks compress search; generative verifiers on diverse tasks may force broader generalization.

## Spurious rewards and reward hacking

[source:promptfoo:reinforcement-learning-with-verifiable-r] documents a striking **spurious-reward effect**: Qwen2.5-Math-7B improved **21.4% on MATH-500 with purely random rewards**, versus 29.1% with ground-truth rewards. This implies that a large fraction of the observed gain stems from the RL update dynamics (entropy reduction, length regularization, etc.) rather than the verifier signal—especially in certain model families (noted for Qwen) and possibly linked to training-data contamination [source:promptfoo:reinforcement-learning-with-verifiable-r]. [source:arxiv:2509.21882] does not test random rewards but finds heavy contamination on legacy benchmarks (Qwen3-14B-Base: 58.2% ACC@80 on MATH-500 prefix probe, 0.0% on fresh AIME-2025), which could amplify spurious gains [source:arxiv:2509.21882].

**Failure modes** identified by [source:promptfoo:reinforcement-learning-with-verifiable-r]:
1. **Partial verifiers** (e.g., syntax-only SQL check) → models exploit gaps to earn rewards for incorrect answers.
2. **Spurious rewards** → gains are RL artifacts, not verifier-driven.
3. **Entropy instability** → rapid entropy collapse in GRPO causes mode collapse, boosting in-distribution accuracy while destroying OOD generalization.

## Extending beyond binary verification: soft rewards and rubrics

To address domains without exact-match verifiers, three directions have emerged:

### Generative verifiers (soft rewards)
[source:arxiv:2503.23829] replaces binary rule-based rewards with a **generative LLM verifier** $\pi_\phi$ that outputs a binary judgment $c \in \{0,1\}$ given $(x, a, y)$. The **soft reward** uses the verifier's token probability:

$$
r_\phi(x,a,y_i) = 
\begin{cases}
\pi_\phi(1 \mid x, a, y_i^T) & \text{if } c_i=1 \\
1 - \pi_\phi(0 \mid x, a, y_i^T) & \text{if } c_i=0 \\
0 & \text{otherwise}
\end{cases}
$$

A compact RM-7B is distilled from a large teacher (Qwen2.5-72B-Instruct) on 160k RL-exploration samples, then used for online REINFORCE/RLOO/REINFORCE++ with z-score normalization and KL penalty [source:arxiv:2503.23829]. This achieves high verifier consistency (Cohen's $\kappa > 0.86$ math, $>0.88$ college) and up to 8.0% accuracy gains over SOTA on free-form reasoning, with better OOD scaling than rule-based rewards [source:arxiv:2503.23829]. Limitations: no CoT in verifier, no format constraints, no process supervision for intermediate steps [source:arxiv:2503.23829].

### Prompt-level rubrics + hard checkers
[source:arxiv:2605.29275] decomposes open-ended rewards into three **normalized $[0,1]$ signals** computed online:
- **Rubric score** $s_r$: LLM judges weighted atomic criteria (yes=1, part=0.5, no=0) → weighted average.
- **Global score** $s_g$: LLM holistic 0–10 score, clipped/10.
- **Code score** $s_c$: executable Python checkers for surface constraints (length, required strings, format) → pass rate.

Hybrid reward aggregates them; during RL the global-score weight $\alpha$ decays linearly 1→0 over 800 steps to shift from helpfulness to fine-grained compliance [source:arxiv:2605.29275]. Results: RewardBench v2 overall 85.1 (hybrid) vs 80.0 (global only) vs 77.0 (rubric only); online RL gains +4.7 to +8.7 avg across backbones; code checkers boost VERINSTRUCT Top-1 exact pass 48.0% → 69.5% and cut constraint-discordant inversions 14.8% → 3.1% [source:arxiv:2605.29275]. Limitations: evaluator dependence, surface-only verification, sandboxing risks, small human eval scale [source:arxiv:2605.29275].

### Rubric-based RL (LLM-as-judge)
[source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin] advocates **structured rubrics** (general or prompt-specific, synthetic or expert-written) decomposed into weighted criteria, evaluated by an LLM judge with CoT, aggregated into a multi-dimensional reward for PPO/GRPO. Qualitative gains: improved LLM-judge reliability vs single prompts; prompt-specific rubrics raise human–judge agreement on MultiChallenge; scalable human-preference approximation [source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin]. **Critical limitation**: the judge introduces **position bias, verbosity bias, self-enhancement bias, capability bias, distribution bias, and prompt sensitivity**, requiring careful calibration and high-quality human labels to align [source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin].

## Training algorithms: GRPO, REINFORCE++, GSPO

| Algorithm | Key mechanism | Used in RLVR context |
|-----------|---------------|----------------------|
| **GRPO** | Value-free; advantage $A_i = R(s_i,a_i) - \text{baseline}(R_{\text{group}})$ where baseline = group mean/median [source:promptfoo:reinforcement-learning-with-verifiable-r]. | Default for binary-verifier RLVR (DeepSeek-R1, Open-R1) [source:promptfoo:reinforcement-learning-with-verifiable-r][source:github:awesome-rlvr-reinforcement-learning-with]. |
| **REINFORCE++ / RLOO** | Policy gradient $\nabla_\theta J = \mathbb{E}[r \nabla_\theta \log \pi_\theta]$ with leave-one-out baseline or z-score normalization [source:arxiv:2503.23829]. | Used with generative verifier RM-7B for free-form domains [source:arxiv:2503.23829]. |
| **GSPO** | Group Sequence Policy Optimization; used with hybrid rubric/global/code rewards [source:arxiv:2605.29275]. | Online RL for open-ended instruction following [source:arxiv:2605.29275]. |
| **PPO** | Clipped surrogate objective with value function; classic RLHF workhorse [source:lucek:reinforcement-learning-with-verifiable-r][source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin]. | Still used for rubric-based and format/accuracy reward tiers [source:lucek:reinforcement-learning-with-verifiable-r][source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin]. |

**Data efficiency**: DEPO (offline curation: keep prompts where base model pass@k ∈ [30%, 70%]; rollout pruning: top 50% rewards; difficulty scheduling) cuts compute 60–70% (1.85×/1.66× speedup on AIME24/25) [source:promptfoo:reinforcement-learning-with-verifiable-r].

## Data contamination and evaluation hygiene

[source:arxiv:2509.21882] demonstrates that **legacy benchmarks are heavily contaminated**: Qwen3-14B-Base achieves 58.2% ACC@80 (greedy suffix completion from 80% prefix) on MATH-500 but 0.0% on fresh AIME-2025. Their **partial-prompt probe** (reveal $x\%$ prefix, greedily decode suffix; measure ACC@x, EM@x) is a high-precision but non-exhaustive contamination screen [source:arxiv:2509.21882]. The **SoberScore** protocol (avg@32, matched prompts/verifiers/decoding budgets) is proposed as a minimum standard to separate genuine reasoning gains from search/compression artifacts [source:arxiv:2509.21882].

## Current status and trajectory

RLVR is **rising as the default paradigm for verifiable domains** (math, code, structured extraction) and **actively expanding into semi-verifiable and open-ended domains** via generative verifiers and rubric-based hybrids. The ICLR/ICML 2026 volume (135 RLVR papers) and open-source ecosystems (OpenRLHF, verl, open-r1) confirm massive momentum [source:github:awesome-rlvr-reinforcement-learning-with]. However, the field is **in a measurement-crisis phase**: the RLVR tax (attempt inflation, miscalibration), spurious-reward artifacts, and contamination mean that *many reported SOTA numbers are not comparable or reproducible* [source:arxiv:2509.21882][source:promptfoo:reinforcement-learning-with-verifiable-r]. Adoption of tax-aware reporting (SoberScore, saturation curves, shared accuracy, ECE) and contamination screens is **not yet widespread**; until it is, the literature will overstate capability gains. The trajectory points toward **hybrid reward systems** (binary + soft + code checkers) as the practical default for production, with pure binary RLVR remaining a research substrate for reasoning studies.

## Key takeaways

- RLVR substitutes learned reward models with **deterministic, programmatic verifiers** (unit tests, math normalizers) for binary or near-binary rewards, enabling unhackable, auditable training in math/code [source:github:awesome-rlvr-reinforcement-learning-with][source:lucek:reinforcement-learning-with-verifiable-r].
- **Reported gains are frequently inflated** by three confounds: (1) budget mismatch (high-$k$ eval vs pass@1 baselines), (2) the **RLVR tax** (attempt inflation, miscalibration, instruction-fidelity erosion), and (3) **data contamination** on legacy benchmarks [source:arxiv:2509.21882].
- RLVR may primarily achieve **search compression** (concentrating mass on already-samplable paths) rather than expanding the reasoning frontier; pass@1 gains far exceed pass@k gains [source:promptfoo:reinforcement-learning-with-verifiable-r][source:arxiv:2509.21882].
- **Spurious rewards** are real: random rewards yield ~70% of the ground-truth gain in some settings, implicating RL dynamics and contamination [source:promptfoo:reinforcement-learning-with-verifiable-r].
- **Rule-based verifiers cover <50% of complex queries**; extension requires **generative soft verifiers** (distilled RM-7B, Cohen's $\kappa>0.86$) [source:arxiv:2503.23829], **prompt-level rubrics + executable checkers** (hybrid reward 85.1 RewardBench) [source:arxiv:2605.29275], or **LLM-judge rubrics** (with known biases) [source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin].
- **Standardized evaluation (SoberScore, avg@32, shared accuracy, ECE, contamination probes) is not yet standard practice**; the field is in a measurement-crisis phase [source:arxiv:2509.21882].

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
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
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
- [source:arxiv:2509.21882] [The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2509.21882)
- [source:promptfoo:reinforcement-learning-with-verifiable-r] [Reinforcement Learning with Verifiable Rewards Makes Models Faster, Not Smarter](https://www.promptfoo.dev/blog/rlvr-explained/)
- [source:arxiv:2605.29275] [Prompt-Level Reward Specifications for Open-Ended Post-Training](https://arxiv.org/abs/2605.29275)
- [source:arxiv:2503.23829] [Expanding RL with Verifiable Rewards Across Diverse Domains](https://arxiv.org/abs/2503.23829)
- [source:github:awesome-rlvr-reinforcement-learning-with] [Awesome RLVR — Reinforcement Learning with Verifiable Rewards](https://github.com/opendilab/awesome-RLVR)
- [source:cameronrwolfe:rubric-based-rewards-for-rl-deep-learnin] [Rubric-Based Rewards for RL - Deep (Learning) Focus](https://cameronrwolfe.substack.com/p/rubric-rl)
- [source:lucek:reinforcement-learning-with-verifiable-r] [Reinforcement Learning with Verifiable Rewards for LLMs](https://lucek.ai/blogs/rlvr-with-llms.html)
