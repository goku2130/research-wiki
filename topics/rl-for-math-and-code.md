---
title: RL for math and code
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2511.22570
- neurips:co-evolving-llm-coder-and-unit-tester-vi
- github:awesome-rlvr-reinforcement-learning-with
- labelstud:reinforcement-learning-from-verifiable-r
- arxiv:2604.25419
- arxiv:2605.09920
- aclanthology:pdf-verifier-free-rl-for-llms-via-intrin
- arxiv:2508.21107
open_questions:
- How does meta-verification scale when the verifier itself becomes superhuman—can
  an LLM meta-verifier reliably audit proofs beyond human checking capacity?
- What is the optimal trade-off between external formal verification (high precision,
  low coverage, high compute) and intrinsic rewards (broad coverage, proxy risk) for
  a given compute budget?
- Can automated labeling flywheels (DeepSeekMath-V2 style) be made fully autonomous
  without human fallback, or will ambiguous cases always require expert routing?
- Does VIGOR's gradient-norm reward implicitly encode a notion of "reasoning difficulty"
  that transfers to open-ended generation, or is it fundamentally tied to verifiable
  domains?
---

Reinforcement learning with verifiable rewards (RLVR) has emerged as the dominant paradigm for eliciting reliable reasoning in mathematics and code generation, replacing learned reward models with deterministic verifiers or intrinsic signals. This article synthesizes the design space of verifiers—from formal proof checkers and unit-test execution to meta-verification and gradient-norm intrinsics—and examines how they shape policy optimization in DeepSeekMath-V2, CURE, UTRL, JURY-RL, and VIGOR.

## Verifier Design Paradigms

### The RLVR Loop and Verifiable Reward Taxonomy

The canonical RLVR loop consists of sampling completions from a policy $\pi_\theta$, evaluating each with a deterministic verification function $r(s, a) \in \{0, \gamma\}$, and updating the policy via PPO or GRPO [source:github:awesome-rlvr-reinforcement-learning-with]. Verifiable rewards are categorized by their grounding mechanism [source:labelstud:reinforcement-learning-from-verifiable-r]:

| Category | Verification Mechanism | Typical Reward Schema |
|----------|------------------------|----------------------|
| Mathematical Correctness | Exact match / symbolic equivalence | $1$ (correct), $0.1$ (format only), $0$ (wrong) |
| Code Execution | Unit-test pass/fail, exception check | $+1$ (all pass), $-1$ (any fail), $-0.2$ (no valid code) |
| Instruction/Format | Regex / structural pattern match | $1.0$ (match), $0.0$ (mismatch) |
| Factual / Logical | Rule-based lookup, theorem prover | Binary or multi-level |

The critical design choice is whether the verifier is **external** (Lean, Python interpreter) or **internal** (LLM-based judge, gradient norm). External verifiers provide high precision but limited coverage; internal verifiers scale broadly but risk false positives.

### DeepSeekMath-V2: Self-Verifiable Mathematical Reasoning with Meta-Verification

DeepSeekMath-V2 addresses the fundamental limitation that correct final answers do not guarantee correct reasoning, especially in theorem proving where no numerical answer exists [source:arxiv:2511.22570]. It introduces a three-stage co-training of a **proof generator**, a **verifier**, and a **meta-verifier**.

**Proof Verification Training.** The verifier $c_\varphi$ consumes a problem $X$, a candidate proof $Y$, and rubric $\mathcal{I}_\nu$, outputting an analysis $Z'$ and score $A' \in \{0, 0.5, 1\}$. It is trained via RL on expert-annotated data $\mathcal{D}_\nu = \{(X_i, Y_i, A_i)\}$ with reward:

$$
R_{\text{ver}} = R_{\text{format}}(Z') \cdot R_{\text{score}}(A', A_i), \quad R_{\text{score}} = 1 - |A' - A_i|
$$

where $R_{\text{format}}$ enforces key phrases ("Here is my evaluation...", "Based on my evaluation, the final overall score should be: \boxed{...}") [source:arxiv:2511.22570].

**Meta-Verification.** To curb hallucinated issues that coincidentally yield correct scores, a meta-verifier $c_\eta$ evaluates the verifier's analysis $Z$ against meta-rubrics $\mathcal{I}_{m\nu}$, producing a quality score $\bar{A} \in \{0, 0.5, 1\}$. The enhanced verifier reward becomes:

$$
R_V = R_{\text{format}} \cdot R_{\text{score}} \cdot R_{\text{meta}}, \quad R_{\text{meta}} = \bar{A}
$$

On a validation split, meta-verification lifted average analysis quality from **0.85 to 0.96** without degrading score prediction accuracy [source:arxiv:2511.22570].

**Proof Generation with Self-Verification.** The generator $c_\theta$ produces a proof $Y$ followed by a self-analysis $Z$ (mirroring verifier format). The verifier scores both: $s = A$ for $Y$, and meta-score $ms = \bar{A}$ for $Z$. The combined reward:

$$
R = R_{\text{format}}(Y, Z) \cdot (\alpha \cdot R_Y + \beta \cdot R_Z), \quad R_Y = s,\; R_Z = R_{\text{score}}(s', s) \cdot R_{\text{meta}}(Z)
$$

with $\alpha=0.76, \beta=0.24$, incentivizes faithful self-evaluation [source:arxiv:2511.22570].

**Synergy and Automated Labeling.** As the generator improves, it produces harder proofs that challenge the verifier. An automated pipeline generates $k$ verifications per proof, filters via $m$ meta-verifications (majority vote), and assigns labels: if $\ge 9$ valid analyses agree on the lowest score, that score is used; if no issues found, score $=1$; else route to humans [source:arxiv:2511.22570].

**Results.** On CNML-level problems, DeepSeekMath-V2 achieves mean proof score **0.60** (algebra) vs. GPT-5-Thinking-High **0.54** and Gemini 2.5-Pro **0.17**. In high-compute search: **IMO 2025** 5/6 solved (83.3%), **CMO 2024** 4/6 + partial (73.8%), **Putnam 2024** 118/120 (human best 90) [source:arxiv:2511.22570]. Limitations: single-attempt proofs often exceed 128K tokens; hardest IMO problems remain challenging even with scaled test-time compute [source:arxiv:2511.22570].

### JURY-RL: Label-Free RLVR with Formal Verification and ResZero Fallback

JURY-RL eliminates human-annotated answers by decoupling **proposal** (majority voting) from **disposal** (formal verification in Lean) [source:arxiv:2604.25419].

**Proposal Stage.** For problem $x$, policy $\pi_\theta$ generates $G$ rollouts $\{y_i\}$, extracts answers $a_i = \text{ans}(y_i)$, and proposes consensus $\hat{a}$ via plurality:

$$
\hat{a} = \arg\max_a \sum_{i=1}^G \mathbb{I}[a_i = a]
$$

**Disposal Stage.** A formal verifier checks $\hat{a}$ against the Lean specification of $x$, yielding binary proof-gate $\delta = \text{verify}(x, \hat{a})$.
- If $\delta = 1$: reward $r_i = \mathbb{I}[a_i = \hat{a}]$ (only trajectories matching the proven answer rewarded).
- If $\delta = 0$: **ResZero** fallback assigns a zero-mean, variance-preserving reward to residual answers.

**ResZero Reward.** Let $M = \{i: a_i = \hat{a}\}$ (majority), $R = \{i: a_i \neq \hat{a}\}$ (residual), $\alpha = |M|/G$. For $i \in R$, define leave-one-out residual share $z_i = u^{(-i)}(a_i)$ where

$$
u^{(-i)}(b) = \frac{1}{|R|-1} \sum_{\substack{j \in R \\ j \neq i}} \mathbb{I}[a_j = b]
$$

and $\bar{u} = \frac{1}{|R|} \sum_{j \in R} z_j$. The ResZero reward:

$$
r_i^{\text{ResZero}} = \underbrace{\alpha \cdot \mathbb{I}[i \in R] \cdot (z_i - \bar{u})}_{\text{Amplify residual}} \underbrace{- c \alpha \cdot \mathbb{I}[i \in M]}_{\text{Penalize majority}} + \underbrace{\gamma}_{\text{Re-center}}, \quad \gamma = c \alpha^2
$$

with $c=0.01$. This rewards minority answers that agree with each other while penalizing the unverified majority [source:arxiv:2604.25419].

**Policy Update.** Group-normalized advantages $\hat{A}_i = (r_i - \bar{r}) / (\text{std}(\{r_j\}) + \varepsilon)$ feed into GRPO objective with KL penalty $\beta$ [source:arxiv:2604.25419].

**Results.** JURY-RL matches or exceeds ground-truth GRPO (GT) and outperforms LLM-as-a-judge across Qwen3-1.7B, Llama-3.2-3B, Qwen2.5-7B on math (avg pass@k: **+4.05 pp** over GT, **+9.06 pp** over LLM-judge on Qwen3-1.7B) and generalizes out-of-domain (+1.88 pts on code/instruction/multi-task) [source:arxiv:2604.25419]. ResZero ablation: outperforms Zero Reward (+1.3 pts), MV Reward (+6.1 pts), Random Reward (+5.4 pts) when $\delta=0$ [source:arxiv:2604.25419]. Lean verifier precision **84.5%** vs. LLM-judge **75.9%** (recall 88.0% vs 96.1%) [source:arxiv:2604.25419]. Autoformalization success: 95.4% on MATH500, but only 66.7% on AIME 2025 [source:arxiv:2604.25419]. Computational overhead: Lean adds ~200s/step vs. ~80s for LLM-judge (cold-start Qwen2.5-7B) [source:arxiv:2604.25419].

**Limitations.** Upstream autoformalization/consistency errors cause inconclusive verification ($\delta=0$), reducing supervision density on hard domains. Computational overhead significant but amortized by caching [source:arxiv:2604.25419].

### VIGOR: Verifier-Free Intrinsic Gradient-Norm Reward

VIGOR removes external verifiers entirely, using the policy's own gradient norm as an intrinsic reward signal [source:arxiv:2605.09920][source:aclanthology:pdf-verifier-free-rl-for-llms-via-intrin].

**Reward Computation.** For completion $y$ of length $T$:
1. Average token-level NLL: $\ell_{\text{mean}}(x,y) = \frac{1}{T} \sum_{t=1}^T -\log \pi_\theta(y_t \mid x, y_{<t})$.
2. Gradient norm: $\|\mathbf{g}(x,y)\|_2 = \|\nabla_\theta \ell_{\text{mean}}(x,y)\|_2$ (detached).
3. Length correction: $S_{\text{GN}}(x,y) = -\sqrt{T} \|\mathbf{g}(x,y)\|_2$ (negative sign converts minimization to maximization; $\sqrt{T}$ counters $O(1/\sqrt{T})$ decay).
4. Rank normalization within group of $G$: sort by $S_{\text{GN}}$ (smaller = worse), assign $\text{rank}_i \in \{0,\dots,G-1\}$, then

$$
R_{\text{GN}}(x,y_i) = 2 \frac{\text{rank}_i}{G-1} - 1 \in [-1, +1]
$$

5. Group-relative advantages $\hat{A}_i$ via mean-std normalization of $\{R_{\text{GN}}\}$, fed to GRPO/PPO objective with KL penalty [source:arxiv:2605.09920][source:aclanthology:pdf-verifier-free-rl-for-llms-via-intrin].

**Results.** On Qwen2.5-7B-Base trained on MATH: VIGOR **69.77%** avg math accuracy vs. INTUITOR (entropy-based) **66.46%** (+3.31 pp); cross-domain code transfer **40.42%** vs **38.51%** (+1.91 pp) [source:arxiv:2605.09920][source:aclanthology:pdf-verifier-free-rl-for-llms-via-intrin]. On Qwen2.5-3B-Base: math **59.14%** vs **57.10%** (+2.04 pp); code **27.95%** vs **26.79%** (+1.16 pp) [source:arxiv:2605.09920][source:aclanthology:pdf-verifier-free-rl-for-llms-via-intrin]. Ablation: removing $\sqrt{T}$ collapses GSM8K from **81.80%** to **0.08%** and code to **0.00%** (3B); removing rank shaping drops MMLU-Pro by 8.90% (7B) [source:arxiv:2605.09920][source:aclanthology:pdf-verifier-free-rl-for-llms-via-intrin]. Training cost: VIGOR 3h35m (28.67 GPU-hrs) vs GT-Reward 2h34m (20.55 GPU-hrs) on 8×H800; LM-head-only variant 2h47m (22.31 GPU-hrs) [source:arxiv:2605.09920][source:aclanthology:pdf-verifier-free-rl-for-llms-via-intrin].

**Limitations.** Unclear transfer to open-ended generation (long-form writing, dialogue); gradient-norm computation adds AD overhead; proxy objective may not track downstream utility consistently [source:arxiv:2605.09920][source:aclanthology:pdf-verifier-free-rl-for-llms-via-intrin].

**Disagreement on Verifier Necessity.** DeepSeekMath-V2 and JURY-RL invest heavily in *external* verifiers (meta-verifier, Lean) to achieve high-precision rewards, accepting computational cost and coverage gaps. VIGOR argues that a *verifier-free* intrinsic signal suffices for math/code reasoning and transfers cross-domain, but acknowledges it is a proxy that may misalign. JURY-RL's ResZero shows that even when formal verification fails ($\delta=0$), a carefully designed fallback using *internal* consensus structure outperforms zero reward or majority-vote reward—suggesting a middle ground where external verification gates high-confidence rewards while intrinsic structure guides the rest [source:arxiv:2604.25419][source:arxiv:2605.09920]. DeepSeekMath-V2's meta-verifier similarly uses an *internal* LLM to audit the verifier, creating a hierarchy of internal checks [source:arxiv:2511.22570].

## Unit Test Generation and Rewards

### CURE: Co-Evolving Coder and Unit Tester via Self-Play

CURE trains a single LLM to alternate between **coder** and **unit tester** roles without ground-truth code supervision [source:neurips:co-evolving-llm-coder-and-unit-tester-vi]. The reward is derived from a **pairwise interaction matrix** between generated codes $\{C\}$ and generated tests $\{\mathcal{T}\}$:
- Coder produces correct and incorrect solutions.
- Tester learns to discriminate: tests that pass ground-truth (if available) but fail buggy solutions are rewarded.
- For Long-CoT models, a **response-length-guided reward transformation** encourages shorter, efficient test generation.

**Results.** CURE-4B: **+6.2%** test-time scaling accuracy, **+25.1%** agentic unit-test generation accuracy vs base; **64.8% inference efficiency** vs Qwen3-4B in test generation; models at 4B, 7B, 14B scales [source:neurips:co-evolving-llm-coder-and-unit-tester-vi]. Limitation: traditional unit-test RL requires ground-truth code, which is costly; CURE removes this but the paper does not explicitly state its own limitations [source:neurips:co-evolving-llm-coder-and-unit-tester-vi].

### UTRL: Adversarial RL for Unit Test Generation

UTRL iteratively trains two separate LLMs—unit test generator $\mathcal{M}_{\text{UT}}$ and code generator $\mathcal{M}_{\text{code}}$—in an adversarial loop using only instruction–ground-truth-code pairs $\mathcal{D} = \{(I, C^*)\}$ [source:arxiv:2508.21107].

**Rewards.** For a generated test suite $\mathcal{T}$ and code set $\mathcal{C}$:
- **Discrimination Reward** (test quality):

$$
R_{\text{disc}}(\mathcal{T}, \mathcal{C}, C^*) = \frac{1}{|\mathcal{C}|} \sum_{C \in \mathcal{C}} \left[ 1 - \prod_{T \in \mathcal{T}} (1 - \text{Pass}(C, T))^{\text{Pass}(C^*, T)} \right]
$$

$\text{Pass}(C^*, T)$ filters invalid tests (those failing ground-truth).
- **Validity Reward** (test coverage/validity):

$$
R_{\text{valid}}(\mathcal{T}, C^*, \tau) = \frac{\sum_{T \in \mathcal{T}} \text{Pass}(C^*, T)}{\max(|\mathcal{T}|, \tau)}
$$

$\tau$ = minimum desired test cases, prevents trivial short tests.
- **Unit Test Generator Reward**: $r_{\text{UT}} = \lambda R_{\text{disc}} + (1-\lambda) R_{\text{valid}}$ ($\lambda=0.85$ optimal).
- **Code Generator Reward**: pass rate on *valid* tests:

$$
R_{\text{code}}(C, \mathcal{T}, C^*) = \frac{\sum_{T \in \mathcal{T}} \text{Pass}(C, T) \cdot \text{Pass}(C^*, T)}{\sum_{T \in \mathcal{T}} \text{Pass}(C^*, T)}
$$

Both updated via GRPO [source:arxiv:2508.21107].

**Results.** Best-of-32 code accuracy: UTRL Qwen3-4B **14.9%** (w/ Qwen3-8B coder) / **17.3%** (w/ Qwen3-14B) vs SFT **11.7%** / **14.0%**; outperforms GPT-4.1/4o. Qwen3-14B UTRL: **15.0%** / **17.7%**. vs CURE: **+4.4% / +3.2%** (Qwen2.5-7B-Instruct). LiveCodeBench: UTRL Qwen3-4B **59.9%** / **59.3%** vs GPT-4.1. Unit test fidelity (Spearman vs GT): Qwen3-14B **0.827**, Qwen3-4B **0.794** (GPT-4.1 lower). Code generator pass@1: UTRL Qwen3-4B **15.3%** vs SFT **3.6%**, near GT-test upper bound **15.9%**. Iterative training: iteration 2 tests surpass iteration 1 and GPT-4.1 despite initial discrimination reward drop (0.626→0.375→0.447). Ablation: $\lambda=0$ → validity 64.3% but low BoN; $\lambda=1$ → validity 49.7%, degraded BoN; without $R_{\text{valid}}$ → >50% invalid tests; without clipping in $R_{\text{valid}}$ → trivial few-test suites [source:arxiv:2508.21107].

**Limitations.** Performance gap vs ground-truth unit tests persists; evaluated only on competitive programming; fixed test-count per suite (adaptive length future work) [source:arxiv:2508.21107].

**Disagreement on Supervision Signal.** CURE uses *no* ground-truth code—pure self-play interaction matrix—while UTRL *requires* ground-truth code $C^*$ to filter invalid tests ($\text{Pass}(C^*, T)$) and compute validity/discrimination rewards. UTRL's adversarial two-model design achieves higher fidelity (0.827 vs CURE's 0.593) and stronger code generation (15.3% pass@1 vs CURE's unreported), but at the cost of needing $C^*$. CURE claims "without relying on ground-truth code solutions" as its core contribution; UTRL explicitly leverages them for reward shaping [source:neurips:co-evolving-llm-coder-and-unit-tester-vi][source:arxiv:2508.21107]. This is a fundamental trade-off: **annotation-free co-evolution** (CURE) vs **annotation-leveraged adversarial refinement** (UTRL).

## DeepSeekMath-V2 Deep Dive: Architecture and Training Dynamics

### Data Curation and Cold-Start
- **Source**: 17,503 proof problems from Art of Problem Solving (AoPS) contests.
- **Generator**: DeepSeek-V3.2-Exp-Thinking variant, iterative refinement.
- **Annotation**: Math experts score proofs 0 (fundamentally flawed), 0.5 (minor errors), 1 (complete/rigorous) per high-level rubrics.
- **Dataset**: $\mathcal{D}_\nu = \{(X_i, Y_i, A_i)\}$ for verifier; $\mathcal{D}_{m\nu} = \{(X_i, Y_i, Z_i, \bar{A}_i)\}$ for meta-verifier [source:arxiv:2511.22570].

### Verifier RL Objective Details
The verifier $c_\varphi$ outputs analysis $Z'$ and score $A'$. Format reward $R_{\text{format}}$ is an indicator for two required phrases. Score reward $R_{\text{score}} = 1 - |A' - A_i|$. The objective:

$$
\max_{c_\varphi} \mathbb{E}_{(X_i,Y_i,A_i)\sim\mathcal{D}_\nu,\;(Z'_i,A'_i)\sim c_\varphi(\cdot|X_i,Y_i)} \left[ R_{\text{format}}(Z'_i) \cdot R_{\text{score}}(A'_i, A_i) \right]
$$

Training on $\mathcal{D}_\nu$ alone yields a verifier that predicts scores accurately but hallucinates issues. Meta-verifier corrects this [source:arxiv:2511.22570].

### Meta-Verifier Training Loop
1. Train initial verifier $c_\varphi$ on $\mathcal{D}_\nu$.
2. Experts score verifier analyses $Z_i$ per meta-rubrics $\mathcal{I}_{m\nu}$ → $\mathcal{D}_{m\nu}$.
3. Train meta-verifier $c_\eta$ on $\mathcal{D}_{m\nu}$ with same RL objective (format + score rewards).
4. Enhanced verifier reward: $R_V = R_{\text{format}} \cdot R_{\text{score}} \cdot R_{\text{meta}}$ where $R_{\text{meta}}$ is meta-verifier's quality score.
5. Retrain verifier on $\mathcal{D}_\nu \cup \mathcal{D}_{m\nu}$ with $R_V$ [source:arxiv:2511.22570].

### Generator Self-Verification Mechanics
Generator prompted to output proof $Y$ then self-analysis $Z$ (same format as verifier). Verifier evaluates both:
- $s = A$ (proof score)
- $ms = \bar{A}$ (self-analysis quality)
- $s'$ = generator's self-predicted score (extracted from $Z$)
Reward:

$$
R = R_{\text{format}}(Y, Z) \cdot \left( \alpha \cdot s + \beta \cdot \left[ R_{\text{score}}(s', s) \cdot ms \right] \right), \quad \alpha=0.76,\; \beta=0.24
$$

This forces the generator to *calibrate* its self-prediction $s'$ to the verifier's $s$, while also producing high-quality self-analyses [source:arxiv:2511.22570].

### Automated Labeling Pipeline (Verifier Self-Improvement)
For each new proof $Y$ from improved generator:
1. Generate $k$ independent verifications $\{(Z^{(j)}, A^{(j)})\}$.
2. For each with $A^{(j)} \in \{0, 0.5\}$, generate $m$ meta-verifications; keep if majority confirm issues.
3. If $\ge 9$ valid analyses agree on lowest score → label with that score.
4. If no valid issues across all $k$ → label 1.
5. Else → discard or human review.
This creates a flywheel: better generator → harder proofs → better verifier training data [source:arxiv:2511.22570].

### Sequential Refinement and High-Compute Search
- **Pass@1** improves substantially with max sequential attempts (self-verification guides refinement).
- **Best@32**: Self-selected best proofs achieve significantly higher verification scores than thread average, demonstrating accurate self-assessment.
- **Competition results** (high-compute search): IMO 2025 5/6 (83.3%), CMO 2024 4/6+partial (73.8%), Putnam 2024 118/120, IMO-ProofBench competitive with DeepMind DeepThink [source:arxiv:2511.22570].

## Current Status and Trajectory

**RLVR with external verifiers (Lean, unit tests) is the default for high-stakes math/code reasoning**—evidenced by DeepSeekMath-V2's competition-level results and JURY-RL's label-free formal verification matching GT rewards. The trajectory is **rising** for hybrid systems that combine external verification (when available) with robust fallbacks (ResZero, meta-verification) to maintain coverage. **Unit-test RL is rising** but split: UTRL's adversarial two-model design with ground-truth filtering achieves higher fidelity, while CURE's fully self-play approach scales without annotations—both active. **Verifier-free intrinsic rewards (VIGOR) are emerging** but not yet default; they show strong cross-domain transfer on math→code but unproven on open-ended tasks, and gradient-norm overhead remains a scaling concern. **Meta-verification (DeepSeekMath-V2) is a rising pattern**—using an LLM to audit the verifier—likely to generalize to other RLVR pipelines. **Automated labeling flywheels** (generator→hard cases→verifier retraining) are **early but promising**; not widely reported outside DeepSeekMath-V2. **Formal verification (Lean) adoption is rising** but bottlenecked by autoformalization success rates (66.7% on AIME 2025 vs 97.1% on miniF2F) and compute overhead (~200s/step) [source:arxiv:2511.22570][source:arxiv:2604.25419][source:arxiv:2605.09920][source:arxiv:2508.21107][source:neurips:co-evolving-llm-coder-and-unit-tester-vi].

## Key Takeaways

- **Verifier design is a spectrum**: external deterministic (Lean, unit tests) → LLM-based with meta-auditing (DeepSeekMath-V2) → intrinsic proxy (VIGOR gradient norm). Each trades off precision, coverage, and compute.
- **Meta-verification is a force multiplier**: DeepSeekMath-V2's meta-verifier lifts analysis quality 0.85→0.96 without hurting score accuracy; JURY-RL's ResZero provides a principled fallback when formal verification is inconclusive.
- **Unit-test RL requires discriminative rewards**: UTRL's $R_{\text{disc}}$ (fail buggy code, pass ground-truth) + $R_{\text{valid}}$ (coverage + validity) outperforms naive pass-rate rewards; CURE achieves similar via self-play interaction matrix without ground-truth code.
- **Automated labeling flywheels close the data gap**: DeepSeekMath-V2's pipeline (k verifications + m meta-verifications → majority vote → label) turns generator improvements into verifier training data without human annotation.
- **Length correction is critical for intrinsic rewards**: VIGOR's $\sqrt{T}$ scaling prevents collapse (GSM8K 81.8% → 0.08% without it); UTRL's $\tau$ in $R_{\text{valid}}$ prevents trivial short tests.
- **Formal verification precision > recall**: Lean verifier 84.5% precision vs LLM-judge 75.9% (JURY-RL); false positives are costlier than missed verifications.
- **Cross-domain transfer exists but is asymmetric**: VIGOR trained on math improves code (+1.91 pp); UTRL trained on code improves math (implied by adversarial loop); DeepSeekMath-V2's theorem-proving verifier does not directly transfer to code.

## Related Topics

- [Verifiable rewards (RLVR)](verifiable-rewards.md) — foundational framework for deterministic reward design
- [RL for reasoning models](rl-for-reasoning.md) — broader reasoning RL context including math/code
- [Process vs outcome reward models](process-vs-outcome-rewards.md) — verifier design as process supervision
- [Reward hacking in RLHF](reward-hacking.md) — why verifiable rewards mitigate hacking
- [GRPO (Group Relative Policy Optimization)](grpo.md) — optimization algorithm used in JURY-RL, VIGOR, UTRL
- [Self-improvement and self-play RL](self-improvement-and-self-play.md) — CURE and DeepSeekMath-V2's generator→verifier flywheel
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md) — unit-test generation as tool-use
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md) — DeepSeekMath-V2's sequential refinement and high-compute search

## References
- [source:arxiv:2511.22570] [DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning](https://arxiv.org/html/2511.22570v1)
- [source:neurips:co-evolving-llm-coder-and-unit-tester-vi] [Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](https://neurips.cc/virtual/2025/poster/115329)
- [source:github:awesome-rlvr-reinforcement-learning-with] [Awesome RLVR — Reinforcement Learning with Verifiable Rewards](https://github.com/opendilab/awesome-RLVR)
- [source:labelstud:reinforcement-learning-from-verifiable-r] [Reinforcement Learning from Verifiable Rewards](https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/)
- [source:arxiv:2604.25419] [JURY-RL: Votes Propose, Proofs Dispose for Label-Free RLVR - arXiv](https://arxiv.org/html/2604.25419v1)
- [source:arxiv:2605.09920] [Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward - arXiv](https://arxiv.org/html/2605.09920v1)
- [source:aclanthology:pdf-verifier-free-rl-for-llms-via-intrin] [[PDF] Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward](https://aclanthology.org/2026.findings-acl.1606.pdf)
- [source:arxiv:2508.21107] [Learning to Generate Unit test via Adversarial Reinforcement Learning](https://arxiv.org/html/2508.21107v2)
