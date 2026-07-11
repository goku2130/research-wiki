---
title: RL for math and code
maturity: comprehensive
updated: '2026-07-11'
sources:
- promptfoo:reinforcement-learning-with-verifiable-r
- arxiv:2506.03136
- github:awesome-rlvr-reinforcement-learning-with
- github:a-survey-of-reinforcement-learning-for-l
- arxiv:2501.12948
- arxiv:2402.03300
- arxiv:2511.22570
- arxiv:2510.19817
- arxiv:2508.21107
- arxiv:2604.25872
- arxiv:2204.02515
open_questions:
- Is the ~70% "search compression" estimate from promptfoo generalizable across model
  scales and task distributions, or does capability expansion dominate at larger scales
  (as suggested by DeepSeek-R1-Zero's emergent behaviors)?
- Can meta-verification (DeepSeekMath-V2) be extended to verify the *meta-verifier
  itself*, or does this require human-in-the-loop calibration at the top level?
- UTRL eliminates ground-truth unit tests but requires ground-truth *code solutions*
  $C^*$. Can the adversarial framework be extended to settings where only instructions
  $I$ are available (no $C^*$)?
- The beneficial-error theory (arXiv:2604.25872) is proven for linear softmax bandits.
  Does the "stalling prevention" mechanism hold for transformer policies with non-linear
  feature learning in multi-step MDPs?
---

Reinforcement learning with verifiable rewards (RLVR) has become the dominant paradigm for advancing mathematical reasoning and code generation in open LLMs, replacing learned reward models with deterministic verifiers such as unit tests, compilers, and formal proof checkers. The DeepSeekMath and DeepSeek-R1 lineages demonstrate that group-relative policy optimization (GRPO) over such verifiers can produce frontier-level reasoning capabilities without human-annotated preference data.

## Verifier design in RLVR

RLVR replaces the neural reward model of classical RLHF with a **deterministic verification function** $r(s, a) \in \{0, \gamma\}$ that returns a non-zero reward only when the completion $a$ passes an objective check for prompt $s$ [source:github:awesome-rlvr-reinforcement-learning-with]. The verifier may be a unit-test suite, a compiler, a formal proof kernel (e.g., Lean), an SQL executor, or a schema validator [source:github:awesome-rlvr-reinforcement-learning-with]; [source:promptfoo:reinforcement-learning-with-verifiable-r]. Because the reward is binary and ground-truth, the approach avoids reward-model overoptimization and enables **intrinsic safety**—every reward is traceable to a transparent verifier run [source:github:awesome-rlvr-reinforcement-learning-with].

**Partial verifiers** are a critical failure mode: if the verifier checks syntax but not execution (e.g., SQL parsing without running the query), the model learns to exploit the gap and "cheat" for reward [source:promptfoo:reinforcement-learning-with-verifiable-r]. The RLVR loop optionally includes **verifier refinement**, where the verifier itself is hardened or expanded to cover new edge cases, allowing the agent to self-bootstrap its learning signal [source:github:awesome-rlvr-reinforcement-learning-with].

**Disagreement on gain attribution:** The promptfoo analysis argues that RLVR gains are largely **search compression**—the model becomes more efficient at sampling reasoning paths it could already generate—rather than **capability expansion** (learning new reasoning paths) [source:promptfoo:reinforcement-learning-with-verifiable-r]. They propose a compression ratio $\frac{\text{RLVR pass@1} - \text{Base pass@1}}{\text{Base pass@k} - \text{Base pass@1}}$; a ratio $> 0.7$ suggests search compression dominates. In one case, pass@1 rose from 40% to 65% (+25 pp) while pass@8 rose only from 75% to 77% (+2 pp), yielding ~71% compression [source:promptfoo:reinforcement-learning-with-verifiable-r]. By contrast, DeepSeek-R1-Zero reports an "aha moment" where the model *emergently* develops self-reflection and verification behaviors during pure RL, suggesting genuine capability expansion [source:arxiv:2501.12948]. The survey of RL for LRMs notes this as a "foundational problem" under active debate [source:github:a-survey-of-reinforcement-learning-for-l].

**Spurious rewards** are another documented pathology: Qwen2.5-Math-7B showed a 21.4% improvement on MATH-500 using *random* rewards versus 29.1% with ground-truth rewards, implying the RL process itself can refine internal pathways independent of the reward signal [source:promptfoo:reinforcement-learning-with-verifiable-r].

**Unit test rewards for vision-language tasks:** olmOCR 2 extends the RLVR paradigm to document OCR by using binary unit tests extracted from synthetic HTML as verifiable rewards [source:arxiv:2510.19817]. The system generates synthetic HTML from real PDFs using a general VLM (Claude-Sonnet-4), then programmatically extracts unit tests for: text presence/absence, natural reading order, table cell accuracy, math formula rendering equivalence (via KaTeX), and baseline robustness (no repeated n-grams). Training a Qwen2.5-VL-7B model with GRPO ($\beta=0.01$ KL) on these unit-test rewards achieves 82.4 on olmOCR-Bench, outperforming GPT-4o (68.9) and DeepSeek-OCR (75.7). The authors note that continuous metrics like edit distance correlate poorly with practical correctness, motivating the binary unit-test approach. Model souping (averaging 6 seeds) further improves stability [source:arxiv:2510.19817].

**Meta-verification for proof assessment:** DeepSeekMath-V2 addresses the "generation-verification gap" in theorem proving by introducing a **meta-verifier** that evaluates the quality of the verifier's own analysis [source:arxiv:2511.22570]. The verifier $\pi_\varphi$ scores proofs on a 3-point rubric (1=rigorous, 0.5=minor errors, 0=fundamentally flawed). The meta-verifier checks whether identified defects actually exist and whether the score matches the findings. The enhanced verifier reward becomes $R_V = R_{\text{format}} \cdot R_{\text{score}} \cdot R_{\text{meta}}$, where $R_{\text{meta}}$ is the meta-verifier's quality score. This prevents the verifier from hallucinating issues to justify low scores. Verifier analysis quality improved from 0.85 to 0.96 (meta-verifier assessed) [source:arxiv:2511.22570].

## Unit-test rewards and co-evolution (CURE)

The CURE framework (Co-Evolving LLM Coder and Unit Tester) addresses the scarcity of ground-truth code solutions by training a single policy to act as both **coder** and **unit tester** [source:arxiv:2506.03136]. For a task $q$, the model generates $n$ candidate solutions $\{s_j\}$ and $m$ task-derived unit tests $\{u_k\}$. Execution yields a binary matrix $\mathcal{B}^\star \in \{0,1\}^{n \times (m + t_q)}$ where $t_q$ is the number of ground-truth tests [source:arxiv:2506.03136].

**Solution reward** is the count of passed ground-truth tests:

$$
\mathcal{R}_{s_j}^{\star} = \sum_{l=1}^{t_q} \mathcal{B}_{j, m+l}^{\star}
$$

[source:arxiv:2506.03136].

**Unit-test reward (reward precision)** penalizes "naive" tests that are overly permissive. Let $\mathcal{I}_{s_j} = \prod_{l=1}^{t_q} \mathcal{B}_{j, m+l}^{\star}$ indicate whether solution $s_j$ passes all ground-truth tests. The reward for generated test $u_k$ is:

$$
\mathcal{R}_{u_k}^{\star} = -\sum_{l=1}^{n}(1-\mathcal{I}_{s_l})\mathcal{B}_{l,k}^{\star} + \left(\prod_{l=1}^{n}\mathcal{I}_{s_l}\mathcal{B}_{l,k}^{\star}\right)\left(\sum_{l=1}^{n}(1-\mathcal{I}_{s_l})\right)
$$

[source:arxiv:2506.03136]. The first term penalizes tests that pass incorrect solutions; the second term rewards tests that *only* pass correct solutions (when at least one incorrect solution exists). This optimizes the **precision** of the generated test as a classifier of solution correctness.

The policy is optimized with a PPO-style objective including a KL penalty [source:arxiv:2506.03136]. For long-CoT models, a **response-length-guided reward transformation** penalizes overly long responses, reducing average unit-test generation length to 64.8% of the original [source:arxiv:2506.03136].

**Results:** ReasonFlux-Coder-7B/14B (from Qwen2.5-Instruct) improved one-shot code generation by 5.3% and Best-of-N by 9.0%, outperforming Qwen-Coder, DeepSeek-Coder, and Seed-Coder [source:arxiv:2506.03136]. As a downstream unit tester for GPT-4o, ReasonFlux-Coder-4B improved one-shot performance by 7.0% while reducing API costs [source:arxiv:2506.03136]. The framework also serves as a **label-free reward model** for RL on base models, achieving improvements comparable to ground-truth supervision [source:arxiv:2506.03136].

**Limitation:** CURE still requires ground-truth *unit tests* to define the correctness indicator $\mathcal{I}_{s_j}$; it eliminates the need for ground-truth *code* but not for ground-truth *tests* [source:arxiv:2506.03136].

**Adversarial co-evolution without ground-truth tests (UTRL):** The UTRL framework removes the need for *any* ground-truth unit tests by framing unit test generation as an adversarial game between a test generator $\mathcal{M}_{\text{UT}}$ and a code generator $\mathcal{M}_{\text{code}}$ [source:arxiv:2508.21107]. Given only instruction-code pairs $(I, C^*)$, $\mathcal{M}_{\text{UT}}$ generates tests $\mathcal{T}$ to maximize a **discrimination reward** $R_{\text{disc}}$ (fraction of sampled imperfect codes $\mathcal{C}$ detected by at least one valid test) and a **validity reward** $R_{\text{valid}}$ (tests pass on $C^*$, normalized by $\max(|\mathcal{T}|, \tau)$ to prevent trivial test sets). $\mathcal{M}_{\text{code}}$ is trained to pass $\mathcal{M}_{\text{UT}}$'s tests. As $\mathcal{M}_{\text{code}}$ improves, $\mathcal{M}_{\text{UT}}$ must generate increasingly discriminative tests, creating co-evolution.

Key formulas:

$$
R_{\text{disc}}(\mathcal{T}, \mathcal{C}, C^*) = \frac{1}{|\mathcal{C}|} \sum_{C \in \mathcal{C}} \left[ 1 - \prod_{T \in \mathcal{T}} (1 - \text{Pass}(C, T))^{\text{Pass}(C^*, T)} \right]
$$

$$
R_{\text{valid}}(\mathcal{T}, C^*, \tau) = \frac{\sum_{T \in \mathcal{T}} \text{Pass}(C^*, T)}{\max(|\mathcal{T}|, \tau)}
$$

$$
r_{\text{UT}} = \lambda R_{\text{disc}} + (1-\lambda) R_{\text{valid}}
$$

$$
R_{\text{code}}(C, \mathcal{T}, C^*) = \frac{\sum_{T \in \mathcal{T}} \text{Pass}(C, T) \cdot \text{Pass}(C^*, T)}{\sum_{T \in \mathcal{T}} \text{Pass}(C^*, T)}
$$

[source:arxiv:2508.21107].

**Results:** UTRL-trained Qwen3-4B unit tests achieve 14.9% code accuracy as evaluators for Best-of-32 (vs. 11.7% SFT, 13.3% GPT-4.1). UTRL Qwen3-14B achieves Spearman's $\rho=0.827$ fidelity to ground-truth test evaluation (outperforming GPT-4.1). The UTRL code generator reaches 15.3% pass@1 (vs. 15.9% upper bound with ground-truth tests). UTRL outperforms CURE (14.1% vs 9.7% code accuracy with Qwen3-8B generator) [source:arxiv:2508.21107].

**Limitations:** Performance gap vs. rigorously verified ground-truth tests remains; evaluated only on competitive programming; fixed test count per task rather than adaptive [source:arxiv:2508.21107].

## DeepSeekMath and GRPO

DeepSeekMath-Base 7B is initialized from DeepSeek-Coder-Base-v1.5 7B and continually trained on 500B tokens (56% DeepSeekMath Corpus, 20% GitHub code, 10% arXiv, 10% Common Crawl, 4% AlgebraicStack) [source:arxiv:2402.03300]. The **DeepSeekMath Corpus** (120B tokens) was built via an iterative fastText classifier: seed from OpenWebMath → classify Common Crawl → identify high-math domains → manual annotation → retrain classifier (4 iterations) [source:arxiv:2402.03300].

**SFT** produces DeepSeekMath-Instruct 7B on 776K CoT/PoT/tool-integrated examples [source:arxiv:2402.03300].

**GRPO** replaces PPO's critic with a group baseline. For a question $q$, sample $G$ outputs $\{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)$. The advantage for output $i$ is:

$$
\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}
$$

where $r_i$ is the reward for the full output $o_i$ (token-level advantages use the same scalar $r_i$) [source:arxiv:2402.03300]; [source:arxiv:2501.12948]. The GRPO objective:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i,t} \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

[source:arxiv:2402.03300]. The KL term uses the unbiased estimator:

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - \log\frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1
$$

[source:arxiv:2402.03300].

**Key findings from DeepSeekMath:**
- **Code pre-training** significantly benefits math reasoning both with and without tool use [source:arxiv:2402.03300].
- **arXiv data** (MathPile, ArXiv-RedPajama) provided *no notable improvement* on adopted benchmarks, contrary to common practice [source:arxiv:2402.03300].
- **RL primarily improves output distribution** (Maj@K) rather than fundamental capability (Pass@K stable) [source:arxiv:2402.03300].
- **Weaknesses:** geometry/theorem-proving (triangles, ellipses); limited few-shot gains due to scale [source:arxiv:2402.03300].

**Results:** DeepSeekMath-RL 7B achieves 51.7% on MATH (Top1), 60.9% with self-consistency (64 samples), 88.2% on GSM8K [source:arxiv:2402.03300]. Base model (36.2% MATH) outperforms Minerva 540B (33.6%) [source:arxiv:2402.03300]. RL on GSM8K+MATH improves CMATH from 84.6% to 88.8% [source:arxiv:2402.03300].

**DeepSeekMath-V2: Self-verifiable theorem proving.** DeepSeekMath-V2 extends the paradigm to natural-language theorem proving by replacing final-answer rewards with a **proof generator + verifier + meta-verifier** framework trained via GRPO [source:arxiv:2511.22570]. The generator produces a proof $Y$ and a self-analysis $Z$; reward is $R = R_{\text{format}}(Y,Z) \cdot (\alpha R_Y + \beta R_Z)$ with $\alpha=0.76, \beta=0.24$. $R_Y$ is the verifier score; $R_Z$ combines self-assessment accuracy and meta-verification of the self-analysis. An automated labeling pipeline scales verifier training without experts: generate multiple verifications, validate with meta-verifier, label low-score if $\ge 9$ valid analyses agree. Results: Putnam 2024 **118/120** (human max 90), IMO 2025 **5/6 solved (83.3%, gold)**, CMO 2024 **73.8% (gold)**. Most challenging IMO problems remain difficult [source:arxiv:2511.22570].

## DeepSeek-R1: scaling pure RL for reasoning

DeepSeek-R1-Zero applies **pure GRPO** to DeepSeek-V3-Base *without any SFT* [source:arxiv:2501.12948]. The rule-based reward combines:
- **Accuracy reward:** deterministic verification (math answer matching, code compilation/execution) [source:arxiv:2501.12948].
- **Format reward:** incentivizes `reasoning`...`/reasoning` and `answer`...`/answer` tags [source:arxiv:2501.12948].
- **Language consistency reward:** $\frac{\text{Num(Words}_{\text{target}})}{\text{Num(Words)}}$ to reduce language mixing [source:arxiv:2501.12948].

Total reward: $R_{\text{rule}} = R_{\text{acc}} + R_{\text{format}}$ [source:arxiv:2501.12948]. GRPO advantage uses group mean/std normalization as above [source:arxiv:2501.12948].

**Emergent behavior:** During training, the model spontaneously develops long CoT with self-reflection and verification ("aha moment") [source:arxiv:2501.12948]. AIME 2024 Pass@1 rises from 15.6% to 77.9% (86.7% with self-consistency) [source:arxiv:2501.12948].

DeepSeek-R1 (final) uses a **four-stage pipeline**:
1. **Cold-start SFT** on a small set of high-quality human-aligned trajectories [source:arxiv:2501.12948].
2. **First RL stage** (GRPO) for reasoning + language consistency [source:arxiv:2501.12948].
3. **Rejection sampling + SFT**: 600K reasoning samples (via rejection sampling) + 200K non-reasoning samples [source:arxiv:2501.12948].
4. **Second RL stage**: rule-based rewards (reasoning) + model-based rewards (helpfulness/harmlessness) [source:arxiv:2501.12948].

**Final results:** AIME 2024 Pass@1 79.8%; MATH-500 Pass@1 97.3%; Codeforces rating 2029; AlpacaEval 2.0 +25%, ArenaHard +17% from final RL stage [source:arxiv:2501.12948].

**Limitations:** suboptimal structured output/tool use; "overthinking" on simple queries; language mixing outside EN/ZH; high prompt sensitivity (zero-shot recommended); limited software-engineering gains due to RL evaluation cost; pure RL susceptible to reward hacking with neural (non-rule-based) reward models [source:arxiv:2501.12948].

## Theoretical analysis of imperfect rewards

**When errors can be beneficial.** A theoretical analysis of policy gradient with proxy rewards $r_P$ vs. ground truth $r_G$ in linear softmax bandits categorizes reward errors by their effect on ground-truth reward increase [source:arxiv:2604.25872]:

- **Harmful Type I (Reward Hacking):** High $r_P$ for low $r_G$ outputs.
- **Harmful Type II (Stalling):** Mediocre $r_P$ for low $r_G$ outputs, causing policy to concentrate and stall on suboptimal outputs.
- **Benign Type I:** $r_P(y) < V_P(\theta_0)$ (initial expected proxy reward), so $y$ never attracts probability.
- **Benign Type II:** $y$ has negligible probability under $\pi_{\theta_0}$.
- **Beneficial Type I:** Low $r_P$ assigned to an output with *mediocre* $r_G$ prevents stalling, steering policy toward the optimal $y_\star$.

**Key theorem:** With orthonormal features, if initial $\pi_{\theta_0}(y_\star)$ is small relative to a mediocre $y_{\text{med}}$, time to reach high $r_G$ when maximizing $r_G$ directly is $\Omega(\pi_{\theta_0}(y_\star)^{-14/13})$, while maximizing a proxy $r_P$ that assigns low reward to $y_{\text{med}}$ requires only $\mathcal{O}(\pi_{\theta_0}(y_\star)^{-1})$. The gap can be arbitrarily large [source:arxiv:2604.25872].

**Partial rewards can be detrimental:** In instruction-following, rewarding partial correctness (e.g., 0.5 for 1 of 2 constraints) causes the policy to learn only the easier constraint if the initial policy is significantly more likely to produce partially correct than fully correct outputs [source:arxiv:2604.25872].

**Harm-Aware Ranking Accuracy (HAcc):** Proposed metric ignores incorrect rankings if the dispreferred output's reward is below the empirical expected proxy reward $\hat{V}_P(x; \theta)$:

$$
\text{HAcc}(r_P; \mathcal{S}) := \frac{1}{|\mathcal{S}|} \sum_{(x,y^+, y_k^-) \in \mathcal{S}} \mathbb{1} \left[ \max_{k} r_P(x, y_k^-) < \max\{r_P(x, y^+), \hat{V}_P(x; \theta)\} \right]
$$

HAcc correlates better with ground-truth reward increase than standard accuracy across Llama, OLMo, Qwen families, though correlations remain below 0.4 [source:arxiv:2604.25872].

**Limitations:** Analysis restricted to linear softmax policies, exact gradients, bandit settings; even HAcc shows weak correlation with actual RL performance; partial rewards may still help if fully correct outputs have near-zero initial probability [source:arxiv:2604.25872].

## Inferring rewards from language

A Bayesian framework infers general reward functions $\theta$ from natural language utterances across repeated interactions, enabling generalization to unseen contexts [source:arxiv:2204.02515]. The listener $L_2$ maintains a posterior over linear reward parameters $\theta$ given utterances $u$ and contexts $\mathcal{M}$:

$$
p(\theta \mid u_{1:i}, \mathcal{M}_{1:i}) \propto p(\theta \mid u_i, \mathcal{M}_i) \times p(\theta \mid u_{1:i-1}, \mathcal{M}_{1:i-1})
$$

The speaker model $S_1$ mixes an **action component** (eliciting correct action in current context) and a **reward component** (describing general preferences), governed by nearsightedness $\alpha$:

$$
p_{S_1}(u \mid \theta, \mathcal{M}) = \alpha p_{\text{action}}(u \mid \theta, \mathcal{M}) + (1-\alpha) p_{\text{reward}}(u \mid \theta)
$$

The action component uses a Boltzmann optimality model $p_{\text{opt}}(\xi \mid \theta, \mathcal{M}) \propto \exp(\beta r_\theta(\xi; \mathcal{M}))$ and a reference model $p_{\text{refer}}$ from a base listener $L_{\text{base}}$. Evaluated on the **FLIGHTPREF** dataset (2,568 utterances, 813 games, 8-dimensional reward vectors), the full model ($\alpha=0.5$) achieves **59.1% held-out accuracy** (selecting optimal flight in unseen option sets), outperforming action-only (52.8%) and reward-only (57.8%) baselines. Performance is comparable to an oracle with nearly 3 perfectly known features [source:arxiv:2204.02515].

**Limitations:** Base listener errors propagate (skipping low-confidence updates improves accuracy by 6%); ambiguity in integrating evidence (e.g., "cheap JetBlue flight" — general preference or unique identifier?); linear rewards only [source:arxiv:2204.02515].

## Current status and trajectory

RLVR with GRPO is the **default, rising paradigm** for math and code reasoning in open LLMs. The DeepSeekMath/R1 lineage and the CURE/UTRL co-evolution frameworks demonstrate that deterministic verifiers (unit tests, compilers, execution) can replace human preference data at the frontier. The ecosystem is consolidating around GRPO (or its variants like REINFORCE++) as the optimization backbone, with infrastructure converging on **verl**, **OpenRLHF**, and **open-r1** [source:github:awesome-rlvr-reinforcement-learning-with]. However, **fundamental disagreements persist** on whether gains reflect search compression vs. capability expansion [source:promptfoo:reinforcement-learning-with-verifiable-r] vs. [source:arxiv:2501.12948], and the survey of RL for LRMs explicitly lists reward design and gain attribution as unresolved "foundational problems" [source:github:a-survey-of-reinforcement-learning-for-l]. **Entropy collapse** during GRPO training—where in-distribution accuracy rises but OOD performance deteriorates—is a documented failure mode not yet widely solved [source:promptfoo:reinforcement-learning-with-verifiable-r]. **Overthinking** (excessive token use) remains a practical issue for deployment [source:arxiv:2501.12948]; [source:github:awesome-rlvr-reinforcement-learning-with]. The field is **not fading**; investment is accelerating (135 papers integrated from ICLR/ICML 2026 alone in the RLVR repo) [source:github:awesome-rlvr-reinforcement-learning-with], but the *theoretical understanding* of why RLVR works lags behind empirical scaling.

**New theoretical insights** challenge the assumption that all reward errors are harmful: beneficial errors can accelerate convergence by preventing stalling on mediocre outputs [source:arxiv:2604.25872], while partial rewards may inadvertently cap performance [source:arxiv:2604.25872]. **Meta-verification** [source:arxiv:2511.22570] and **adversarial co-evolution without ground-truth tests** [source:arxiv:2508.21107] are emerging as solutions to verifier scalability and completeness. **RLVR for vision-language** (olmOCR 2) demonstrates the paradigm's generality beyond text-only reasoning [source:arxiv:2510.19817].

## Key takeaways

- **RLVR replaces learned reward models with deterministic verifiers** (unit tests, compilers, executors), eliminating reward-model overoptimization and enabling traceable rewards [source:github:awesome-rlvr-reinforcement-learning-with]; [source:promptfoo:reinforcement-learning-with-verifiable-r].
- **GRPO is the de facto optimizer**: it removes the critic by computing advantages from group statistics (mean/std of rewards across $G$ samples per prompt) and uses a token-level PPO-clip objective with a KL penalty [source:arxiv:2402.03300]; [source:arxiv:2501.12948].
- **Verifier completeness is critical**: partial verifiers (e.g., syntax-only) induce reward hacking; CURE co-evolves the verifier (unit tests) with the generator to maximize *reward precision* [source:promptfoo:reinforcement-learning-with-verifiable-r]; [source:arxiv:2506.03136].
- **DeepSeekMath shows code pre-training > arXiv for math reasoning**; RL improves majority-vote performance (Maj@K) more than single-pass capability (Pass@K) [source:arxiv:2402.03300].
- **DeepSeek-R1-Zero demonstrates pure RL can elicit emergent reasoning** (self-reflection, long CoT) without any SFT, but requires rule-based rewards to avoid hacking [source:arxiv:2501.12948].
- **Gain attribution is contested**: compression ratio analysis suggests ~70%+ of gains may be search compression, yet "aha moment" narratives argue for genuine capability expansion [source:promptfoo:reinforcement-learning-with-verifiable-r]; [source:arxiv:2501.12948].
- **Spurious rewards and entropy collapse** are understudied failure modes: random rewards can improve performance on some models, and entropy decline correlates with OOD degradation [source:promptfoo:reinforcement-learning-with-verifiable-r].
- **Infrastructure is standardizing** on verl, OpenRLHF, open-r1 for GRPO/RLVR training at scale [source:github:awesome-rlvr-reinforcement-learning-with].
- **Meta-verification enables self-verifiable theorem proving**: DeepSeekMath-V2's verifier+meta-verifier framework achieves gold-level IMO/Putnam performance by training the verifier to assess proof rigor and the meta-verifier to validate the verifier's analysis [source:arxiv:2511.22570].
- **Adversarial co-evolution (UTRL) removes need for ground-truth unit tests**: a test generator and code generator co-evolve via discrimination and validity rewards, achieving fidelity to ground-truth evaluation exceeding GPT-4.1 [source:arxiv:2508.21107].
- **Not all reward errors are harmful**: theoretical analysis shows low proxy rewards on mediocre outputs can be *beneficial* by preventing policy stalling; partial rewards can cap performance if easier sub-tasks dominate early exploration [source:arxiv:2604.25872].
- **RLVR extends to vision-language**: olmOCR 2 uses binary unit tests (KaTeX rendering, table structure, reading order) as verifiable rewards for document OCR, achieving SOTA on olmOCR-Bench [source:arxiv:2510.19817].
- **Reward inference from language** enables generalization: Bayesian updating over reward parameters from natural language utterances outperforms action-only and reward-only baselines on held-out contexts [source:arxiv:2204.02515].

## Related topics

- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Length and format bias](length-and-format-bias.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)

## References
- [source:promptfoo:reinforcement-learning-with-verifiable-r] [Reinforcement Learning with Verifiable Rewards Makes Models Smarter](https://www.promptfoo.dev/blog/rlvr-explained/)
- [source:arxiv:2506.03136] [Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](https://arxiv.org/html/2506.03136v2)
- [source:github:awesome-rlvr-reinforcement-learning-with] [Awesome RLVR — Reinforcement Learning with Verifiable Rewards](https://github.com/opendilab/awesome-RLVR)
- [source:github:a-survey-of-reinforcement-learning-for-l] [A Survey of Reinforcement Learning for Large Reasoning Models](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)
- [source:arxiv:2501.12948] [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (RLVR/GRPO)](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2511.22570] [DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning](https://arxiv.org/abs/2511.22570)
- [source:arxiv:2510.19817] [olmOCR 2: Unit Test Rewards for Document OCR](https://arxiv.org/abs/2510.19817)
- [source:arxiv:2508.21107] [Learning to Generate Unit test via Adversarial Reinforcement Learning](https://arxiv.org/abs/2508.21107)
- [source:arxiv:2604.25872] [When Errors Can Be Beneficial: A Categorization of Imperfect Rewards for Policy Gradient](https://arxiv.org/abs/2604.25872)
- [source:arxiv:2204.02515] [Inferring Rewards from Language in Context](https://arxiv.org/abs/2204.02515)
