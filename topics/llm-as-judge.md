---
title: LLM-as-judge
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2306.05685
- arxiv:2411.15594
- arxiv:2410.02736
- aclanthology:a-systematic-study-of-position-bias-in-l
- arxiv:2604.16790
- wandb:exploring-llm-as-a-judge-weights-biases
- mbrenndoerfer:position-bias-in-llm-judges-measurement-
open_questions:
- Can architectural modifications (e.g., attention masking, positional encoding changes)
  fundamentally reduce position bias, or is it an inherent property of autoregressive
  next-token prediction?
- How do judge biases interact with RLHF optimization dynamics—does optimizing against
  a biased judge amplify specific failure modes (sycophancy, verbosity, format mimicry)
  in predictable ways?
- What is the minimal human-audit protocol that reliably bounds judge error rates
  for high-stakes domains (medical, legal, code deployment) without full human re-evaluation?
- Do fine-tuned specialist judges trained on bias-augmented data (CALM perturbations)
  generalize robustness to unseen bias compositions, or do they overfit to the perturbation
  taxonomy?
---

LLM-as-a-judge has become the de facto standard for scalable evaluation of generative models, replacing static n-gram metrics with context-sensitive reasoning. Yet the paradigm introduces systematic biases—position, verbosity, self-enhancement, and framing effects—that distort rankings and undermine reproducibility unless rigorously quantified and mitigated.

## Formal Framework and Evaluation Paradigms

The evaluation process is formally defined as $\mathcal{E} \leftarrow \mathcal{P}_{\mathcal{LLM}}(x \oplus \mathcal{C})$, where $\mathcal{E}$ is the evaluation output, $\mathcal{P}_{\mathcal{LLM}}$ the judge's autoregressive distribution, $x$ the input (query, response(s), optional reference), and $\mathcal{C}$ the prompt context [source:arxiv:2411.15594]. Reliability $\mathcal{R}$ is then a function $\mathcal{R} \leftarrow f_{\mathrm{R}}(\mathcal{P}_{\mathrm{LLM}}, x, \mathcal{C})$ capturing consistency under perturbation [source:arxiv:2411.15594].

Three primary judging formats dominate:
1. **Pairwise comparison**: The judge selects the better of two responses (or declares a tie) [source:arxiv:2306.05685][source:arxiv:2411.15594].
2. **Single-answer grading**: The judge assigns an absolute score on a discrete or continuous scale [source:arxiv:2306.05685][source:arxiv:2411.15594].
3. **Reference-guided grading**: A ground-truth or reference solution is provided to anchor the judgment, critical for math and reasoning [source:arxiv:2306.05685][source:arxiv:2411.15594].

List-wise ranking (selecting the best among $k \ge 3$ candidates) is less studied but shows sharply degraded robustness [source:arxiv:2410.02736][source:aclanthology:a-systematic-study-of-position-bias-in-l].

## Judge Model Architectures and Selection

Judges fall into two families [source:arxiv:2411.15594]:
- **General-purpose LLMs**: Proprietary APIs (GPT-4, Claude, Gemini) or large open models (Llama-3). GPT-4 remains the most validated judge, achieving 85% agreement with human experts on MT-bench (S2, non-tie), exceeding inter-human agreement (81%) [source:arxiv:2306.05685]. Weights & Biases reports >80% agreement, comparable to human–human consistency [source:wandb:exploring-llm-as-a-judge-weights-biases].
- **Fine-tuned specialist judges**: Models like PandaLM, JudgeLM, and Auto-J are trained via a three-stage pipeline: data collection → prompt design → fine-tuning [source:arxiv:2411.15594]. These aim to reduce API dependency and improve domain alignment but often struggle with out-of-distribution generalization [source:arxiv:2411.15594].

Model capacity strongly modulates bias susceptibility. Smaller models (e.g., Qwen2.5-Coder-3B) exhibit stronger, less consistent position biases and rely on shallow heuristics [source:mbrenndoerfer:position-bias-in-llm-judges-measurement-][source:arxiv:2604.16790]. Capable judges (GPT-4, Claude-3.5-Sonnet, Llama-3-70B) show high repetition stability (RS > 0.95), confirming bias is systematic, not stochastic [source:aclanthology:a-systematic-study-of-position-bias-in-l].

## Bias Taxonomy and Quantification

Wang et al. (2024) categorize 12 biases across three families [source:arxiv:2410.02736]:

| Family | Biases |
|--------|--------|
| **Content-related** | Verbosity, Fallacy-Oversight, Sentiment, Authority |
| **Context-related** | Position, Compassion-Fade, Bandwagon, Distraction |
| **Identity/Process-related** | Diversity, CoT Influence, Self-Enhancement, Refinement-Aware |

The CALM framework quantifies bias via automated perturbation: a perturbation function $g(\cdot)$ (instantiated by GPT-4o) modifies the prompt $P$ to $\hat{P}$ without altering factual correctness, then measures **Robustness Rate** $\mathbf{RR} = \frac{1}{|D|}\sum \mathbb{I}(y^i = \hat{y}^i)$ and **Consistency Rate** $\mathbf{CR} = \frac{1}{|D|}\sum \mathbb{I}(y^i = y_{\text{rand}}^i)$ [source:arxiv:2410.02736].

### Position Bias

Position bias—the preference for a candidate based on its ordinal position—is the most pervasive and best-quantified failure mode.

**Measurement.** The standard "swap" protocol evaluates each pair in both orders $(A,B)$ and $(B,A)$ [source:arxiv:2306.05685][source:mbrenndoerfer:position-bias-in-llm-judges-measurement-][source:aclanthology:a-systematic-study-of-position-bias-in-l]. Key metrics:
- **Swap Consistency (SC)**: $\frac{1}{N}\sum_{i=1}^N \mathbb{1}[v_1^{(i)} = v_2^{(i)}]$ [source:mbrenndoerfer:position-bias-in-llm-judges-measurement-].
- **Position Consistency (PC)**: Fraction of instances where the same solution is preferred regardless of position [source:aclanthology:a-systematic-study-of-position-bias-in-l].
- **Preference Fairness (PF)**: Normalized to $[-1,1]$; positive = recency bias, negative = primacy bias [source:aclanthology:a-systematic-study-of-position-bias-in-l].
- **First-Position Bias (FPB)**: Among inconsistent pairs, $\frac{n_{A\to A}}{n_{A\to A} + n_{B\to B}}$ [source:mbrenndoerfer:position-bias-in-llm-judges-measurement-].

**Empirical findings.**
- GPT-4 default consistency 65.0% (Zheng et al.) vs. swap consistency 0.7–0.8 in practice (20–30% of judgments flip on swap) [source:arxiv:2306.05685][source:mbrenndoerfer:position-bias-in-llm-judges-measurement-]. Among flips, GPT-4 prefers the first position 60–70% of the time [source:mbrenndoerfer:position-bias-in-llm-judges-measurement-].
- Few-shot prompting raises GPT-4 consistency to 77.5%; Claude-v1 remains at 23.8% [source:arxiv:2306.05685].
- **Task and benchmark dependence**: Claude-3.5-Sonnet is nearly fair on MT-Bench (PF = 0.01) but shows strong recency bias on DevBench (PF = 0.22) [source:aclanthology:a-systematic-study-of-position-bias-in-l]. **Disagreement**: Zheng et al. report position bias as a consistent property of the judge; the ACL study shows it is highly volatile across tasks and benchmarks.
- **List-wise scaling**: When evaluating 3–4 candidates, most models' RR drops below 0.5 [source:arxiv:2410.02736]. Pairwise studies underestimate real-world bias in multi-candidate settings (e.g., Chatbot Arena side-by-side battles).
- **Quality gap correlation**: PC increases with the answer quality gap $\delta_0$; larger disparities make judges more consistent and fair [source:aclanthology:a-systematic-study-of-position-bias-in-l].
- **Code-specific**: Qwen2.5-Coder-3B has a strong baseline preference for position A; no-bias accuracy falls below chance (38–49%) when the gold answer is at B [source:arxiv:2604.16790]. GPT maintains ~80% no-bias accuracy but remains sensitive [source:arxiv:2604.16790].

**Mechanistic drivers.** Recency bias (attention to late tokens), primacy bias (first item as reference), and the "U-curve" attention pattern (strong begin/end, weak middle) jointly produce position bias [source:mbrenndoerfer:position-bias-in-llm-judges-measurement-]. Prompt and input length have negligible impact; task output length is a statistically significant but minimally influential predictor of PF [source:aclanthology:a-systematic-study-of-position-bias-in-l].

### Verbosity Bias

Verbosity bias—the tendency to favor longer responses—interacts with position bias because length often correlates with position in swapped evaluations.

- Zheng et al.: GPT-4 fails a "repetitive list" verbosity attack 8.7% of the time; GPT-3.5 and Claude-v1 fail 91.3% [source:arxiv:2306.05685].
- Regression quantification: $s_i = \alpha + \beta_1 q_i + \beta_2 \ell_i + \epsilon_i$; verbosity bias present if $\beta_2 > 0$ significantly. Some judges show $\beta_2/\beta_1 \approx 0.4$, i.e., length contributes ~40% as much as quality [source:mbrenndoerfer:position-bias-in-llm-judges-measurement-].
- **Code domain**: Verbosity consistently pushes judges toward candidate B (the second position) [source:arxiv:2604.16790]. **Disagreement**: General studies treat verbosity as a content bias favoring longer outputs regardless of position; the code audit finds it acts as a positional prior favoring the second slot. This may reflect the specific prompt template used in CodeJudgeBench.

### Self-Enhancement and Identity Biases

- **Self-enhancement**: Models rate their own outputs higher. Wang et al. find "significant bias" across all six tested models [source:arxiv:2410.02736]. Zheng et al. label it "inconclusive" for GPT-4 [source:arxiv:2306.05685]. **Disagreement**: The CALM framework's automated perturbation (swapping model identity labels) detects self-enhancement more sensitively than Zheng et al.'s direct comparison.
- **Compassion-fade / model-name bias**: Revealing model identity shifts judgments. In code evaluation, "Model-name" bias drops GPT's TestGen accuracy from 77.46% to 62.51% [source:arxiv:2604.16790].
- **Refinement-aware bias**: Judges favor responses known to be refined. Error rate $\text{ErrorRate}_{\text{RA}} = |1 - y_{\text{ref}}/y_{\text{ref}}'|$ [source:arxiv:2410.02736]. In code, "Refined" bias saturates accuracy when the refined answer is at A but collapses when at B [source:arxiv:2604.16790].

### Contextual and Framing Biases

- **Authority bias**: Citation format matters—book/quote formats influence more than URLs [source:arxiv:2410.02736].
- **Sentiment bias**: Emotional language (especially fear) shifts preferences [source:arxiv:2410.02736].
- **Bandwagon / distraction**: Majority-opinion cues and irrelevant details sway judgments [source:arxiv:2410.02736].
- **CoT influence**: Step-by-step reasoning generally improves accuracy (GLM-4 +7%, GPT-4-Turbo +0.7%) but can also act as a positional prior pushing toward candidate A in code evaluation [source:arxiv:2410.02736][source:arxiv:2604.16790].

## Mitigation Strategies

| Strategy | Mechanism | Efficacy / Caveats |
|----------|-----------|-------------------|
| **Conservative swap** | Run both orders; declare win only if same winner both times | Reduces position bias but doubles cost; does not eliminate bias if judge is inconsistent in both orders [source:arxiv:2306.05685] |
| **Neutral labels** | Use "A/B" not "1/2"; place criteria before responses; reduce inter-response distance | Dampens but cannot eliminate bias [source:mbrenndoerfer:position-bias-in-llm-judges-measurement-] |
| **Chain-of-Thought** | Ask judge to solve independently first (math) or reason stepwise | Improves math grading (GPT-4 failure 14/20 → 6/20 → 3/20 with reference) [source:arxiv:2306.05685]; but CoT can become a positional prior [source:arxiv:2604.16790] |
| **Reference-guided grading** | Provide ground-truth solution | Strongest for verifiable domains [source:arxiv:2306.05685] |
| **Decomposed criteria** | Break coarse metrics into fine-grained sub-criteria | Improves reliability per Gu et al. [source:arxiv:2411.15594] |
| **Multi-judge ensembles** | Majority vote, average scores, conservative consensus, specialized panels | Reduces variance; cost scales linearly [source:wandb:exploring-llm-as-a-judge-weights-biases][source:arxiv:2411.15594] |
| **Fine-tuning / RL** | Train specialist judges (PandaLM, JudgeLM) or use GRPO for online refinement | Improves in-distribution alignment; generalization remains a challenge [source:arxiv:2411.15594] |
| **Constrained decoding / logit normalization** | FSMs (XGrammar: 100× speedup) for structured output; score smoothing via token probabilities | Addresses output brittleness, not semantic bias [source:arxiv:2411.15594] |

**Critical limitation**: High consistency (CR) ≠ correctness. In code evaluation, injecting Sentiment/Refined biases inflates Qwen2.5-Coder-3B's CR from 50% to ~86% while anchoring on the bias—a "dangerous illusion of reliability" [source:arxiv:2604.16790].

## Meta-Evaluation and Reliability Metrics

Agreement with human judgment is the gold standard:

$$
\text{Agreement} = \frac{\sum_{i \in D} \mathbb{I}(S_{\text{llm}} = S_{\text{human}})}{|D|}
$$

supplemented by Cohen's $\kappa$ and Spearman's $\rho$ [source:arxiv:2411.15594].

Benchmarks:
- **MT-bench**: 80 multi-turn questions, 8 categories [source:arxiv:2306.05685].
- **Chatbot Arena**: ~30k crowdsourced battles [source:arxiv:2306.05685].
- **CALM / JudgeBench**: 4,356 samples for bias quantification [source:arxiv:2410.02736][source:arxiv:2411.15594].
- **CodeJudgeBench**: Triplets for CodeGen, CodeRepair, TestGen [source:arxiv:2604.16790].
- **DevBench**: 14 tasks, 8 questions each, for position bias analysis [source:aclanthology:a-systematic-study-of-position-bias-in-l].

Factor analysis via bidirectional stepwise regression (AIC) identifies answer quality gap as the dominant predictor of position consistency; prompt length and input length are negligible [source:aclanthology:a-systematic-study-of-position-bias-in-l].

## Current Status and Trajectory

LLM-as-a-judge is the **default evaluation paradigm** for open-ended generation, embedded in major leaderboards (Chatbot Arena, MT-bench, AlpacaEval) and RLHF pipelines (reward modeling, GRPO ranking). The trajectory is **rising in adoption but hardening in scrutiny**: early work (Zheng et al. 2023) established viability; subsequent work (Wang et al. 2024, ACL 2024, CodeJudgeBench 2024) has shifted from "does it work?" to "where does it fail?" and "how do we audit it?".

Key unresolved tensions:
- **Proprietary vs. open judges**: API judges (GPT-4, Claude) offer strongest raw agreement but suffer version drift, opacity, and cost; open specialist judges lag in generalization [source:arxiv:2411.15594][source:arxiv:2604.16790]. Not widely reported that any open judge matches GPT-4's cross-domain robustness.
- **Bias mitigation vs. bias measurement**: Most mitigations are prompt-level heuristics (swap, neutral labels, CoT) that reduce but do not eliminate bias. No widely adopted *architectural* or *training-time* intervention guarantees unbiased judging.
- **Domain specificity**: Code, math, and reasoning have verifiable ground truth, enabling reference-guided grading and process reward models (PRMs). For open-ended chat, creative writing, and subjective tasks, no ground truth exists—judge bias *is* the evaluation noise floor.
- **Agentic cascades**: In multi-step agent workflows, sequential LLM judgments accumulate errors; systemic risk is acknowledged but not quantified at scale [source:arxiv:2411.15594].

The field is not fading; it is **consolidating around rigorous bias audits (CALM, systematic position studies) and hybrid human–LLM evaluation** for high-stakes decisions.

## Key Takeaways

- LLM-as-a-judge achieves human-level agreement (80–85%) on helpfulness for strong models (GPT-4, Claude-3.5) but exhibits systematic biases: position, verbosity, self-enhancement, authority, sentiment, and refinement-aware.
- Position bias is the most pervasive: 20–30% of pairwise judgments flip on order swap; list-wise (3–4 candidates) drops robustness below 0.5 for most models.
- Bias magnitude depends on **judge capacity**, **task domain**, **quality gap between candidates**, and **number of candidates**—not a fixed property of the judge alone.
- Mitigations (swap, CoT, reference-guided, neutral labels, ensembles) reduce but do not eliminate bias; high consistency can mask systematic error ("dangerous illusion of reliability").
- Meta-evaluation requires human-annotated ground truth; automated bias frameworks (CALM) enable scaling but cannot replace targeted human audits for high-stakes domains.
- Open specialist judges remain less robust than proprietary APIs; version drift and reproducibility are unresolved for API-based judges.

## Related Topics

- [Reward modeling for LLMs](reward-modeling.md) — Judges as reward models in RLHF
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md) — Preference learning from judge outputs
- [GRPO (Group Relative Policy Optimization)](grpo.md) — Uses LLM judges for group-wise ranking
- [RLAIF (RL from AI feedback)](rlaif.md) — Constitutional AI and AI feedback loops
- [Alignment and win-rate evals](alignment-and-winrate-evals.md) — Evaluation methodology using judges
- [Judging bias and contamination](judging-bias-and-contamination.md) — Deep dive on bias types and data contamination
- [Length and format bias](length-and-format-bias.md) — Specific analysis of verbosity and formatting effects
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md) — Judge agreement with user premises
- [Reward hacking in RLHF](reward-hacking.md) — Over-optimization against judge proxies
- [Process vs outcome reward models](process-vs-outcome-rewards.md) — Step-wise (PRM) vs final-output judging

## References
- [source:arxiv:2306.05685] [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [source:arxiv:2411.15594] [A Survey on LLM-as-a-Judge (Gu et al., 2024)](https://arxiv.org/html/2411.15594v6)
- [source:arxiv:2410.02736] [Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge (Wang et al., 2024)](https://arxiv.org/html/2410.02736v1)
- [source:aclanthology:a-systematic-study-of-position-bias-in-l] [A Systematic Study of Position Bias in LLM-as-a-Judge (ACL Anthology)](https://aclanthology.org/2025.ijcnlp-long.18.pdf)
- [source:arxiv:2604.16790] [Bias in the Loop: Auditing LLM-as-a-Judge for Software Engineering](https://arxiv.org/html/2604.16790v1)
- [source:wandb:exploring-llm-as-a-judge-weights-biases] [Exploring LLM-as-a-Judge (Weights & Biases)](https://wandb.ai/site/articles/exploring-llm-as-a-judge/)
- [source:mbrenndoerfer:position-bias-in-llm-judges-measurement-] [Position Bias in LLM Judges: Measurement and Mitigation](https://mbrenndoerfer.com/writing/position-bias-in-llm-judges)
