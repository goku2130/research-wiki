---
title: Process vs outcome reward models
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2305.20050
- arxiv:2406.06592
- arxiv:2510.08049
- arxiv:2211.14275
- arxiv:2312.08935
open_questions:
- Does the ORM-emulates-PRM phenomenon (Uesato et al.) hold for coding, agentic tool
  use, or open-ended reasoning where "undesirable behaviors" can lead to high-rated
  outcomes?
- What is the optimal granularity for step segmentation — fixed token windows, newline-delimited,
  semantic units, or learned boundaries — and how does it interact with PRM architecture?
- Can PRMs trained on automated labels (with known noise distributions) be made robust
  via noise-aware losses, or is a small human-verified "anchor set" necessary for
  calibration?
- 'How to close the proxy-reward gap: high ProcessBench/PRMBench accuracy does not
  guarantee effective MCTS/beam-search navigation — what training objectives align
  static step-accuracy with dynamic search utility?'
---

Process Reward Models (PRMs) and Outcome Reward Models (ORMs) represent two paradigms for supervising LLM reasoning: ORMs score only the final answer, while PRMs assign credit or blame to individual reasoning steps. This distinction fundamentally changes the credit-assignment problem, the data-collection burden, and the resulting model's ability to detect and correct intermediate errors.

## Definitions and Formalization

An **Outcome Reward Model (ORM)** $R_{\text{ORM}}(x, y)$ maps a full solution $y$ (conditioned on prompt $x$) to a scalar reward reflecting final-answer correctness. It is typically trained as a binary classifier on $(x, y, \mathbb{I}[\text{ans}(y)=a^*])$ tuples [source:arxiv:2211.14275]. A **Process Reward Model (PRM)** $R_{\text{PRM}}(x, y_{1:t})$ scores a prefix $y_{1:t}$ of a step-by-step solution. The overall solution score is often aggregated multiplicatively:

$$
\text{Score}(y) = \prod_{t=1}^{T} R_{\text{PRM}}(x, y_{1:t}) \quad \text{or} \quad \sum_{t=1}^{T} \log R_{\text{PRM}}(x, y_{1:t})
$$

[source:arxiv:2305.20050]. The survey [source:arxiv:2510.08049] formalizes the discriminative PRM reward as $r_t = \sigma(f_\theta(x, s_{1:t})) \in (0,1)$ where $f_\theta$ is a learned scoring head.

**Trace error** — reaching a correct final answer via flawed reasoning — is the central safety and reliability motivation for PRMs [source:arxiv:2211.14275]. ORMs cannot directly penalize trace errors; PRMs explicitly target them.

## Training Paradigms: Human vs Automated Supervision

| Paradigm | Method | Key Example | Scale | Cost |
|----------|--------|-------------|-------|------|
| Human annotation | Experts label each step as correct/incorrect/neutral | PRM800K (75k solutions, 800k step labels) | Medium | Very high |
| Monte Carlo completion | Sample $N$ rollouts from step $t$; label by fraction reaching $a^*$ | Math-Shepherd HE/SE | High | High (decoding) |
| MCTS with binary search | Tree search + binary search to locate first error | OmegaPRM | Very high | Moderate (75× vs brute force) |
| Semi-automated / hybrid | Small human "anchors" + automated expansion | Survey [source:arxiv:2510.08049] | Variable | Medium |

**Human annotation** (PRM800K) provides high-fidelity labels but is expensive and may hit a "human ceiling" when models surpass annotator ability [source:arxiv:2510.08049]. **Active learning** on PRM800K — surfacing "convincing wrong answers" (high PRM score, wrong final answer) — yielded ~2.6× data-efficiency gains [source:arxiv:2305.20050].

**Automated supervision** trades label noise for scale. Math-Shepherd defines step quality as the probability of reaching the golden answer $a^*$ under a completer policy [source:arxiv:2312.08935]:
- **Hard Estimation (HE)**: $y_{s_i}^{HE} = \mathbb{I}[\exists a_j \in A: a_j = a^*]$
- **Soft Estimation (SE)**: $y_{s_i}^{SE} = \frac{1}{N}\sum_{j=1}^N \mathbb{I}[a_j = a^*]$

OmegaPRM improves efficiency by framing error localization as binary search on the reasoning chain, reducing Monte Carlo cost from $O(kM)$ to $O(k \log M)$ where $M$ is step count [source:arxiv:2406.06592]. It stores tree statistics $N(s)$, $\text{MC}(s)$, and a heuristic $Q(s,r) = \alpha^{1-\text{MC}(s)} \beta^{\text{len}(r)/L}$ that prioritizes "supposed-to-be-correct" wrong-answer rollouts [source:arxiv:2406.06592].

**Disagreement on label quality**: Lightman et al. treat human labels as ground truth; Math-Shepherd and OmegaPRM argue automated labels, despite noise, enable better coverage and distribution [source:arxiv:2312.08935][source:arxiv:2406.06592]. The survey notes automated pipelines risk "Echo Chambers" where models reinforce shared misconceptions [source:arxiv:2510.08049].

## PRM Architectures and Loss Functions

The survey [source:arxiv:2510.08049] categorizes PRMs into four families:

1. **Discriminative PRMs**: Directly predict $r_t = \sigma(f_\theta(x, s_{1:t}))$. Trained with:
   - Pointwise BCE: $\mathcal{L} = \mathbb{E}[-y_t \log r_t - (1-y_t)\log(1-r_t)]$
   - Pointwise MSE: $\mathcal{L} = \mathbb{E}[(r_t - y_t)^2]$
   - Pairwise: $\mathbb{P}_\theta(u \succ v) = \sigma(f_\theta(u)-f_\theta(v))$

2. **Generative PRMs**: "Think-then-judge" — generate critique $z_t \sim p_\phi(z_t|x,s_{1:t})$ then score $r_t = h_\psi(x,s_{1:t},z_t)$. Joint loss:

$$
\mathcal{L}_{\text{gen}} = -\log p_\phi(z_t^\star|x,s_{1:t}) + \lambda \text{BCE}(r_t, y_t)
$$

3. **Implicit PRMs**: Infer step rewards from outcome-only signals (e.g., via credit assignment algorithms).

4. **Other**: Graph-based, retrieval-augmented, hierarchical.

OmegaPRM found **pointwise soft labels** ($\hat{y} = \text{MC}(s)$) outperformed hard labels and pairwise objectives, achieving 70.1% per-step classification accuracy [source:arxiv:2406.06592]. Math-Shepherd used pointwise BCE on HE/SE labels [source:arxiv:2312.08935]. Lightman et al. trained PRM as a classifier on three-class human labels (positive/negative/neutral) [source:arxiv:2305.20050].

## Inference-Time Usage: Verification and Search

PRMs enable **test-time compute scaling** via:

| Strategy | Mechanism |
|----------|-----------|
| Best-of-N / Weighted Majority Voting | Score each candidate solution by $\prod_t r_t$ (or weighted sum with self-consistency) and pick top |
| Verification-guided decoding | At each step, sample $k$ continuations, score with PRM, keep top-$b$ (beam annealing) |
| MCTS / Guided Search | Use PRM as state-value heuristic in tree search (e.g., OmegaPRM's own MCTS) |

Lightman et al. used Best-of-1860 with product-of-step-probs, achieving 78.2% on MATH vs ORM 72.4% [source:arxiv:2305.20050]. Math-Shepherd combined PRM with self-consistency: $a_{\text{sc+rm}} = \arg\max_a \sum_i \mathbb{I}[a_i=a] \cdot \text{RM}(p, \mathcal{S}_i)$ [source:arxiv:2312.08935]. OmegaPRM used **weighted self-consistency decoding** with its PRM [source:arxiv:2406.06592].

**Key result**: On MATH500, OmegaPRM + weighted majority voting @64 lifted Gemini Pro from 51% → 69.4% and Gemma2-27B from 42.3% → 58.2% [source:arxiv:2406.06592]. Math-Shepherd's Best-of-256 reached 89.1% (Mistral-7B, GSM8K) and 48.1% (DeepSeek-67B, MATH) [source:arxiv:2312.08935].

## RL Integration: Step-Level Rewards

PRMs provide **dense, step-level rewards** for RL (typically PPO), replacing sparse outcome rewards. Math-Shepherd applied step-by-step PPO: at each step $t$, reward $r_t = R_{\text{PRM}}(x, y_{1:t})$ [source:arxiv:2312.08935]. This improved Mistral-7B from 77.9% → 84.1% (GSM8K) and 28.6% → 33.0% (MATH) [source:arxiv:2312.08935].

Uesato et al. compared **Final-Answer RL** (expert = correct final answer), **ORM-RL** (expert = high ORM score), and **PRM-RL** (per-step expert = high PRM score) [source:arxiv:2211.14275]. PRM-RL treated each step as an episode. The best final-answer error (12.7%) came from SFT + ORM-RL with ORM reranking, but PRM-RL achieved lower trace error (3.4% vs higher for ORM variants) [source:arxiv:2211.14275].

**Disagreement on RL necessity**: Lightman et al. focus on inference-time verification; Math-Shepherd and Uesato et al. show RL benefits. The survey notes PRMs accelerate policy convergence via better credit assignment but warns of **reward hacking** (length/verbosity bias) more severe than for ORMs [source:arxiv:2510.08049].

## Benchmark Results and Comparisons

| Model / Method | GSM8K | MATH | MATH500 | OOD (AP Calc, AMC, etc.) |
|----------------|-------|------|---------|--------------------------|
| ORM (Lightman) | — | 72.4% (Bo1860) | — | 68.9% (AP Calc, Bo100) |
| PRM (Lightman, human) | — | **78.2%** (Bo1860) | — | **86.7%** (AP Calc, Bo100) |
| Math-Shepherd PRM (Mistral-7B, Bo256) | 89.1% | 43.5% | — | Hungarian exam: +9 vs ORM |
| OmegaPRM (Gemini Pro, WMV@64) | 93.6% | — | 69.4% | — |
| OmegaPRM (Gemma2-27B, WMV@64) | 92.2% | — | 58.2% | — |
| o1-mini (ProcessBench) | — | — | — | 87.9% avg |
| ACTPRM-X (ProcessBench) | — | — | — | 76.0% avg |

**Critical nuance from Uesato et al.**: On GSM8K, ORM and PRM achieved **similar final-answer error rates** (~22-23%), but PRM reduced trace error (11.4% vs 19.8% for outcome-only RL) [source:arxiv:2211.14275]. They found ORM predictions **agreed more with PRM labels than with outcome labels** — in math, incorrect steps rarely yield correct answers, so ORM implicitly learns process supervision [source:arxiv:2211.14275]. This may not hold in domains where "undesirable behaviors" help achieve high-rated outcomes [source:arxiv:2211.14275].

**Data efficiency**: Math-Shepherd reports PRM outperforms ORM by ~4% with only 10k training instances [source:arxiv:2312.08935]. OmegaPRM generated 1.5M annotations at 75× efficiency vs brute-force MC [source:arxiv:2406.06592].

## Limitations and Open Challenges

1. **Resource efficiency**: PRMs require step-wise labels — significantly more expensive than ORMs [source:arxiv:2510.08049].
2. **Length/verbosity hacking**: PRMs more susceptible than ORMs to rewarding longer, verbose steps [source:arxiv:2510.08049].
3. **Cross-domain generalization**: Step definitions vary (math vs code vs agentic); PRMs show limited transfer [source:arxiv:2510.08049].
4. **Annotation noise**: Automated labels contain false positives/negatives; impact not fully quantified [source:arxiv:2312.08935][source:arxiv:2406.06592].
5. **Granularity tension**: Rigid segmentation (e.g., by newline) conflicts with semantic reasoning flow [source:arxiv:2510.08049].
6. **Proxy-reward gap**: High static benchmark accuracy ≠ effective dynamic search navigation [source:arxiv:2510.08049].
7. **Dependence on golden answers**: Automated methods (Math-Shepherd, OmegaPRM) require reference answers, limiting open-ended use [source:arxiv:2406.06592].
8. **Human ceiling**: Annotators cannot reliably label steps beyond their own reasoning ability [source:arxiv:2510.08049].

## Current Status and Trajectory

PRMs are **rising rapidly** as the dominant paradigm for mathematical and code reasoning, but **not yet default** for general alignment. Key evidence:
- **Rising**: OmegaPRM (2024) demonstrates scalable automation; major labs (OpenAI o1, Google) deploy PRM-like verifiers; survey (2024) catalogs explosion of architectures and benchmarks (ProcessBench, PRMBench) [source:arxiv:2406.06592][source:arxiv:2510.08049].
- **Not default**: Most open RLHF pipelines (e.g., standard PPO/DPO) still use ORMs or scalar reward models; PRM integration requires step-wise tokenization, specialized loss, and inference-time search infrastructure not yet standardized [source:arxiv:2510.08049].
- **Fading?** No — but the **human-annotation paradigm** (PRM800K-style) is likely fading in favor of automated/hybrid pipelines due to cost and ceiling effects [source:arxiv:2510.08049].
- **Hedge**: The Uesato et al. finding that ORMs emulate PRMs *in mathematics* suggests PRM gains may be domain-specific; not widely reported whether this holds for coding, agentic tasks, or creative reasoning [source:arxiv:2211.14275].

## Key Takeaways

- **PRMs provide granular credit assignment** via per-step rewards $r_t$, enabling detection of trace errors that ORMs miss [source:arxiv:2211.14275][source:arxiv:2305.20050].
- **Automated annotation (MC completion, MCTS) has largely replaced human labeling** for scale, despite label noise [source:arxiv:2312.08935][source:arxiv:2406.06592].
- **OmegaPRM's binary-search MCTS** reduces annotation cost 75× vs brute-force MC; pointwise soft labels work best [source:arxiv:2406.06592].
- **In math, ORMs implicitly learn process supervision** because wrong steps rarely yield right answers — PRM advantage may not transfer to other domains [source:arxiv:2211.14275].
- **PRMs excel at test-time scaling** (Best-of-N, MCTS, beam annealing) and improve RL convergence, but are more prone to length hacking and generalization gaps [source:arxiv:2510.08049][source:arxiv:2312.08935].
- **No standardized PRM training/inference stack exists yet** — architectures (discriminative, generative, implicit), losses (BCE, MSE, pairwise, soft), and aggregation (product, sum, weighted) remain active design choices [source:arxiv:2510.08049].

## Related Topics

- [Reward modeling for LLMs](reward-modeling.md) — broader context of scalar vs step-wise reward models
- [RL for reasoning models](rl-for-reasoning.md) — RL with dense PRM rewards for math/code
- [RL for math and code](rl-for-math-and-code.md) — domain-specific PRM applications
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md) — PRMs as inference-time search heuristics
- [Reward hacking in RLHF](reward-hacking.md) — PRM-specific vulnerabilities (length bias, verbosity)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md) — PRM-based reranking strategies
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md) — step-level PPO with PRM rewards
- [Verifiable rewards (RLVR)](verifiable-rewards.md) — outcome verification as PRM training signal
- [Self-improvement and self-play RL](self-improvement-and-self-play.md) — PRMs in iterative self-training loops

## References
- [source:arxiv:2305.20050] [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)
- [source:arxiv:2406.06592] [Improve Mathematical Reasoning in Language Models by Automated Process Supervision (OmegaPRM)](https://arxiv.org/abs/2406.06592)
- [source:arxiv:2510.08049] [A Survey of Process Reward Models: From Outcome to Process Supervision](https://arxiv.org/html/2510.08049v3)
- [source:arxiv:2211.14275] [Outcome-supervised Verifiers for Planning in Complex Reasoning Tasks](https://arxiv.org/abs/2211.14275)
- [source:arxiv:2312.08935] [Math-Shepherd: A Label-free Step-by-Step Verifier for Mathematical Reasoning (Wang et al., 2023)](https://arxiv.org/abs/2312.08935)
