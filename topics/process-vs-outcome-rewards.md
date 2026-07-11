---
title: Process vs outcome reward models
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2211.14275
- arxiv:2312.08935
- arxiv:2510.08049
- arxiv:2305.20050
- arxiv:2406.06592
- arxiv:2506.12446
- arxiv:2505.13427
- arxiv:2506.18896
- arxiv:2605.30451
- arxiv:2508.04088
open_questions:
- Can ORM-to-PRM distillation (SP-PRM) match or exceed automated Monte Carlo pipelines
  at scale, and does it generalize beyond math/dialogue/summarization?
- How do trajectory-aware PRMs (ReasonFlux-PRM) perform on non-reasoning-model outputs
  (standard CoT, code, agentic traces)?
- Will verifier-gated RL (VeriGate) become standard for GRPO, or will pure outcome-based
  RL with better verifiers (RLVR) dominate?
- Can generative PRMs (GM-PRM) be trained without GPT-4o-level teachers, and do they
  generalize to non-mathematical multimodal reasoning?
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
| **ORM-guided PRM distillation** | **Derive PRM from ORM via dual-consistency training (score + preference)** | **SP-PRM** | **High** | **Low (no new human labels)** |
| **Multimodal MCTS + soft labels** | **MCTS rollouts with binary search + continuous MC scores as targets** | **MM-PRM** | **Very high (700k+ steps)** | **Moderate** |
| **Trajectory-aware supervision** | **Step-level (alignment/quality/coherence) + trajectory-level (template-guided) rewards** | **ReasonFlux-PRM** | **Medium (1k-59k)** | **Moderate (GPT-4o judge)** |
| **Generative critique + MC filtering** | **GPT-4o analyzes steps (intent, alignment, logic, refinement); MC hard estimation filters** | **GM-PRM** | **Medium (~20k)** | **Moderate** |

**Human annotation** (PRM800K) provides high-fidelity labels but is expensive and may hit a "human ceiling" when models surpass annotator ability [source:arxiv:2510.08049]. **Active learning** on PRM800K — surfacing "convincing wrong answers" (high PRM score, wrong final answer) — yielded ~2.6× data-efficiency gains [source:arxiv:2305.20050].

**Automated supervision** trades label noise for scale. Math-Shepherd defines step quality as the probability of reaching the golden answer $a^*$ under a completer policy [source:arxiv:2312.08935]:
- **Hard Estimation (HE)**: $y_{s_i}^{HE} = \mathbb{I}[\exists a_j \in A: a_j = a^*]$
- **Soft Estimation (SE)**: $y_{s_i}^{SE} = \frac{1}{N}\sum_{j=1}^N \mathbb{I}[a_j = a^*]$

OmegaPRM improves efficiency by framing error localization as binary search on the reasoning chain, reducing Monte Carlo cost from $O(kM)$ to $O(k \log M)$ where $M$ is step count [source:arxiv:2406.06592]. It stores tree statistics $N(s)$, $\text{MC}(s)$, and a heuristic $Q(s,r) = \alpha^{1-\text{MC}(s)} \beta^{\text{len}(r)/L}$ that prioritizes "supposed-to-be-correct" wrong-answer rollouts [source:arxiv:2406.06592].

**SP-PRM** introduces a distinct paradigm: **deriving a PRM from an existing ORM without new human annotations** [source:arxiv:2506.12446]. It constructs a partial-sequence dataset $\mathcal{D}_{\text{partial}}$ via Token-Level Truncation (TLT) or Stochastic Sampling Truncation (SST), then optimizes for **Score Consistency** (partial sequences align with final outcome score) and **Preference Consistency** (partial evaluations align with human preferences). A reference ORM $r_\phi$ provides guidance: samples are retained only if $r_\phi$'s preference matches the score-consistency requirement, and weighted by inverse entropy $w_t = 1/H_t$ where $H_t$ is the Shannon entropy of the reward gap. This resolves the **granularity mismatch** that causes myopic decoding when ORMs are used directly for process rewards.

**MM-PRM** extends automated supervision to **multimodal mathematical reasoning** [source:arxiv:2505.13427]. It builds a policy model (MM-Policy) by fine-tuning InternVL2.5-8B on 5.1M restructured CoT examples, then uses MCTS with binary search on a 10k seed dataset (MM-K12) to generate 700k+ step-level annotations. Crucially, it uses **soft labels** — continuous Monte Carlo scores $\hat{y}_t = \text{MC}(x_{<t}) \in [0,1]$ — as targets for cross-entropy loss, significantly outperforming hard labels (43% vs 34.4% on MM-K12 with Average aggregator).

**ReasonFlux-PRM** addresses the **trajectory-response format** of frontier reasoning models (e.g., DeepSeek-R1) where models generate lengthy, unorganized thinking trajectories before a structured final response [source:arxiv:2605.30451]. It introduces **trajectory-aware supervision** with two reward levels: (1) **Step-level** $r_t^{\text{step}}$ aggregating alignment (semantic similarity to response step), quality (GPT-4o logical soundness), and coherence (contrastive mutual information between adjacent steps); (2) **Trajectory-level** $r^{\text{final}}$ via template-guided generation — extract reasoning template $\mathcal{T}$, generate $N$ responses following $\mathcal{T}$, average correctness. Joint MSE loss trains both simultaneously.

**GM-PRM** pioneers a **generative multimodal PRM** that actively critiques and corrects steps rather than passively scoring [source:arxiv:2508.04088]. Training data from VisualPRM400K is processed by GPT-4o across four dimensions (Step Intent, Image Alignment, Reasoning Logic, Step Refinement), then filtered by consistency with Monte Carlo hard estimation (a step is correct iff $\exists$ completion reaching $a^*$). The model (Qwen2.5-VL-7B-Instruct) is trained via SFT to output critiques $c_i$ and judgments $j_i$ for each step.

**Disagreement on label quality**: Lightman et al. treat human labels as ground truth; Math-Shepherd and OmegaPRM argue automated labels, despite noise, enable better coverage and distribution [source:arxiv:2312.08935][source:arxiv:2406.06592]. The survey notes automated pipelines risk "Echo Chambers" where models reinforce shared misconceptions [source:arxiv:2510.08049]. **New disagreement**: SP-PRM shows ORM-guided distillation can achieve strong PRMs *without* automated rollouts or human step labels, challenging the necessity of Monte Carlo completion pipelines. MM-PRM and GM-PRM demonstrate that **multimodal PRMs require specialized data pipelines** (visual restructuring, GPT-4o critique generation) that differ substantially from text-only automation. ReasonFlux-PRM argues that **standard PRMs trained on polished CoT fail on raw thinking trajectories**, necessitating trajectory-aware architectures — a new axis of specialization.

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

**New architectures from recent work**:

5. **ORM-Distilled PRMs (SP-PRM)**: Trained via **weighted Bradley-Terry loss** on partial sequences [source:arxiv:2506.12446]:

$$
\mathcal{L}_{\text{SP-PRM}} = -\mathbb{E}_{(x, y_{<t}^w, y_{<t}^l) \sim \mathcal{D}_{\text{partial}}} w \log \sigma\left(r_\theta(x, y_{<t}^w) - r_\theta(x, y_{<t}^l)\right)
$$

   where $w = 1/H_t$ if reference ORM prefers $y_{<t}^w$, else $w=0$. $H_t$ is entropy of $p_t^w = \sigma(|r_\phi(x,y_{<t}^w) - r_\phi(x,y_{<t}^l)|)$.

6. **Multimodal Discriminative PRMs (MM-PRM)**: Special marker token $\sigma$ (instantiated as `<prm>`) interleaved after each step; predicts probability via softmax over "Yes"/"No" logits [source:arxiv:2505.13427]:

$$
p^{(i)} = \frac{\exp(z_{\text{Yes}}^{(i)})}{\exp(z_{\text{Yes}}^{(i)}) + \exp(z_{\text{No}}^{(i)})}, \quad \mathcal{L}_{\text{PRM}} = -\sum_i \left[\hat{y}^{(i)}\log p^{(i)} + (1-\hat{y}^{(i)})\log(1-p^{(i)})\right]
$$

   with soft labels $\hat{y}^{(i)} = \text{MC}(x_{<t})$.

7. **Trajectory-Aware PRMs (ReasonFlux-PRM)**: Joint step-level and trajectory-level MSE [source:arxiv:2605.30451]:

$$
\mathcal{L} = \sum_{i=1}^N \sum_{t=1}^{T^{(i)}} \lambda_{\text{step}} (R_\phi(s_t^{(i)} | \cdot) - r_t^{\text{step}})^2 + \lambda_{\text{final}} (R_\phi(y^{(i)}) - r^{\text{final}})^2
$$

   Step reward $r_t^{\text{step}}$ = softmax-weighted sum of alignment, quality, coherence scores.

8. **Generative Multimodal PRMs (GM-PRM)**: SFT to generate critique $c_i$ and judgment $j_i$ per step [source:arxiv:2508.04088]:

$$
f_{\text{GM-PRM}}: (Q, I, R) \mapsto (c_1, j_1, \dots, c_T, j_T)
$$

   Trained on GPT-4o analyses filtered by MC hard estimation consistency.

OmegaPRM found **pointwise soft labels** ($\hat{y} = \text{MC}(s)$) outperformed hard labels and pairwise objectives, achieving 70.1% per-step classification accuracy [source:arxiv:2406.06592]. Math-Shepherd used pointwise BCE on HE/SE labels [source:arxiv:2312.08935]. Lightman et al. trained PRM as a classifier on three-class human labels (positive/negative/neutral) [source:arxiv:2305.20050]. **MM-PRM confirms soft labels superiority in multimodal setting** (43% vs 34.4% on MM-K12). **ReasonFlux-PRM uses MSE on continuous rewards** (not classification), enabling fine-grained credit assignment for trajectory steps.

## Inference-Time Usage: Verification and Search

PRMs enable **test-time compute scaling** via:

| Strategy | Mechanism |
|----------|-----------|
| Best-of-N / Weighted Majority Voting | Score each candidate solution by $\prod_t r_t$ (or weighted sum with self-consistency) and pick top |
| Verification-guided decoding | At each step, sample $k$ continuations, score with PRM, keep top-$b$ (beam annealing) |
| MCTS / Guided Search | Use PRM as state-value heuristic in tree search (e.g., OmegaPRM's own MCTS) |
| **Refined-BoN (GM-PRM)** | **Active refinement: PRM critiques errors, policy regenerates from corrected step; iterate to improve pool** |
| **Trajectory-aware BoN (ReasonFlux-PRM)** | **Aggregate step + trajectory rewards: $\hat{r} = \frac{1}{T}\sum \hat{r}_t^{\text{step}} + \alpha \hat{r}^{\text{final}}$** |

Lightman et al. used Best-of-1860 with product-of-step-probs, achieving 78.2% on MATH vs ORM 72.4% [source:arxiv:2305.20050]. Math-Shepherd combined PRM with self-consistency: $a_{\text{sc+rm}} = \arg\max_a \sum_i \mathbb{I}[a_i=a] \cdot \text{RM}(p, \mathcal{S}_i)$ [source:arxiv:2312.08935]. OmegaPRM used **weighted self-consistency decoding** with its PRM [source:arxiv:2406.06592].

**Key result**: On MATH500, OmegaPRM + weighted majority voting @64 lifted Gemini Pro from 51% → 69.4% and Gemma2-27B from 42.3% → 58.2% [source:arxiv:2406.06592]. Math-Shepherd's Best-of-256 reached 89.1% (Mistral-7B, GSM8K) and 48.1% (DeepSeek-67B, MATH) [source:arxiv:2312.08935].

**SP-PRM** evaluated across dialogue (HH-RLHF), summarization (TL;DR), and reasoning (GSM8K) with 1B-8B models [source:arxiv:2506.12446]:
- 3.6–10.3% improvement in GPT-4 evaluation scores across tasks
- TL;DR: BoN-16 average reward +11.7% (0.60 → 0.67)
- GSM8K: CBS + SP-PRM accuracy +3.5% (1B), +2.5% (3B); BoN-16 reached 65.5% (1B), 69.5% (3B)
- AdvBench safety: ASR reduced 20% vs base RGS
- Score consistency Agreement Rate: from ~60% (ORM) to 55% at 5 tokens, 64.7% at 50 tokens

**MM-PRM** evaluated via BoN-16 across multimodal benchmarks [source:arxiv:2505.13427]:
| Model | MM-K12 | OlympiadBench | MathVista | MathVerse | MathVision |
|-------|--------|---------------|-----------|-----------|------------|
| MM-Policy → +MM-PRM | 33.92→42.80% | 15.41→24.00% | 62.93→67.60% | 42.99→46.27% | 21.74→27.11% |
| InternVL2.5-8B → +MM-PRM | 27.01→37.80% | 5.23→15.33% | 56.43→63.50% | 36.26→42.56% | 10.04→19.41% |
| InternVL2.5-78B → +MM-PRM | 42.24→48.80% | 30.98→34.67% | 69.48→73.20% | 50.18→54.47% | 31.50→33.26% |

**ReasonFlux-PRM** test-time scaling (Best-of-N) achieved **average 6.3% gain** across AIME, MATH500, GPQA-Diamond [source:arxiv:2605.30451]. SFT on 1k samples selected by ReasonFlux-PRM-7B outperformed 59k raw trajectories on MATH500 (84.8% vs 78.8%).

**GM-PRM** Refined-BoN: policy generates $N/2$ solutions, GM-PRM critiques and corrects errors, policy regenerates remaining $N/2$ from corrected prefixes, final selection by average "Correct" token probability [source:arxiv:2508.04088]. Average accuracy improvements: MiniCPM-V2.6-8B +5.9% (WeMath +12.4%), Llama-3.2-11B-Vision +4.5%, Qwen2.5-VL-7B +4.5%, InternVL3-8B +5.6%. Refined-BoN increased Pass@8 vs standard BoN: MiniCPM +0.9, Llama +1.3, InternVL3 +0.9.

## RL Integration: Step-Level Rewards

PRMs provide **dense, step-level rewards** for RL (typically PPO), replacing sparse outcome rewards. Math-Shepherd applied step-by-step PPO: at each step $t$, reward $r_t = R_{\text{PRM}}(x, y_{1:t})$ [source:arxiv:2312.08935]. This improved Mistral-7B from 77.9% → 84.1% (GSM8K) and 28.6% → 33.0% (MATH) [source:arxiv:2312.08935].

Uesato et al. compared **Final-Answer RL** (expert = correct final answer), **ORM-RL** (expert = high ORM score), and **PRM-RL** (per-step expert = high PRM score) [source:arxiv:2211.14275]. PRM-RL treated each step as an episode. The best final-answer error (12.7%) came from SFT + ORM-RL with ORM reranking, but PRM-RL achieved lower trace error (3.4% vs higher for ORM variants) [source:arxiv:2211.14275].

**VeriGate** introduces a **verifier-gated extension of GRPO** that integrates process supervision only when outcome rewards are uninformative [source:arxiv:2605.30451]:
- **S1 Verifier Gating**: If group rewards mixed → standard GRPO; if all zero → activate PRM token-level supervision; if all correct → no PRM.
- **S2 Future-Cumulated Token Rewards (FCTR)**: Token in step $j$ gets $c_{i,j} = \sum_{k=j}^{S_i} r_{i,k}$ (sum of subsequent PRM step rewards).
- **S3 Group-Normalized Token Advantages**: $\bar{c} = \frac{\sum_{i,j} c_{i,j}}{\sum_i S_i}$, $A_{i,j} = (c_{i,j} - \bar{c}) / \sigma(c)$.
- **Effective Advantage**: $\widetilde{A}_{i,j} = A_i^{\text{GRPO}}$ if mixed rewards, else $A_{i,j}$ above.

VeriGate on Qwen2.5-Instruct (1.5B/7B) trained on MATH: **~20% accuracy gain (1.5B), ~12% (7B)** across six benchmarks. 7B model: AIME 6.67%→10.00%, AMC 42.17%→45.78%, Minerva 18.38%→20.22%. **Reward hacking mitigation**: Cross-PRM verification (Math-Shepherd evaluator) VeriGate 0.1424 vs PRM-as-ORM 0.0331. Faster reduction in zero-verifier-reward prompts.

**ReasonFlux-PRM** for online GRPO: composite reward $\boldsymbol{r}_{\text{new}} = (1-\beta) r_{\text{out}} + \beta \hat{r}$ where $\hat{r}$ is PRM reward [source:arxiv:2605.30451]. DeepSeek-R1-Distill-Qwen-7B on MATH500: 89.6% (rule-based) → 94.8% (ReasonFlux-PRM-7B). Average RL gains 4.5%.

**Disagreement on RL necessity**: Lightman et al. focus on inference-time verification; Math-Shepherd and Uesato et al. show RL benefits. The survey notes PRMs accelerate policy convergence via better credit assignment but warns of **reward hacking** (length/verbosity bias) more severe than for ORMs [source:arxiv:2510.08049]. **VeriGate directly addresses this** by gating PRM usage to zero-reward groups and using group-normalized advantages, showing PRM-as-ORM leads to hacking (high training PRM score, low external PRM score) while VeriGate maintains external validity.

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
| **SP-PRM (1B, BoN-16 GSM8K)** | **65.5%** | — | — | — |
| **SP-PRM (3B, BoN-16 GSM8K)** | **69.5%** | — | — | — |
| **MM-PRM (InternVL2.5-8B, BoN-16 MathVista)** | — | — | — | **63.50%** |
| **MM-PRM (InternVL2.5-78B, BoN-16 MathVista)** | — | — | — | **73.20%** |
| **ReasonFlux-PRM (SFT, MATH500)** | — | — | **84.8%** | AIME24 40.0%, GPQA-Dia 45.2% |
| **ReasonFlux-PRM (GRPO, MATH500)** | — | — | **94.8%** | — |
| **GM-PRM (MiniCPM-V2.6-8B, Refined-BoN avg)** | — | — | — | **+5.9%** (WeMath +12.4%) |
| **VeriGate (7B, AIME)** | — | — | — | **10.00%** (vs 6.67% GRPO) |

**Critical nuance from Uesato et al.**: On GSM8K, ORM and PRM achieved **similar final-answer error rates** (~22-23%), but PRM reduced trace error (11.4% vs 19.8% for outcome-only RL) [source:arxiv:2211.14275]. They found ORM predictions **agreed more with PRM labels than with outcome labels** — in math, incorrect steps rarely yield correct answers, so ORM implicitly learns process supervision [source:arxiv:2211.14275]. This may not hold in domains where "undesirable behaviors" help achieve high-rated outcomes [source:arxiv:2211.14275].

**Data efficiency**: Math-Shepherd reports PRM outperforms ORM by ~4% with only 10k training instances [source:arxiv:2312.08935]. OmegaPRM generated 1.5M annotations at 75× efficiency vs brute-force MC [source:arxiv:2406.06592]. **SP-PRM achieves strong results without new annotations** (distills from ORM). **ReasonFlux-PRM achieves SOTA with only 1k curated samples** (vs 59k raw). **GM-PRM uses only ~20k samples** for strong multimodal gains.

## Limitations and Open Challenges

1. **Resource efficiency**: PRMs require step-wise labels — significantly more expensive than ORMs [source:arxiv:2510.08049].
2. **Length/verbosity hacking**: PRMs more susceptible than ORMs to rewarding longer, verbose steps [source:arxiv:2510.08049].
3. **Cross-domain generalization**: Step definitions vary (math vs code vs agentic); PRMs show limited transfer [source:arxiv:2510.08049].
4. **Annotation noise**: Automated labels contain false positives/negatives; impact not fully quantified [source:arxiv:2312.08935][source:arxiv:2406.06592].
5. **Granularity tension**: Rigid segmentation (e.g., by newline) conflicts with semantic reasoning flow [source:arxiv:2510.08049].
6. **Proxy-reward gap**: High static benchmark accuracy ≠ effective dynamic search navigation [source:arxiv:2510.08049].
7. **Dependence on golden answers**: Automated methods (Math-Shepherd, OmegaPRM) require reference answers, limiting open-ended use [source:arxiv:2406.06592].
8. **Human ceiling**: Annotators cannot reliably label steps beyond their own reasoning ability [source:arxiv:2510.08049].
9. **ORM granularity mismatch**: Using ORMs directly for process rewards causes inconsistent scoring (high final score, low prefix scores) leading to myopic decoding [source:arxiv:2506.12446].
10. **Trajectory-format mismatch**: Standard PRMs trained on polished CoT fail on raw thinking trajectories with branching/backtracking [source:arxiv:2605.30451].
11. **Multimodal complexity**: Visual perception errors cascade; PRMs need image-alignment supervision (GM-PRM's Image Alignment dimension) [source:arxiv:2508.04088].
12. **PRM dependency in RL**: VeriGate still relies on PRM quality; systematic PRM biases influence updates on degenerate prompts [source:arxiv:2605.30451].
13. **Segmentation sensitivity**: Credit assignment (FCTR, step rewards) depends on model's step segmentation; poor segmentation weakens supervision [source:arxiv:2605.30451].
14. **Domain scope**: Most PRM work limited to math; transfer to code, tool use, long-horizon planning untested [source:arxiv:2605.30451][source:arxiv:2505.13427].
15. **Model scale**: SP-PRM (1B-8B), MM-PRM (8B), ReasonFlux-PRM (7B), GM-PRM (7B), VeriGate (1.5B/7B) — findings may not transfer to larger models [source:arxiv:2506.12446][source:arxiv:2505.13427][source:arxiv:2605.30451][source:arxiv:2508.04088][source:arxiv:2605.30451].
16. **Inference speed**: RGS methods (SP-PRM) and Refined-BoN (GM-PRM) require further optimization for generation speed [source:arxiv:2506.12446][source:arxiv:2508.04088].

## Current Status and Trajectory

PRMs are **rising rapidly** as the dominant paradigm for mathematical and code reasoning, but **not yet default** for general alignment. Key evidence:
- **Rising**: OmegaPRM (2024) demonstrates scalable automation; major labs (OpenAI o1, Google) deploy PRM-like verifiers; survey (2024) catalogs explosion of architectures and benchmarks (ProcessBench, PRMBench) [source:arxiv:2406.06592][source:arxiv:2510.08049].
- **Not default**: Most open RLHF pipelines (e.g., standard PPO/DPO) still use ORMs or scalar reward models; PRM integration requires step-wise tokenization, specialized loss, and inference-time search infrastructure not yet standardized [source:arxiv:2510.08049].
- **Fading?** No — but the **human-annotation paradigm** (PRM800K-style) is likely fading in favor of automated/hybrid pipelines due to cost and ceiling effects [source:arxiv:2510.08049].
- **Hedge**: The Uesato et al. finding that ORMs emulate PRMs *in mathematics* suggests PRM gains may be domain-specific; not widely reported whether this holds for coding, agentic tasks, or creative reasoning [source:arxiv:2211.14275].
- **New frontier**: **ORM-to-PRM distillation** (SP-PRM) offers a low-cost path to process supervision [source:arxiv:2506.12446]. **Multimodal PRMs** (MM-PRM, GM-PRM) are emerging for visual reasoning [source:arxiv:2505.13427][source:arxiv:2508.04088]. **Trajectory-aware PRMs** (ReasonFlux-PRM) address the "thinking trajectory" format of new reasoning models [source:arxiv:2605.30451]. **Verifier-gated RL** (VeriGate) mitigates reward hacking in GRPO [source:arxiv:2605.30451]. **Generative PRMs** (GM-PRM) shift from passive scoring to active correction [source:arxiv:2508.04088].

## Key Takeaways

- **PRMs provide granular credit assignment** via per-step rewards $r_t$, enabling detection of trace errors that ORMs miss [source:arxiv:2211.14275][source:arxiv:2305.20050].
- **Automated annotation (MC completion, MCTS) has largely replaced human labeling** for scale, despite label noise [source:arxiv:2312.08935][source:arxiv:2406.06592].
- **OmegaPRM's binary-search MCTS** reduces annotation cost 75× vs brute-force MC; pointwise soft labels work best [source:arxiv:2406.06592].
- **In math, ORMs implicitly learn process supervision** because wrong steps rarely yield right answers — PRM advantage may not transfer to other domains [source:arxiv:2211.14275].
- **PRMs excel at test-time scaling** (Best-of-N, MCTS, beam annealing) and improve RL convergence, but are more prone to length hacking and generalization gaps [source:arxiv:2510.08049][source:arxiv:2312.08935].
- **No standardized PRM training/inference stack exists yet** — architectures (discriminative, generative, implicit, trajectory-aware, ORM-distilled), losses (BCE, MSE, pairwise, soft, weighted Bradley-Terry), and aggregation (product, sum, weighted, step+trajectory) remain active design choices [source:arxiv:2510.08049].
- **SP-PRM enables PRM learning from ORMs alone** via dual-consistency (score + preference), resolving granularity mismatch for inference-time alignment [source:arxiv:2506.12446].
- **Multimodal PRMs require specialized pipelines**: MM-PRM uses MCTS + soft labels on restructured CoT; GM-PRM uses GPT-4o critique generation + MC filtering + Refined-BoN active correction [source:arxiv:2505.13427][source:arxiv:2508.04088].
- **Reasoning models need trajectory-aware PRMs**: Standard PRMs fail on raw thinking trajectories; ReasonFlux-PRM's step (alignment/quality/coherence) + trajectory (template-guided) rewards achieve SOTA with 1k samples [source:arxiv:2605.30451].
- **VeriGate fixes PRM reward hacking in GRPO** by gating process supervision to zero-reward groups and using future-cumulated token rewards with group-normalized advantages [source:arxiv:2605.30451].

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
- [GRPO (Group Relative Policy Optimization)](grpo.md) — VeriGate's verifier-gated GRPO extension
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md) — SP-PRM's Bradley-Terry loss connection
- [LLM-as-judge](llm-as-judge.md) — ReasonFlux-PRM's GPT-4o quality scoring, GM-PRM's GPT-4o critique generation
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md) — PRM generalization gap to agentic tasks
- [Length and format bias](length-and-format-bias.md) — PRM susceptibility to verbosity hacking
- [Reward model over-optimization](reward-model-overoptimization.md) — VeriGate's mitigation of PRM over-optimization

## References
- [source:arxiv:2211.14275] [Outcome-supervised Verifiers for Planning in Complex Reasoning Tasks](https://arxiv.org/abs/2211.14275)
- [source:arxiv:2312.08935] [Math-Shepherd: A Label-free Step-by-Step Verifier for Mathematical Reasoning (Wang et al., 2023)](https://arxiv.org/abs/2312.08935)
- [source:arxiv:2510.08049] [A Survey of Process Reward Models: From Outcome to Process Supervision](https://arxiv.org/html/2510.08049v3)
- [source:arxiv:2305.20050] [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)
- [source:arxiv:2406.06592] [Improve Mathematical Reasoning in Language Models by Automated Process Supervision (OmegaPRM)](https://arxiv.org/abs/2406.06592)
- [source:arxiv:2506.12446] [From Outcomes to Processes: Guiding PRM Learning from ORM for Inference-Time Alignment](https://arxiv.org/abs/2506.12446)
- [source:arxiv:2505.13427] [MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision](https://arxiv.org/abs/2505.13427)
- [source:arxiv:2506.18896] [ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2506.18896)
- [source:arxiv:2605.30451] [VeriGate: Verifier-Gated Step-Level Supervision for GRPO](https://arxiv.org/abs/2605.30451)
- [source:arxiv:2508.04088] [GM-PRM: A Generative Multimodal Process Reward Model for Multimodal Mathematical Reasoning](https://arxiv.org/abs/2508.04088)
