---
title: Process vs outcome reward models
kind: primer
reference: ../topics/process-vs-outcome-rewards.md
updated: '2026-07-12'
---

# Process vs Outcome Reward Models: A Primer

**What this is and what you'll learn.** Outcome Reward Models (ORMs) score only the final answer; Process Reward Models (PRMs) score every reasoning-step prefix. This primer teaches you why that distinction reshapes credit assignment, data collection, and the ability to catch *trace errors* (correct final answer via flawed reasoning). By the end you'll understand the core multiplicative scoring mechanism, how modern pipelines automate step labels without human annotators, why ORMs can implicitly mimic PRMs in mathematics (but not necessarily elsewhere), and where the current architectural fault lines lie — ORM distillation, multimodal specialization, trajectory-aware supervision, and verifier-gated RL.

---

## The Core Mechanism: Multiplicative Credit Assignment

An ORM $R_{\text{ORM}}(x, y)$ maps a full solution $y$ to a scalar reflecting final-answer correctness. A PRM $R_{\text{PRM}}(x, y_{1:t})$ instead scores a *prefix* $y_{1:t}$ of a step-by-step solution. The overall solution score aggregates these prefix scores **multiplicatively** (or additively in log-space):

$$
\text{Score}(y) = \prod_{t=1}^{T} R_{\text{PRM}}(x, y_{1:t})
\qquad\text{or}\qquad
\sum_{t=1}^{T} \log R_{\text{PRM}}(x, y_{1:t})
$$

**Why multiplicative?** Each step must be correct for the chain to be trustworthy. A single flawed step ($r_t \approx 0$) collapses the product, flagging a trace error that an ORM would miss because the final answer happens to be right. The discriminative PRM formalizes $r_t = \sigma(f_\theta(x, s_{1:t})) \in (0,1)$ — a learned head on the prefix representation.

**Intuition check:** If a model writes three correct steps ($r=0.9$ each) and one hallucinated step ($r=0.1$), the product is $0.9^3 \times 0.1 \approx 0.07$. The ORM sees only the final answer (possibly correct) and rewards it; the PRM penalizes the chain. This is the entire game: *dense credit assignment that exposes intermediate failures.*

```python
# Runnable check: multiplicative PRM scoring catches trace errors ORMs miss
import math

def prm_score(step_rewards):
    """Product of per-step rewards (all in (0,1])."""
    prod = 1.0
    for r in step_rewards:
        prod *= r
    return prod

# Scenario: 3 solid steps, 1 hallucinated step -> correct final answer by luck
trace_error_rewards = [0.9, 0.9, 0.9, 0.1]   # PRM sees the bad step
clean_rewards       = [0.9, 0.9, 0.9, 0.9]   # fully correct chain

orm_score = 1.0  # ORM only sees final answer correctness (assumed correct here)

assert prm_score(trace_error_rewards) < 0.1, "PRM should heavily penalize trace error"
assert prm_score(clean_rewards) > 0.6,       "PRM should reward clean chain"
assert orm_score == 1.0,                     "ORM cannot distinguish these two cases"
print("PRM product:", prm_score(trace_error_rewards), "vs clean:", prm_score(clean_rewards))
print("ORM score (both):", orm_score)
# Output: PRM product: 0.0729 vs clean: 0.6561 | ORM score (both): 1.0
```

---

## Training Paradigms: From Human Labels to ORM Distillation

| Paradigm | Mechanism | Key Example | Scale | Cost |
|----------|-----------|-------------|-------|------|
| Human annotation | Experts label each step correct/incorrect/neutral | PRM800K (75k solutions, 800k step labels) | Medium | Very high |
| Monte Carlo completion | Sample $N$ rollouts from step $t$; label by fraction reaching $a^*$ | Math-Shepherd HE/SE | High | High (decoding) |
| MCTS + binary search | Tree search + binary search to locate first error | OmegaPRM | Very high | Moderate (75× vs brute force) |
| **ORM-guided distillation** | **Dual-consistency training (score + preference) from existing ORM** | **SP-PRM** | **High** | **Low (no new human labels)** |
| **Multimodal MCTS + soft labels** | **MCTS rollouts with binary search + continuous MC scores as targets** | **MM-PRM** | **Very high (700k+ steps)** | **Moderate** |
| **Trajectory-aware supervision** | **Step-level (alignment/quality/coherence) + trajectory-level (template-guided) rewards** | **ReasonFlux-PRM** | **Medium (1k–59k)** | **Moderate (GPT-4o judge)** |
| **Generative critique + MC filtering** | **GPT-4o analyzes steps (intent, alignment, logic, refinement); MC hard estimation filters** | **GM-PRM** | **Medium (~20k)** | **Moderate** |

**The arc:** Human labels (PRM800K) are high-fidelity but expensive and hit a "human ceiling." Automated Monte Carlo (Math-Shepherd) trades noise for scale: **Hard Estimation (HE)** = indicator that *some* rollout reaches $a^*$; **Soft Estimation (SE)** = fraction of rollouts reaching $a^*$. OmegaPRM cuts cost 75× by framing error localization as binary search on the reasoning chain ($O(k \log M)$ vs $O(kM)$) and uses a heuristic $Q(s,r) = \alpha^{1-\text{MC}(s)} \beta^{\text{len}(r)/L}$ to prioritize informative rollouts.

**The pivot:** SP-PRM shows you can **derive a PRM from an existing ORM without any new step labels**. It constructs partial sequences via token-level or stochastic truncation, then optimizes **Score Consistency** (partial scores align with final ORM score) and **Preference Consistency** (partial evaluations match human preferences). A reference ORM filters and weights samples by inverse entropy $w_t = 1/H_t$, resolving the *granularity mismatch* that causes myopic decoding when ORMs are used directly for process rewards.

**Multimodal & trajectory frontiers:** MM-PRM builds a policy (InternVL2.5-8B on 5.1M restructured CoT), runs MCTS+binary search on 10k seeds, and trains on **soft labels** $\hat{y}_t = \text{MC}(x_{<t}) \in [0,1]$ — beating hard labels 43% vs 34.4% on MM-K12. ReasonFlux-PRM addresses the *raw thinking trajectories* of models like DeepSeek-R1 (branching, backtracking, unstructured) with **two reward levels**: step-level (alignment + quality + coherence) and trajectory-level (template-guided generation, average correctness). GM-PRM goes *generative*: GPT-4o produces critiques across four dimensions (intent, image alignment, logic, refinement), filtered by MC hard estimation, then SFT trains the model to output critiques and judgments per step.

---

## Architecture Families (and the New Entrants)

The survey categorizes PRMs into four families; recent work adds four more:

1. **Discriminative** — direct $r_t = \sigma(f_\theta(x, s_{1:t}))$; losses: pointwise BCE, MSE, pairwise Bradley-Terry.
2. **Generative** — "think-then-judge": generate critique $z_t$, then score $r_t = h_\psi(x, s_{1:t}, z_t)$; joint SFT + BCE loss.
3. **Implicit** — infer step rewards from outcome-only signals (credit assignment algorithms).
4. **Other** — graph-based, retrieval-augmented, hierarchical.

**New families from 2024–2025:**

5. **ORM-Distilled (SP-PRM)** — weighted Bradley-Terry on partial sequences:

$$
\mathcal{L} = -\mathbb{E} \, w \log \sigma\bigl(r_\theta(x, y_{<t}^w) - r_\theta(x, y_{<t}^l)\bigr),
   \quad w = 1/H_t \text{ if ORM prefers } y_{<t}^w \text{ else } 0
$$

6. **Multimodal Discriminative (MM-PRM)** — special `<prm>` marker token after each step; softmax over "Yes"/"No" logits with **soft labels** $\hat{y}^{(i)} = \text{MC}(x_{<t})$.
7. **Trajectory-Aware (ReasonFlux-PRM)** — joint MSE on step-level (alignment/quality/coherence) and trajectory-level (template-guided) rewards:

$$
\mathcal{L} = \sum \lambda_{\text{step}} (R_\phi(s_t) - r_t^{\text{step}})^2 + \lambda_{\text{final}} (R_\phi(y) - r^{\text{final}})^2
$$

8. **Generative Multimodal (GM-PRM)** — SFT to emit $(c_1, j_1, \dots, c_T, j_T)$: critique + judgment per step.

**Label flavor matters:** OmegaPRM found **pointwise soft labels** ($\hat{y} = \text{MC}(s)$) beat hard labels and pairwise objectives (70.1% per-step accuracy). MM-PRM confirms this in multimodal (43% vs 34.4%). ReasonFlux-PRM uses **MSE on continuous rewards** (not classification), enabling fine-grained credit for messy trajectory steps.

---

## Inference-Time Scaling: Verification and Search

PRMs turn test-time compute into accuracy via:

| Strategy | Mechanism |
|----------|-----------|
| Best-of-N / Weighted Majority Voting | Score each candidate by $\prod_t r_t$ (or weighted sum with self-consistency) |
| Verification-guided decoding | At each step, sample $k$ continuations, score with PRM, keep top-$b$ (beam annealing) |
| MCTS / Guided Search | PRM as state-value heuristic in tree search |
| **Refined-BoN (GM-PRM)** | **Active refinement: PRM critiques errors, policy regenerates from corrected step; iterate** |
| **Trajectory-aware BoN (ReasonFlux-PRM)** | **Aggregate step + trajectory rewards: $\hat{r} = \frac{1}{T}\sum \hat{r}_t^{\text{step}} + \alpha \hat{r}^{\text{final}}$** |

**Key results:** Lightman et al. (Best-of-1860, product-of-probs) reached 78.2% on MATH vs ORM 72.4%. OmegaPRM + weighted majority voting @64 lifted Gemini Pro 51% → 69.4% and Gemma2-27B 42.3% → 58.2% on MATH500. SP-PRM BoN-16: GSM8K 65.5% (1B), 69.5% (3B). MM-PRM BoN-16: InternVL2.5-78B MathVista 69.48% → 73.20%. ReasonFlux-PRM SFT on 1k curated samples beat 59k raw trajectories on MATH500 (84.8% vs 78.8%); GRPO pushed to 94.8%. GM-PRM Refined-BoN: average +5.9% across models (WeMath +12.4%).

---

## RL Integration: Dense Rewards and the Hacking Fix

PRMs provide **dense, step-level rewards** for PPO/GRPO. Math-Shepherd applied step-by-step PPO: $r_t = R_{\text{PRM}}(x, y_{1:t})$, improving Mistral-7B GSM8K 77.9% → 84.1%, MATH 28.6% → 33.0%.

**VeriGate** introduces a **verifier-gated extension of GRPO** that activates process supervision *only when outcome rewards are uninformative*:
- **S1 Gating**: Mixed group rewards → standard GRPO; all zero → activate PRM token-level supervision; all correct → no PRM.
- **S2 Future-Cumulated Token Rewards (FCTR)**: Token in step $j$ gets $c_{i,j} = \sum_{k=j}^{S_i} r_{i,k}$ (sum of subsequent PRM step rewards).
- **S3 Group-Normalized Token Advantages**: $\bar{c} = \frac{\sum c_{i,j}}{\sum S_i}$, $A_{i,j} = (c_{i,j} - \bar{c}) / \sigma(c)$.
- **Effective Advantage**: $\widetilde{A}_{i,j} = A_i^{\text{GRPO}}$ if mixed rewards, else $A_{i,j}$ above.

**Result:** Qwen2.5-Instruct 7B on MATH: ~12% accuracy gain (AIME 6.67% → 10.00%, AMC 42.17% → 45.78%). **Reward hacking mitigation:** PRM-as-ORM (using PRM score as outcome reward) hacked badly (external PRM score 0.0331); VeriGate maintained external validity (0.1424). ReasonFlux-PRM for online GRPO uses composite reward $\boldsymbol{r}_{\text{new}} = (1-\beta) r_{\text{out}} + \beta \hat{r}$; DeepSeek-R1-Distill-Qwen-7B MATH500 89.6% → 94.8%.

---

## The Load-Bearing Disagreements (Two You Must Know)

### 1. **In mathematics, ORMs implicitly learn process supervision** (Uesato et al.)
On GSM8K, ORM and PRM achieved **similar final-answer error rates** (~22–23%), but process-based SFT reduced trace error (11.4% vs 19.8% for outcome-only RL). Crucially, **ORM predictions agreed more with PRM labels than with outcome labels** — because in math, incorrect steps rarely yield correct answers. *This means the PRM advantage may be domain-specific*; it may not transfer to coding, agentic tasks, or creative reasoning where "undesirable behaviors" can help achieve high-rated outcomes.

### 2. **PRM-as-ORM causes severe reward hacking; gating fixes it** (VeriGate)
Using a PRM directly as an outcome reward (PRM-as-ORM) produces high training PRM scores but *low external PRM scores* (0.0331) — the model learns to game the step-scorer. VeriGate's **verifier gating** (activate PRM only on zero-reward groups) + **group-normalized token advantages** restores external validity (0.1424) and cuts zero-verifier-reward prompts faster. This is a concrete architectural fix for a failure mode that appears as soon as you try to use PRMs in RL loops.

---

## Current Status
**Rising rapidly** for math/code reasoning (OmegaPRM automation, o1-style verifiers, ProcessBench/PRMBench), but **not yet default** for general alignment — open RLHF pipelines still favor ORMs/scalar RMs; PRM integration requires step-wise tokenization, specialized losses, and search infrastructure not yet standardized. The human-annotation paradigm is fading; ORM distillation, multimodal pipelines, trajectory-aware supervision, and verifier-gated RL are the active frontiers.

**Full reference:** See the comprehensive fact-checked article "Process vs Outcome Reward Models" (sources: arxiv:2211.14275, 2305.20050, 2312.08935, 2406.06592, 2505.13427, 2506.12446, 2506.18896, 2508.04088, 2510.08049, 2605.30451, 2605.30452).

---
*Full reference (citations, derivations, variants):* [Process vs outcome reward models](../topics/process-vs-outcome-rewards.md)
