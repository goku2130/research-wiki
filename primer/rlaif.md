---
title: RLAIF (RL from AI feedback)
kind: primer
reference: ../topics/rlaif.md
updated: '2026-07-12'
---

# RLAIF: Reinforcement Learning from AI Feedback

**Scaffold.** RLAIF replaces human preference annotations with AI-generated feedback, making scalable alignment economically viable. By the end of this primer you will understand the two canonical architectures—Constitutional AI (CAI) and Self-Rewarding Language Models (SRLM)—how they convert language model outputs into training signals, why soft labels and calibration matter, and where the load-bearing disagreements live (separate judge vs. unified judge; scalar reward vs. pairwise preference). This connects directly to PPO/DPO pipelines, reward modeling, and the broader Recursive Alignment Loop formalism.

---

## Core Mechanism: From Principles to Preference Signals

All RLAIF methods share a three-step loop: **generate → evaluate → update**. The evaluation step is where human labels are replaced.

### Constitutional AI (CAI): Explicit Principles, Separate Judge

CAI (Bai et al., 2022) operationalizes this in two phases.

**Phase 1 — SL-CAI (Supervised Critique-Revision).**  
Start from a helpful-only RLHF model. For each red-team prompt (182K total), sample a response, then iterate $N=4$ times: pick a constitutional principle (e.g., "Choose the response a wise, ethical, polite person would say"), **critique** the response for violations, then **revise** to satisfy the principle. Fine-tune a fresh pretrained model on the final revisions (731K examples) mixed with helpfulness data. Revisions monotonically improve harmlessness; principle diversity beats principle count.

**Phase 2 — RL-CAI (RL from AI Feedback).**  
The SL-CAI model generates two responses per prompt. A *separate* feedback model (a pretrained LM) receives a multiple-choice prompt containing the pair and a principle. Its normalized log-probabilities yield a **soft preference label** $p(A \succ B)$:

$$
p(A \succ B) = \frac{\exp(\log P(A))}{\exp(\log P(A)) + \exp(\log P(B))}
$$

Example: $\log P(A)=-0.3,\; \log P(B)=-1.6 \;\rightarrow\; p(A \succ B)=0.79$. Labels are ensembled over multiple principles. Chain-of-thought (CoT) prompting improves discrimination but produces overconfident probabilities; these are **clamped to $[0.4, 0.6]$** before preference model (PM) training to avoid extreme RL targets. A PM is trained on mixed AI harmlessness + human helpfulness labels, then the SL-CAI model is fine-tuned via PPO:

$$
\max_\theta \; \mathbb{E}[r(x,y)] - \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]
$$

where $\pi_{\text{ref}}$ is the SL-CAI model and $r$ comes from the AI-trained PM. Result: Pareto improvement over HH-RLHF—higher harmlessness at matched helpfulness, zero evasiveness.

### Self-Rewarding Language Models (SRLM): Unified Policy-as-Judge

SRLM (Yuan et al., 2024) collapses policy and reward model into **one model** that iteratively self-improves via DPO. At iteration $t$:

1. **Generate** $k$ candidates $y^{(j)} \sim \pi_t(\cdot|x)$.
2. **Self-evaluate**: $\pi_t$ scores each candidate (LLM-as-a-Judge with CoT, 5 criteria, score $\in[0,5]$).
3. **Construct pairs**: highest score = winner $y_w$, lowest = loser $y_l$; discard ties.
4. **Update**: $\pi_{t+1} = \text{DPO}(\pi_t, \text{pairs})$.

No explicit constitution—the model's own scoring is the feedback function $F$ in the Recursive Alignment Loop $(\pi_0, G, F, T)$. On Llama 2 70B: AlpacaEval 2.0 win rate vs GPT-4 Turbo rises from 9.9% ($M_1$) → 15.4% ($M_2$) → **20.4%** ($M_3$), surpassing Claude 2 (17.2%) and GPT-4 0613 (15.8%). Reward modeling accuracy climbs 65.1% → 81.7%. **Caveat**: response length grows 1092 → 2552 tokens (length bias); minimal gains in math/logic; spontaneous reward hacking observed.

---

## Why Soft Labels and Calibration Matter

CAI demonstrates that **soft labels (probabilities) outperform hard labels (argmax)** when CoT is not used—the feedback model's log-probabilities preserve preference margins that binary choices discard. DAR (2025) goes further: **AI reward (scalar per response) consistently beats AI preference (pairwise)** in human-AI agreement across annotators and tasks (e.g., Llama-3.1-405B: 79.3% vs 72.8% on TL;DR). Scalar rewards preserve equivalences and margins; online RLHF from AI reward needs **3–5× fewer annotations** than from AI preference.

**Calibration is not emergent—it's engineered.** CAI clamps CoT probabilities to $[0.4, 0.6]$. DAR normalizes advantages and clips importance weights ($w_{\text{clip}}=20$). SER filters pairs by predicted probability thresholds ($\tau_{\text{high}}=0.55, \tau_{\text{low}}=0.45, \tau_\Delta=0.3$). All are practical interventions, not model properties.

---

## Runnable Check: Soft Preference from Log-Probabilities

```python
import math

def soft_preference(logp_a: float, logp_b: float) -> float:
    """Convert normalized log-probs to P(A > B) via softmax."""
    # Numerical stability: subtract max
    m = max(logp_a, logp_b)
    exp_a = math.exp(logp_a - m)
    exp_b = math.exp(logp_b - m)
    return exp_a / (exp_a + exp_b)

# CAI example from reference: log P(A)=-0.3, log P(B)=-1.6 -> 0.79
p = soft_preference(-0.3, -1.6)
assert abs(p - 0.79) < 0.01, f"Expected ~0.79, got {p:.3f}"

# Clamping check: CoT probabilities clamped to [0.4, 0.6]
def clamp(p: float, lo: float = 0.4, hi: float = 0.6) -> float:
    return max(lo, min(hi, p))

assert clamp(0.95) == 0.6
assert clamp(0.05) == 0.4
assert clamp(0.5) == 0.5

print("All checks passed.")
```

---

## Load-Bearing Disagreements

1. **Separate judge (CAI) vs. unified judge (SRLM).**  
   CAI argues a separate feedback model improves critique quality; SRLM argues unification enables fully autonomous loops. The Rahman survey notes SPIN-style offline DPO bottlenecks once policy matches human data quality, whereas CAI and DAR continue improving via online AI feedback—this is unsettled without matched compute/data comparisons.

2. **AI reward (scalar) vs. AI preference (pairwise).**  
   DAR shows scalar rewards achieve higher human agreement and 3–5× annotation efficiency. CAI and SRLM rely on pairwise comparisons (soft or hard). CAI's soft labels partially recover margin information, but the scalar-vs-pairwise gap remains a live design choice.

**Additional tension:** Lee et al. (2023) find **few-shot prompting decreases alignment** for summarization/helpfulness (contrary to ICL wisdom) and **self-consistency (multiple CoT samples) degrades labeler alignment**—while CAI uses single CoT with clamping and principle ensembling without few-shot. Wang et al. (2024) show **15% human seed data suffices** for near-full performance ($0.3\%$ avg gap), implying human labels are far more redundant than assumed.

---

## Current Status
RLAIF is a default pillar in frontier alignment stacks (hybrid human/AI/verifiable reward pipelines); open challenges remain in reward hacking amplification, distributional collapse, positional bias in LLM judges, and reasoning gaps in self-rewarding models.

## Full Reference
See the comprehensive reference article for all sources, equations, and empirical tables.

---
*Full reference (citations, derivations, variants):* [RLAIF (RL from AI feedback)](../topics/rlaif.md)
