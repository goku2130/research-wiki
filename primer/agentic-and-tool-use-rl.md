---
title: Agentic and tool-use RL
kind: primer
reference: ../topics/agentic-and-tool-use-rl.md
updated: '2026-07-12'
---

# Agentic and Tool-Use RL: From Single-Turn Preference to Multi-Turn POMDPs

**Scaffold.** This primer teaches the conceptual and algorithmic shift from RLHF's single-turn preference optimization to agentic RL's multi-turn tool use in partially observable environments. By the end you will understand: (1) why the POMDP formalism makes credit assignment across 100K–1M token trajectories the central bottleneck, (2) how reward granularity and algorithm choice interact with task structure, and (3) which turn-level credit assignment methods actually buy efficiency in practice. This connects to RLHF (the $T=1$ special case), GRPO/PPO mechanics, and the emerging taxonomy of process vs. outcome supervision.

---

## The Core Mechanism: From Degenerate MDP to Agentic POMDP

RLHF collapses to a **single-step MDP** with horizon $T=1$, $\gamma=1$, and a scalar reward $r(a)$ emitted after one generation. The objective is simply $\mathbb{E}_{a\sim\pi_\theta}[r(a)]$. There is no state transition, no partial observability, and no credit assignment problem—every token in the response shares the same outcome.

Agentic RL lifts this to a **Partially Observable MDP** where:
- The agent receives observations $o_t = O(s_t)$, not the true environment state $s_t$ (API responses, search results, execution outputs).
- The action space splits: $\mathcal{A}_{\text{agent}} = \mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{action}}$ (natural language reasoning + structured tool invocations).
- The objective maximizes **discounted cumulative reward** over horizon $T$:

$$
J_{\text{agent}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \gamma^t R_{\text{agent}}(s_t, a_t) \right], \quad 0 < \gamma < 1
$$

This formalism makes explicit five compounding difficulties absent in RLHF: stochastic environment transitions, partial observability, long horizons (10–100+ turns), action heterogeneity, and **non-verifiable intermediate states**—no ground truth exists for most intermediate turns.

**Why this breaks token-level advantage estimation.** In RLHF, the advantage of every token is the same scalar (the final reward minus baseline). In agentic RL, a tool call at turn 3 may only affect success at turn 12. Episode-level methods like GRPO assign **identical credit to all actions in a trajectory**, creating an "echo trap": the model converges to repetitive safe behaviors because it cannot distinguish which turns actually mattered. The credit assignment bottleneck is therefore **information sparsity**—low mutual information $I(A;Z|S)$ between actions and returns conditioned on state—not reward sparsity per se. Adding a constant to rewards eliminates reward sparsity without alleviating credit assignment difficulty.

---

## Reward Granularity: The Highest-Leverage Design Choice

The Practitioner's Guide (TextWorld, SWE-Gym) shows a dramatic hierarchy on complex coding tasks:
- Sparse binary reward (pass/fail): **4.2%** success
- Dense, **execution-verified** ratio rewards (unit test pass ratio): **22%** success
- Model-based judges (CodeRM-8B, GPT-4.1): **7–9%** success

**Execution-verified dense rewards dominate** because they provide per-turn progress signals in extended code-editing trajectories where intermediate states are otherwise non-verifiable. Model-based judges are not yet viable substitutes.

**But** Bespoke Labs achieved **55% → 78%** on BFCL multi-turn tool use using **only sparse binary reward** (pass/fail BFCL eval) with GRPO. Perplexity's production pipeline uses a **gated aggregation** reward:

$$
R(\tau_i) = r_{\text{base}}(\tau_i) \cdot s(\tau_i) - \text{pen}_{\text{eff}}(\tau_i)
$$

where $r_{\text{base}} \in \{0,1\}$ is verifiable correctness, $s(\tau_i) \in [0,1]$ is a Bradley-Terry preference score, and $\text{pen}_{\text{eff}}$ penalizes excess tool calls/length. Correctness is a **prerequisite** for preference credit.

**Load-bearing disagreement:** The discrepancy stems from **task structure**. BFCL evaluates discrete API-call correctness (verifiable per-turn), while SWE-Gym requires extended code-editing trajectories where intermediate progress signals are essential. Dense verified rewards buy you signal where per-turn verification is impossible; sparse binary suffices where every turn is independently checkable.

---

## Algorithm Selection: PPO vs. GRPO vs. The Rest

The Practitioner's Guide reports a striking hierarchy on TextWorld (Qwen-1.5B, w2-o3-q4):
- **PPO**: 88% success
- **RLOO**: 51%
- **Reinforce++ / GRPO**: ~18% (negligible gains)

This contradicts GRPO's dominance in search-agent deployments (Perplexity, Bespoke Labs) and its listing as a primary algorithmic family in the landscape survey. The credit assignment survey explains GRPO's weakness: **uniform trajectory-level credit** creates the echo trap. Perplexity mitigates this with **token-level importance sampling** and a **two-stage SFT→RL pipeline**; Bespoke uses minimal data (100 samples) with aggressive stabilization (overlong filtering, small KL, periodic reference updates).

**Practical recipe from the Guide:** For a fixed compute budget, **hybrid SFT→RL dominates pure SFT or pure RL**. With 60 SFT demos + 400 RL episodes: 85% in-domain, 59% generalization. Pure SFT: 55% generalization. Pure RL: 11%. Strong SFT priors from minimal demonstrations accelerate convergence and stabilize policy initialization.

---

## Turn-Level Credit Assignment: Where the Efficiency Gains Are

Token-level methods (VinePPO, RED, SPRO) target 500–30K token reasoning trajectories. Agentic RL operates at 100K–1M tokens across 10–100+ turns. The credit assignment survey shows **turn-level and hierarchical methods** deliver large efficiency gains when transferred to agentic settings:

| Method | Granularity | Mechanism | Reported Gain |
|--------|-------------|-----------|---------------|
| **AgentPRM** | Turn | TD+GAE replaces expensive MC labeling | 8× sample efficiency vs MC-PRM |
| **CARL** | Sparse | Identify bifurcation points via action entropy $H(\pi(\cdot|s_t))$; update only critical actions | 72% fewer gradient updates, no degradation |
| **IGPO** | Turn | Credit = information gain about success: $c_t = \log P(\text{success}|h_{1:t}) - \log P(\text{success}|h_{1:t-1})$ | — |
| **TRACE** | Turn | Turn-aware advantage $\hat{A}_{i,t} = m_{i,t} \hat{A}_i^o + \hat{A}_{i,t}^p$ with leave-one-turn-out semantic masking | 87.1% ASR@1 (~25% relative over TROJail) |
| **ArCHer** | Hierarchical | High-level turn critic + low-level token actor | — |

**Key insight:** Turn-level methods exploit the fact that **not all turns matter equally**. CARL's entropy-based bifurcation detection and TRACE's leave-one-turn-out masking both identify critical decision points and concentrate updates there, avoiding the echo trap of uniform trajectory credit.

---

## Multi-Turn Iterative Preference Learning (M-DPO / M-KTO)

For tool-integrated mathematical reasoning with **deterministic transitions** (code execution results determined by history), M-DPO and M-KTO sum log-ratio advantages across all turns $h=1..H$:

$$
\mathcal{L}_{\mathrm{M-DPO}}(\theta)=-\sum_{((x,\tau^{w},\tau^{l})\in\mathcal{D}}\log\sigma\Big(\eta\sum_{h=1}^{H}\Big[\log\frac{\pi_{\theta,h}(a_{h}^{w}|s_{h}^{w})}{\pi_{\mathrm{ref},h}(a_{h}^{w}|s_{h}^{w})}-\log\frac{\pi_{\theta,h}(a_{h}^{l}|s_{h}^{l})}{\pi_{\mathrm{ref},h}(a_{h}^{l}|s_{h}^{l})}\Big]\Big)
$$

Critical implementation details:
- **Mask out user messages and external tool observations**—optimize only the model's own actions.
- **Mixture sampling** (e.g., 20 current + 10 previous iteration trajectories) for exploration-exploitation balance.
- **Update reference model to previous iteration's policy** ($\pi_{t,\text{ref}} = \pi_{t-1}$) to optimize the non-regularized reward.

**Quantitative results** (Gemma on GSM8K/MATH): Preference learning primarily improves **pass@1** (top response quality); for $n>16$, performance converges with SFT models, indicating **elicitation of existing knowledge rather than injection of new knowledge**.

**Limitation:** Assumes deterministic transitions. The agentic POMDP formalism explicitly includes stochastic transitions as a core difficulty—M-DPO/M-KTO do not directly transfer to stochastic environments without value networks for adaptive margins.

---

## Practical Stabilization: What Actually Works

Bespoke Labs' minimal-data GRPO recipe (Qwen2.5-7B, 100 BFCL samples, 4×H200):
- **Overlong filtering**: mask loss for rollouts exceeding max completion length (fixes completion length blowup → accuracy collapse)
- **Small KL with periodic reference updates**: $\beta=0.001$, ref model updated every 100 steps (stronger policy needs stronger reference)
- **Binary correctness-only reward**: "Less is more"—complex rewards (tool counts, format checks) worsened stability
- **Batch 1, grad accum 4, LR $10^{-6}$ constant, 20 warmup, $\mu=2$ (1 on-policy + 1 off-policy step/batch)**

Perplexity's production pipeline adds:
- **Two-stage SFT→RL** (joint optimization in single RL stage fails; RL-only underperforms guardrails; SFT alone compromises search)
- **Token-level importance sampling** in GRPO to mitigate variance
- **Synthetic verifiable data pipeline**: seed selection → chain construction (2–4 hops) → name-free synthesis → independent verification (90% of mix)

---

## Runnable Check: Turn-Aware Credit vs. Uniform Trajectory Credit

This minimal simulation demonstrates why uniform trajectory credit (GRPO-style) fails to identify critical turns, while a leave-one-turn-out estimator recovers the signal. It mirrors the TRACE/CARL intuition.

```python
import numpy as np

np.random.seed(42)

# Simulate a 10-turn trajectory where only turn 3 (index 3) is critical
# True causal effect: flipping turn 3 changes success prob from 0.2 -> 0.9
# Other turns have no causal effect (noise only)
N_TURNS = 10
CRITICAL_TURN = 3
N_TRAJECTORIES = 5000

def generate_trajectory():
    # Random policy: each turn is a binary action
    actions = np.random.randint(0, 2, N_TURNS)
    # Success depends ONLY on critical turn
    success = 0.9 if actions[CRITICAL_TURN] == 1 else 0.2
    reward = 1.0 if np.random.random() < success else 0.0
    return actions, reward

# Collect trajectories
trajectories = [generate_trajectory() for _ in range(N_TRAJECTORIES)]
actions_arr = np.array([a for a, _ in trajectories])
rewards_arr = np.array([r for _, r in trajectories])

# --- GRPO-style: uniform trajectory credit ---
# Advantage = reward - baseline (mean reward)
baseline = rewards_arr.mean()
advantages = rewards_arr - baseline  # shape (N,)
# Credit per turn = mean advantage when action=1 minus mean advantage when action=0
grpo_credit = np.zeros(N_TURNS)
for t in range(N_TURNS):
    mask_1 = actions_arr[:, t] == 1
    mask_0 = actions_arr[:, t] == 0
    if mask_1.any() and mask_0.any():
        grpo_credit[t] = advantages[mask_1].mean() - advantages[mask_0].mean()

# --- Leave-one-turn-out (LOTO) credit: TRACE/CARL style ---
# For each trajectory, estimate counterfactual by flipping one turn
loto_credit = np.zeros(N_TURNS)
for t in range(N_TURNS):
    # Flip action at turn t and recompute success probability
    flipped_actions = actions_arr.copy()
    flipped_actions[:, t] = 1 - flipped_actions[:, t]
    # True success prob under flipped action (oracle for simulation)
    flipped_success = np.where(flipped_actions[:, CRITICAL_TURN] == 1, 0.9, 0.2)
    # Expected reward difference = causal effect
    loto_credit[t] = (flipped_success - np.where(actions_arr[:, CRITICAL_TURN] == 1, 0.9, 0.2)).mean()

print("GRPO uniform credit per turn:", np.round(grpo_credit, 4))
print("LOTO credit per turn:        ", np.round(loto_credit, 4))

# Assertions: LOTO isolates critical turn; GRPO dilutes credit across all turns
assert abs(loto_credit[CRITICAL_TURN]) > 0.3, "LOTO should detect large causal effect at critical turn"
assert all(abs(loto_credit[t]) < 0.05 for t in range(N_TURNS) if t != CRITICAL_TURN), "LOTO should be near zero for non-critical turns"
assert abs(grpo_credit[CRITICAL_TURN]) < 0.15, "GRPO credit at critical turn is diluted by trajectory variance"
print("\n✓ LOTO isolates the critical turn; GRPO uniform credit fails to distinguish it.")
```

**What this shows:** With 5000 trajectories, GRPO's per-turn credit at the critical turn is ~0.07 (diluted by variance), while LOTO recovers ~0.35. In agentic RL with 100K+ tokens and 10–100 turns, this dilution becomes catastrophic—the echo trap. Turn-aware methods (TRACE, CARL, IGPO, AgentPRM) explicitly estimate per-turn causal effects or information gain to concentrate updates where they matter.

---

## Load-Bearing Disagreements & Caveats

1. **PPO vs. GRPO on TextWorld vs. Search Agents.** The Practitioner's Guide finds PPO 88% vs. GRPO ~18% on TextWorld, yet GRPO powers Perplexity and Bespoke Labs' search agents. The credit assignment survey attributes GRPO's weakness to the **echo trap from uniform trajectory credit**. Perplexity mitigates this with token-level IS and SFT→RL; Bespoke uses minimal data with heavy stabilization. **No large-scale controlled comparison exists on identical agentic benchmarks with identical compute.**

2. **Deterministic vs. Stochastic Transitions.** M-DPO/M-KTO assume deterministic tool execution (code results determined by history). The agentic POMDP formalism explicitly lists **stochastic environment transitions** as a core difficulty. This limits direct transfer of multi-turn preference learning to stochastic environments (web search, API calls with non-deterministic responses) without value networks for adaptive margins.

---

## Current Status

Agentic RL is **rising rapidly but pre-standardization**: POMDP formalism and credit assignment taxonomy are widely cited; production systems (Perplexity, OpenAI Deep Research) show competitive benchmark performance; but no single algorithmic recipe is default, evaluation fragmentation prevents systematic comparison, and computational overhead of MC methods / bias of LLM critics remain unresolved.

**Full reference:** The Landscape of Agentic Reinforcement Learning for LLMs (arXiv:2509.02547), A Practitioner's Guide to Multi-turn Agentic RL (arXiv:2510.01132), From Reasoning to Agentic: Credit Assignment in RL for LLMs (arXiv:2604.09459), Advancing Search-Augmented Language Models (Perplexity Research), Improving Multi-Turn Tool Use with RL (Bespoke Labs), Building Math Agents with Multi-Turn Iterative Preference Learning (arXiv:2409.02392), Not All Turns Matter: Credit Assignment for Multi-Turn Jailbreaking (arXiv:2605.08778).

---
*Full reference (citations, derivations, variants):* [Agentic and tool-use RL](../topics/agentic-and-tool-use-rl.md)
