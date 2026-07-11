---
id: arxiv:2605.08778
type: paper
title: 'Not All Turns Matter: Credit Assignment for Multi-Turn Jailbreaking'
url: https://arxiv.org/abs/2605.08778
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Not All Turns Matter: Credit Assignment for Multi-Turn Jailbreaking

### Core Problem
Reinforcement learning (RL) for multi-turn jailbreaking typically relies on trajectory-level outcome rewards, which broadcast a uniform signal to every turn in a dialogue. The authors identify this as a **credit assignment problem**, noting that turn-level contributions are actually:
1. **Non-uniform:** Only a few turns typically drive success; many are redundant.
2. **Phase-dependent:** The effectiveness of a prompt depends on whether it is introduced during the early "contextual setup" or later phases.
3. **Target-specific:** Different target models have distinct safety boundaries and refusal sensitivities.

Coarse outcome supervision leads to over-rewarding redundant turns in successful trajectories and under-crediting useful intermediate turns in failed ones.

### Method: TRACE
**TRACE** (Turn-level Assignment for Credit) is a framework that replaces the trajectory-level advantage $\hat{A}_i$ in Multi-turn GRPO with a turn-aware advantage $\hat{A}_{i,t}$:

$$
\hat{A}_{i,t} = m_{i,t} \hat{A}_i^o + \hat{A}_{i,t}^p
$$

Where $\hat{A}_i^o$ is the group-normalized outcome advantage, $m_{i,t}$ is a turn-aware multiplier, and $\hat{A}_{i,t}^p$ is a local process penalty.

#### 1. Success-Side Credit Assignment ($\tau_i \in \mathcal{S}^+$)
For successful trajectories, TRACE uses **leave-one-turn-out semantic masking**. For each non-final turn $t < T$, the interaction $(x_t, y_t)$ is removed, and the final response $y'_T$ is resampled. The raw credit is:

$$
c_{i,t} := r(x_0, y_{i,T_i}) - r(x_0, y'_{i,T_i})
$$

This value is normalized to $z_{i,t}$ and converted into the multiplier $m_{i,t}^+$.

#### 2. Failure-Side Deviation Penalty ($\tau_i \in \mathcal{S}^-$)
For failed trajectories, TRACE applies target-specific penalties based on successful trajectory priors:
*   **Harmfulness Penalty ($B_{i,t}^H$):** Uses a concentration-adaptive threshold based on the success prior $q_t$:

$$
B_{i,t}^H := \max\left(0, \sum_{\ell \in \mathcal{C}} (q_{t,\ell})^2 - q_{t,\ell_{i,t}}\right)
$$

*   **Relevance Penalty ($B_{i,t}^R$):** Penalizes prompts that deviate from the original intent $x_0$ using cosine similarity $E_{i,t}$ and a lower reference $L_t$:

$$
B_{i,t}^R := \max\left(0, (L_t - E_{i,t}) / L_t\right)
$$

The multiplier $m_{i,t}^-$ is then derived from the combined penalty $B_{i,t} = B_{i,t}^H + B_{i,t}^R$.

#### 3. Refusal-Aware Local Process Penalty
To capture target-specific refusals, a local penalty is added:

$$
\hat{A}_{i,t}^p = \lambda_p r_{i,t}^p, \quad \text{where } r_{i,t}^p = -\mathbb{I}\{\text{refusal}(x_{i,t}, y_{i,t})\}
$$

#### 4. Multi-turn Defense Alignment
TRACE reuses attack-side signals for defense by splitting successful attack turns into **latent-risk turns** (attack-critical intermediate steps) and **direct-harm turns** (the final jailbreak step). This allows for early risk intervention via DPO alignment.

### Key Quantitative Results
*   **Effectiveness:** TRACE (mix) achieved an average **ASR@1 of 87.10%**, a relative improvement of approximately **25%** over the strongest RL baseline, TROJail (69.97%).
*   **Transferability:** TRACE (mix) transferred effectively to closed-source models, achieving **ASR@5 of 96.7% on GPT-4o** and **93.1% on Gemini-2.5-Pro**.
*   **Efficiency:** TRACE reached over **80% ASR@1 on gpt-oss-20b within four turns**, outperforming workflow-based baselines.
*   **Defense:** TRACE-based alignment significantly reduced multi-turn ASR; for example, against a TRACE attack, the ASR dropped from **87.84% (undefended) to 9.43% (defended)** while maintaining or improving general capabilities (e.g., GPQA accuracy).

### Stated Limitations
The authors identify three primary areas for future work:
1. Improving the **diversity of jailbreak strategies** to cover more varied adversarial settings.
2. Exploring more effective **defense alignment methods** to better balance safety and helpfulness.
3. Extending turn-aware credit assignment to **broader multi-turn agentic settings**.
