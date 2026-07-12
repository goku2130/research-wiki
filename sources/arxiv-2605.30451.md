---
id: arxiv:2605.30451
type: paper
title: 'VeriGate: Verifier-Gated Step-Level Supervision for GRPO'
url: https://arxiv.org/abs/2605.30451
retrieved: '2026-07-12'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# VeriGate: Verifier-Gated Step-Level Supervision for GRPO

VeriGate is a verifier-gated extension of Group Relative Policy Optimization (GRPO) designed to improve the training of large reasoning models (LRMs) by integrating process supervision without sacrificing the reliability of outcome-based verifiers.

### Core Problem
Standard GRPO relies on outcome-only rewards, which leads to three primary failures:
1. **Zero-Gradient Collapse:** On hard prompts, if all sampled trajectories receive the same reward (typically zero), the group-relative advantage collapses to zero, and learning stalls.
2. **Poor Credit Assignment:** Outcome rewards provide a single scalar for the entire trajectory, meaning all tokens in a correct solution are reinforced equally, regardless of their actual contribution to the result.
3. **Reward Hacking:** Naive attempts to fix these issues by replacing verifiers with Process Reward Models (PRMs)—referred to as "PRM-as-ORM"—often lead to reward hacking, where the model exploits PRM biases (e.g., verbosity or specific stylistic patterns) to increase scores without improving actual correctness.

### Method
VeriGate introduces a gated mechanism that uses process supervision only as a fallback when outcome rewards are uninformative. The method follows a three-step recipe:

**Step 1: Verifier Gating (S1)**
VeriGate maintains the authority of the verifier. For a prompt $x$ and a group of $G$ sampled trajectories:
* If the verifier rewards $\{r_i\}_{i=1}^G$ are mixed (at least one correct and one incorrect), the system uses standard GRPO.
* If all trajectories are incorrect ($r_1 = \dots = r_G = 0$), VeriGate activates PRM-based token-level supervision.
* If all trajectories are correct, no PRM supervision is used.

**Step 2: Future-Cumulated Token Rewards (S2)**
To avoid brittle trajectory-level aggregation, VeriGate uses Future-Cumulated Token Rewards (FCTR). Instead of a single scalar, each token in step $j$ of trajectory $i$ is assigned a reward $c_{i,j}$ based on the sum of all subsequent step rewards:

$$
c_{i,j}=\sum_{k=j}^{S_{i}}r_{i,k}
$$

where $r_{i,k}$ is the PRM score for step $k$ and $S_i$ is the total number of steps. This ensures tokens are credited based on the quality of the continuation they enable.

**Step 3: Group-Normalized Token-Level Advantages (S3)**
These rewards are converted into dense, group-normalized advantages to maintain stability and prevent absolute-value hacking. The prompt-level baseline $\bar{c}$ is calculated as:

$$
\bar{c}=\frac{\sum_{i=1}^{G}\sum_{j=1}^{S_{i}}c_{i,j}}{\sum_{i=1}^{G}S_{i}}
$$

The final token-level advantage $A_{i,j}$ is:

$$
A_{i,j}=\frac{c_{i,j}-\bar{c}}{\sigma(c)}
$$

where $\sigma(c)$ is the standard deviation of future-cumulated rewards across the group.

**Effective Advantage ($\widetilde{A}_{i,j}$):**

$$
\widetilde{A}_{i,j}=\begin{cases} A_{i}^{\mathrm{GRPO}}, & \text{if verifier rewards are mixed} \\ \frac{c_{i,j}-\bar{c}}{\sigma(c)}, & \text{if } r_{1}=\cdots=r_{G}=0 \end{cases}
$$

### Key Quantitative Results
Evaluated using Qwen2.5-Instruct (1.5B and 7B) trained on MATH, VeriGate demonstrated significant improvements over Vanilla-GRPO and PRM-as-ORM:
* **Accuracy Gains:** Average accuracy improved by approximately 20% for the 1.5B model and 12% for the 7B model across six reasoning benchmarks.
* **7B Model Performance:** On specific benchmarks, VeriGate (7B) increased AIME accuracy from 6.67% to 10.00%, AMC from 42.17% to 45.78%, and Minerva from 18.38% to 20.22%.
* **Reward Hacking Mitigation:** In cross-PRM verification (using Math-Shepherd as an external evaluator), VeriGate achieved a reward of 0.1424, significantly higher than PRM-as-ORM (0.0331), despite PRM-as-ORM having higher scores on the training PRM.
* **Zero-Gradient Recovery:** VeriGate showed a faster reduction in the fraction of prompts receiving zero verifier reward compared to baselines.

### Limitations
* **PRM Dependency:** The method still relies on the availability and quality of PRMs; systematic biases in the PRM can still influence updates on degenerate prompts.
* **Segmentation:** Credit assignment is dependent on the model's step segmentation; poor segmentation can weaken the effectiveness of the supervision.
* **Domain Scope:** Experiments were limited to mathematical reasoning; the authors note that transfer to code generation, tool use, or long-horizon planning remains untested.
