---
id: arxiv:2604.13902
type: paper
title: 'DiPO: Disentangled Perplexity Policy Optimization for Fine-grained Fine-tuning'
url: https://arxiv.org/html/2604.13902v1
retrieved: '2026-07-12'
maturity: comprehensive
topic: entropy-and-exploration
---

# DiPO: Disentangled Perplexity Policy Optimization

DiPO is a fine-grained exploration-exploitation trade-off (EETO) mechanism designed for Reinforcement Learning with Verifiable Rewards (RLVR) in Large Language Models (LLMs).

### Core Problem
The authors identify two primary dilemmas in GRPO-based methods regarding the EETO:
1. **Degradation of Extreme Samples:** "Hard groups" (all samples receive zero reward) and "easy groups" (all samples receive one reward) yield zero advantage, resulting in a lack of training gradients for both exploration and exploitation.
2. **Ineffective EETO:** Perplexity (PPL) distributions often overlap, where some error samples exhibit an exploitative tendency (low PPL) and some correct samples exhibit an exploratory tendency (high PPL), destabilizing policy optimization.

### Method
DiPO addresses these dilemmas through two primary modules: **Perplexity Space Disentangling (PSD)** and **Bidirectional Reward Reallocation (BRR)**.

#### 1. Perplexity Space Disentangling (PSD)
PSD partitions the sample space into an **Exploitation Space (EiS)** and an **Exploration Space (ErS)** based on a dynamically determined threshold $\tau^*$.
* **Online Estimation:** A PPL queue $\mathcal{Q}$ caches (PPL, reward) pairs from the most recent two batches.
* **Advantage Judgment:** To ensure PPL is meaningfully correlated with correctness, the system checks that the "correctness advantage" in EiS and "error advantage" in ErS are both positive:

$$
\Delta_{\text{EiS}}(\tau) = \Pr(R = 1 \mid P < \tau) - \Pr(R = 0 \mid P < \tau) > 0
$$

$$
\Delta_{\text{ErS}}(\tau) = \Pr(R = 0 \mid P > \tau) - \Pr(R = 1 \mid P > \tau) > 0
$$

* **Threshold Optimization:** The optimal threshold $\tau^*$ is found by minimizing classification errors:

$$
\tau^* = \arg \min_{\tau} \frac{1}{|\mathcal{Q}|} \sum_{(r_i, p_i) \in \mathcal{Q}} |r_i - \mathbb{I}(p_i < \tau)|
$$

This divides samples into four quadrants: Correct-High (CH), Correct-Low (CL), Error-High (EH), and Error-Low (EL).

#### 2. Bidirectional Reward Reallocation (BRR)
To avoid the instability of using PPL directly as a reward, BRR reallocates rewards only for zero-gradient groups (all 0s or all 1s) to drive entropy changes:
* **Hard Groups in EiS:** The sample with the maximum PPL is assigned a reward of 1 to encourage exploration (increase entropy).
* **Easy Groups in ErS:** The sample with the maximum PPL is assigned a reward of 0 to encourage exploitation (decrease entropy).
* **Orthogonality:** Reallocated rewards $\mathcal{R}_r$ are kept orthogonal to verification rewards $\mathcal{R}$ by setting rewards of "normal groups" to zero.

#### 3. Policy Optimization
The final objective function combines the baseline DAPO objective with the reallocated reward objective, weighted by hyperparameter $\alpha$:

$$
\mathcal{J}_{DiPO}(\theta) = \mathcal{J}_{DAPO}(\theta, \mathcal{R}) + \alpha \times \mathcal{J}_{DAPO}(\theta, \mathcal{R}_r)
$$

### Key Quantitative Results
DiPO was evaluated on mathematical reasoning and function calling tasks.

**Mathematical Reasoning (AVG ACC/mean@8):**
* **Qwen3-4B-Base:** DiPO achieved **50.55%**, surpassing DAPO (49.43%) and GRPO (48.92%).
* **Qwen3-8B-Base:** DiPO achieved **54.79%**, surpassing DAPO (53.23%) and GRPO (53.24%).
* **Qwen2.5-7B:** DiPO achieved **43.56%**, surpassing DAPO (42.73%) and GRPO (42.13%).

**Function Calling (Overall Accuracy on BFCLv3):**
* **Qwen2.5-3B-Instruct:** DiPO achieved **55.03%** vs. ToolRL+DAPO (53.21%).
* **Qwen2.5-7B-Instruct:** DiPO achieved **62.51%** vs. ToolRL+DAPO (61.06%).

**Risk Prediction (Qwen3-8B):**
DiPO achieved an Accuracy of **78.37%**, Recall of **79.49%**, and F1 Score of **86.84%**.

**Hyperparameter Analysis:**
The optimal value for $\alpha$ was found to be **0.1**. DiPO demonstrated higher robustness than entropy loss; for the 8B model, a tenfold increase in $\alpha$ (from 0.1 to 1.0) resulted in only a 1.43 point drop in AVG, whereas a small increase in entropy loss coefficients caused catastrophic performance collapse.

### Limitations
The authors state that a strict mathematical proof for the entropy changes induced by BRR is "unrealizable" due to the vast and complex parameter space of LLMs. The provided proof relies on "multiple idealized assumptions" and serves as an "approximate estimate of the trend."
