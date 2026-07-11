---
id: aclanthology:adaptive-length-bias-mitigation-in-rewar
type: web
title: Adaptive Length Bias Mitigation in Reward Models for RLHF
url: https://aclanthology.org/2025.findings-naacl.169.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: length-and-format-bias
---

# Adaptive Length Bias Mitigation (ALBM) in Reward Models

### Core Problem
In Reinforcement Learning from Human Feedback (RLHF), reward models (RMs) frequently exhibit **length bias**, where they erroneously assign higher rewards to longer responses regardless of actual quality. This leads to "reward hacking," where policy models generate excessively verbose responses to maximize rewards. While existing debiasing methods attempt to remove this correlation, they often do so uniformly. The authors argue that response length is a context-specific factor; some queries are "length-neutral" (where brevity is preferred), while others are "length-sensitive" (where detail is legitimate). Indiscriminate suppression of length bias reduces alignment accuracy by penalizing legitimate length preferences.

### Method: Adaptive Length Bias Mitigation (ALBM)
ALBM decouples length bias from quality and then reintegrates them dynamically based on the prompt context. The process follows these steps:

1.  **Reward Decomposition**: The RM uses a shared backbone $f_\phi$ with two distinct linear heads: a quality reward head $g_{\psi_q}$ and a length reward head $g_{\psi_l}$.
2.  **Disentanglement Constraints**: To ensure the heads specialize, two constraints are applied:
    *   **Explicit Constraint ($\mathcal{L}_{EL}$)**: This loss maximizes the Pearson correlation $\rho$ between the length reward and the actual token length $L(y)$, while minimizing the correlation between the quality reward and length:

$$
\mathcal{L}_{EL} = \|\rho(g_{\psi_q} \circ f_\phi(x,y), L(y))\| - \rho(g_{\psi_l} \circ f_\phi(x,y), L(y))
$$

    *   **Implicit Constraint ($\mathcal{L}_{IL}$)**: This enforces orthogonality between the projection weights of the two heads:

$$
\mathcal{L}_{IL} = \| \mathbf{W}_{\psi_q} \mathbf{W}_{\psi_l}^T \|
$$

3.  **Adaptive Integration**: A third head, the prompt analyzer $g_{\psi_p}$, is introduced to predict the appropriate weight of the length reward based on the input prompt $x$.
4.  **Final Reward Calculation**: The total reward is the sum of the quality reward and the weighted length reward:

$$
r_{final} = g_{\psi_q} \circ f_\phi(x,y) + (g_{\psi_p} \circ f_\phi(x)) \circ g_{\psi_l} \circ f_\phi(x,y)
$$

5.  **Training**: The model is trained using a combined loss function:

$$
\mathcal{L}_{final} = \mathcal{L}_R + \lambda_{DR}\mathcal{L}_{DR} + \lambda_{EL}\mathcal{L}_{EL} + \lambda_{IL}\mathcal{L}_{IL}
$$

    where $\mathcal{L}_R$ is the adaptive ranking loss and $\lambda$ values are regularization coefficients.

### Key Quantitative Results
The authors evaluated ALBM on the WebGPT dataset using Vicuna-7B as the base model, comparing it against a Vanilla RM and two baselines: **Bal** (data intervention) and **Odin** (model intervention).

*   **Alignment Accuracy and Correlation**: ALBM achieved the highest accuracy (**0.6318**) compared to Vanilla (0.6223), Bal (0.5906), and Odin (0.5792). It significantly reduced the Spearman correlation between reward and length to **-0.0209**, compared to the Vanilla RM's **0.5105**.
*   **Failure Rates**: ALBM balanced performance across query types, showing a length-neutral failure rate (LN-FR) of **0.4655** and a length-sensitive failure rate (LS-FR) of **0.3049**.
*   **Downstream RL Impact**:
    *   **Verbosity**: The average response length for ALBM was **228 tokens**, effectively reducing the verbosity of the Vanilla RM (**261 tokens**) while remaining closer to the SFT baseline (**198 tokens**) than the Vanilla model.
    *   **Win Rates**: In GPT-4o evaluations, ALBM achieved a **0.37 overall win rate** against the Vanilla model, outperforming it on both length-sensitive (0.40 win rate) and length-neutral (0.34 win rate) data.
*   **Efficiency**: The forward pass time increased by only **0.75%** compared to a standard RM.

### Limitations
The authors note that experiments were conducted on a limited set of datasets. Preliminary findings suggest that length hacking does not manifest universally across all models and datasets; its occurrence depends on specific combinations of training data, model architecture, and methodology. Consequently, the generalizability of ALBM may be restricted to scenarios where length hacking is present.
