---
id: bair:rethinking-the-role-of-ppo-in-rlhf-bair-
type: web
title: Rethinking the Role of PPO in RLHF (BAIR Blog)
url: http://bair.berkeley.edu/blog/2023/10/16/p3o/
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Pairwise Proximal Policy Optimization (P3O)

### Core Problem
In the standard Reinforcement Learning from Human Feedback (RLHF) pipeline, there is a fundamental discrepancy between the reward learning phase and the RL fine-tuning phase. Reward models are trained using **relative feedback** (comparisons between response pairs), yet the RL fine-tuning stage (typically using PPO) optimizes for a **single, absolute reward**. 

This inconsistency makes the RL process sensitive to "reward translation"—where shifting a reward function by a constant $\delta(x)$ does not change the preference loss but can significantly mislead an RL algorithm. Consequently, the policy may be disrupted by the reward scale of a prompt rather than learning the actual relative preferences.

### Method
To resolve this, the authors propose **Pairwise Proximal Policy Optimization (P3O)**, which shifts RL from absolute feedback to relative feedback. The method follows these steps:

1.  **Supervised Fine-Tuning (SFT):** The pre-trained model is trained via maximum likelihood loss on high-quality data.
2.  **Reward Modeling:** A reward model $r_\phi$ is trained using a comparative loss based on human preferences ($y_w \succ y_l$):

$$
L_R = \mathbb{E}_{(x, y_l, y_w) \sim D} \log \sigma(r_\phi(y_w|x) - r_\phi(y_l|x))
$$

3.  **Pairwise Policy Gradient (PPG):** Instead of the Vanilla Policy Gradient (VPG), which relies on absolute reward magnitudes, P3O utilizes a comparative form involving two responses ($y_1, y_2$) for the same prompt:

$$
\mathbb{E}_{y_1, y_2 \sim \pi_\theta} (r(y_1|x) - r(y_2|x)) \frac{\nabla(\log \pi_\theta(y_1|x) \pi_\theta(y_2|x))}{2}
$$

    This formulation ensures the algorithm is invariant to reward translation.
4.  **PPO Enhancements:** To stabilize training and improve performance, P3O incorporates:
    *   **Importance Sampling:** A replay buffer stores responses from the old policy ($\pi_{old}$), and gradients are computed as a weighted sum using importance sampling ratios.
    *   **Clipping:** To prevent excessively large updates and better manage the trade-off between KL divergence and reward, the algorithm clips both the importance sampling ratio and the gradient updates.
    *   **Variants:** The authors identify two implementation styles for clipping: **V1** (separate clipping) and **V2** (joint clipping).

### Key Quantitative Results
P3O was evaluated on the **TL;DR** (summarization) and **Anthropic Helpful and Harmless (HH)** (question-answering) datasets.

*   **KL-Reward Frontier:** P3O demonstrated "strictly dominant frontiers" compared to PPO and DPO, meaning it achieved higher rewards for the same level of KL divergence from the reference policy.
*   **Head-to-Head Comparisons (HH Dataset):**
    *   **Win Rates (GPT-4 Evaluation):** P3O achieved a win rate of **57.0% against PPO** and **69.3% against SFT**.
    *   **P3O vs. DPO:** While DPO marginally surpassed P3O in raw reward, it exhibited considerably higher KL divergence. This resulted in DPO having a reward win rate of 49.5% against P3O, but a lower **GPT-4 win rate of 45.4%**.
*   **Overall Ranking:** In terms of average KL-divergence and reward, the ranking was: $\text{DPO} > \text{P3O} > \text{PPO} > \text{SFT}$.

### Limitations
The authors note that excessive deviation from the reference policy (high KL divergence) can lead the online policy to "cut corners" of the reward model, resulting in the generation of incoherent continuations. While P3O provides better KL control than DPO, the balance between reward maximization and maintaining the integrity of the reference policy remains a critical constraint in LLM alignment.
