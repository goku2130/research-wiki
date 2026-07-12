---
id: bair:rethinking-the-role-of-ppo-in-rlhf
type: web
title: Rethinking the Role of PPO in RLHF
url: http://bair.berkeley.edu/blog/2023/10/16/p3o/
retrieved: '2026-07-12'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

The core problem addressed is the inconsistency in Reinforcement Learning with Human Feedback (RLHF) between the reward modeling stage, which uses comparative human preferences, and the RL fine-tuning stage, which optimizes a single, non-comparative reward. This discrepancy can lead to instability and implementation complications in existing RL algorithms like PPO, and makes the RL fine-tuning susceptible to issues arising from the non-uniqueness of the reward function (e.g., reward translation invariance).

The method proposed is Pairwise Proximal Policy Optimization (P3O), which aims to unify the reward learning and RL fine-tuning stages by performing RL in a comparative manner.

The method involves the following steps:

1.  **Supervised Fine-Tuning (SFT) Stage**: A pre-trained model is fine-tuned using maximum likelihood loss on a high-quality dataset to learn to respond to human queries.
2.  **Reward Modeling Stage**: The SFT model generates pairs of answers ($y_1, y_2$) for given prompts ($x$). Human labelers express a preference ($y_w \succ y_l$) for one answer over the other. A reward model ($r_\phi$) is trained using a comparative loss function:

$$
L_R = E_{(x, y_l, y_w) \sim D} [\log \sigma(r_\phi(y_w|x) - r_\phi(y_l|x))]
$$

3.  **RL Fine-Tuning Stage (P3O)**:
    *   **Pairwise Policy Gradient (PPG)**: Instead of the vanilla policy gradient (VPG) which relies on the absolute magnitude of the reward, P3O uses a comparative form of the policy gradient that leverages reward differences between two responses for the same prompt. This makes the algorithm invariant to reward translation. The pairwise policy gradient is formulated as:

$$
E_{y_1, y_2 \sim \pi_\theta} [(r(y_1|x) - r(y_2|x)) \nabla (\log \pi_\theta(y_1|x) - \log \pi_\theta(y_2|x))/2]
$$

    *   **Importance Sampling**: To improve performance and utilize a replay buffer, importance sampling is applied. A batch of responses is sampled from the replay buffer (generated from $\pi_{old}$), and an importance sampling ratio is computed for each response pair. The gradient is then a weighted sum of gradients from these pairs.
    *   **Clipping**: To prevent excessively large updates and control the trade-off between KL divergence and reward, clipping is applied to both the importance sampling ratio and the gradient update. Two variants, P3O-V1 and P3O-V2, are distinguished by either separate or joint clipping techniques.

Key quantitative results and numbers:

*   **KL-Reward Frontier**: P3O consistently shows strictly dominant frontiers compared to PPO and DPO across various model sizes on both TL;DR summarization and Anthropic Helpful and Harmless (HH) question-answering datasets. This indicates P3O achieves higher reward for a given KL divergence from the reference policy, or lower KL divergence for a given reward.
*   **Head-to-Head Comparisons (HH dataset)**:
    *   **Reward Win Rate**: DPO marginally surpasses P3O in reward, with a 49.5% win rate against P3O.
    *   **GPT-4 Win Rate**: P3O exhibits a GPT-4 win rate of 57.0% against PPO and 69.3% against SFT. Against DPO, P3O has a GPT-4 win rate of 54.6% (100% - 45.4%). This suggests that while DPO might achieve slightly higher proxy reward, its higher KL-divergence leads to lower quality generations as judged by GPT-4.
    *   The average KL-divergence and reward ranking of models is DPO > P3O > PPO > SFT. However, when considering GPT-4 evaluations, P3O aligns better with human preference.

Stated limitations:

*   The blog post does not explicitly state limitations of P3O itself. However, it highlights the general challenge in RLHF that the reward function is non-unique, and a simple shift in reward $r(y|x) + \delta(x)$ creates another valid reward function that can mislead RL algorithms if not handled properly. P3O is designed to address this specific issue by being invariant to reward translation.
*   The evaluation notes that DPO, while achieving a higher reward win rate against P3O (49.5%), has a considerably higher KL-divergence, which may be detrimental to the quality of generation, resulting in a lower GPT-4 win rate (45.4%) against P3O. This implies that optimizing solely for the learned reward without sufficient KL control can lead to undesirable outcomes.
