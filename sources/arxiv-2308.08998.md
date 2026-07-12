---
id: arxiv:2308.08998
type: paper
title: 'ReST: Reinforced Self-Training for Language Modeling'
url: https://arxiv.org/abs/2308.08998
retrieved: '2026-07-12'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# Reinforced Self-Training (ReST) for Language Modeling

Reinforced Self-Training (ReST) is a "growing batch" reinforcement learning (RL) algorithm designed to align large language models (LLMs) with human preferences. 

### Core Problem
The authors address the limitations of existing Reinforcement Learning from Human Feedback (RLHF) methods. Online RL methods (e.g., PPO) are computationally expensive due to the need for continuous sampling and are prone to "reward hacking." Conversely, offline RL is more efficient but is heavily constrained by the quality of the initial static dataset. ReST aims to combine the computational efficiency of offline RL with the exploration capabilities of online RL.

### Method/Recipe
ReST decouples the RL pipeline into two distinct offline stages: a **Grow** step (data generation) and an **Improve** step (policy optimization).

1.  **Initialization**: Train an initial supervised policy $\pi_\theta$ on a dataset $\mathcal{D}$ using the negative log likelihood (NLL) loss:

$$
\mathcal{L}_{\text{NLL}}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}} \left[ \sum_{t=1}^T \log \pi_\theta(y_t \mid y_{1:t-1}, x) \right]
$$

2.  **Grow Step (Outer Loop)**: Generate multiple output sequences $y$ for each context $x$ sampled from $\mathcal{D}$ using the current policy $\pi_\theta$. This creates an augmented dataset $\mathcal{D}_g$:

$$
\mathcal{D}_g = \{ (x^i, y^i)|_{i=1}^{N_g} \text{ such that } x^i \sim \mathcal{D}, y^i \sim \pi_\theta(y|x^i) \} \cup \mathcal{D}
$$

3.  **Improve Step (Inner Loop)**:
    *   **Scoring**: Each sample in $\mathcal{D}_g$ is scored using a reward model $R(x, y)$.
    *   **Filtering**: A filtering function $F$ retains only samples exceeding a threshold $\tau$:

$$
F(x, y; \tau) = \mathbb{1}_{R(x, y) > \tau}
$$

    *   **Optimization**: The policy is fine-tuned using a reward-weighted loss $J$:

$$
J(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}_g} [ F(x, y; \tau) \mathcal{L}(x, y; \theta) ]
$$

    *   **Iteration**: The Improve step is repeated multiple times with an increasing sequence of thresholds ($\tau_1 < \tau_2 < \dots < \tau_N$) and decreasing learning rates to prevent overfitting to small, high-quality subsets.
4.  **Cycle**: The final policy from the Improve steps is used to initiate the next Grow step.

### Key Quantitative Results
The authors evaluated ReST on machine translation benchmarks (IWSLT 2014 De-En, WMT 2020 Zh-En, and an internal Web Domain En-Zh dataset) using Metric X as the reward model.

*   **Performance Gains**: On IWSLT 2014, ReST (G=2, I=3) achieved an average reward of **83.1**, significantly outperforming the supervised baseline (BC G=0, I=0) at **70.9** and Online RL at **71.6**.
*   **Iterative Growth**: Adding a second Grow step provided a **5.3 point** improvement on IWSLT 2014 and a **0.8 point** improvement on the Web Domain task over the first Grow step.
*   **Inference Efficiency**: With Best-of-N sampling, a ReST variant with $N < 10$ matched the performance of a supervised BC model using $N = 200$.
*   **Loss Comparison**: The authors found that the simple BC loss outperformed other offline RL losses (such as GOLD, BVMPO, and OAC) when used within the ReST framework to improve reward model scores.

### Stated Limitations
*   **Reward Model as Proxy**: The learned reward model is an imperfect proxy for human preferences. Human evaluation rankings did not always match reward model rankings, particularly as the policy moved further from the behavior model.
*   **Overfitting and Generalization**: Repeated iterations of Grow and Improve steps increase the risk of overfitting to the reward model. The reward model's ability to generalize decreases as the policy diverges from the original training distribution.
*   **Reward Delusions**: The authors noted "delusions" in the reward model, such as rewards increasing when a translation repeated the same sentence regardless of quality.
