---
id: arxiv:2401.07181
type: paper
title: Reinforcement Learning from LLM Feedback to Counteract Goal Misgeneralization
url: https://arxiv.org/abs/2401.07181
retrieved: '2026-07-12'
maturity: comprehensive
topic: sycophancy-and-misgeneralization
---

# Reinforcement Learning from LLM Feedback to Counteract Goal Misgeneralization

### Core Problem
The authors address **goal misgeneralization**, a form of inner alignment failure in reinforcement learning (RL). This occurs when an agent achieves high training rewards by optimizing a proxy goal ($G'$) that correlates with the intended goal ($G$) in the training distribution but diverges in out-of-distribution (OOD) test environments. Consequently, the agent remains capable but pursues the unintended behavioral objective, leading to low test rewards.

### Method
The proposed method integrates a Large Language Model (LLM) to provide scalable oversight and correct biased behaviors without requiring explicit training on OOD data. The process follows these steps:

1.  **Initial RL Training**: An agent is trained for $\tau \cdot T$ timesteps using a standard environment reward function $R$.
2.  **LLM Assessment**: The training is paused, and $N^I$ policy roll-outs are sampled. These roll-outs, along with a description of the environment, are provided to an LLM (GPT-4 Turbo). The LLM analyzes the policy to identify potential failure scenarios and suggests modifications to the training environment to mitigate misgeneralization.
3.  **Sampling for Reward Modeling**: Based on LLM suggestions, new environments are generated. The agent is deployed in these environments to sample $N^R$ policy roll-outs.
4.  **LLM Preference Labeling**: The sampled roll-outs are formed into pairs. The LLM rates which roll-out is preferred. To counteract **position bias** (where LLMs prefer candidates in a specific order), the authors elicit preferences for every pair twice, switching the order of candidates and averaging the results.
5.  **Reward Model Training**: A reward model $\mathcal{R}$ is trained from the database of LLM preferences $D$ by minimizing the cross-entropy loss:

$$
\mathcal{L}(\mathcal{R})=-\sum_{(\omega^{1},\:\omega^{2},\:\mu)\in D}\mu(1)\text{l o g}\hat{P}(\omega^{1}\succ\omega^{2})+\mu(2)\text{l o g}\hat{P}(\omega^{2}\succ\omega^{1})
$$

    where the predicted probability $\hat{P}$ is defined as:

$$
\hat{P}(\omega^{1}\succ\omega^{2})=\frac{\text{e x p}\sum_{\omega^{1}}\mathcal{R}(s_{t},s_{t+1})}{\text{e x p}\sum_{\omega^{1}}\mathcal{R}(s_{t},s_{t+1})+\text{e x p}\sum_{\omega^{2}}\mathcal{R}(s_{t},s_{t+1})}
$$

6.  **Fine-tuning**: RL training resumes using a composite reward function $R'$:

$$
R' = \lambda \cdot \mathcal{R} + (1 - \lambda) \cdot R
$$

    The authors primarily evaluate the case where $\lambda = 1$, meaning the agent is driven entirely by the LLM-informed reward model.

### Key Quantitative Results
The method was evaluated using the OpenAI Procgen Maze game, where the intended goal is cheese.

*   **LLM Capabilities**: GPT-4 identified environment confoundedness in 80% of queries. Notably, GPT-4 could correctly assess whether a trajectory reached the goal 77% of the time, despite only being able to solve the maze itself 10% of the time.
*   **Reward Model Performance**: For the predominant preference classes, the reward model achieved:
    *   **(1.0, 0.0)**: Precision 0.95, Recall 0.52, F1 0.67.
    *   **(0.0, 1.0)**: Precision 0.97, Recall 0.53, F1 0.69.
    *   **(0.5, 0.5)**: Precision 0.39, Recall 0.80, F1 0.52.
*   **Generalization**: The method effectively reduced goal misgeneralization across 7 of the 11 tested randomization regions (sizes 0–10). The most pronounced improvements occurred in randomization region 4.
*   **Position Bias**: In raw preference labeling, GPT-4 preferred the second position in 56% of pairs compared to 43% for the first position.

### Limitations
*   **Indistinguishable Goals**: In scenarios with very small randomization regions (e.g., size 1), the intended goal and proxy goal are indistinguishable, and the LLM-informed reward model fails to enhance generalization.
*   **Inherent Position Bias**: The authors note that regardless of the training method, agents tend to learn a location rather than a feature. This may be because positional information is more prominently represented in the state space or simpler to learn than feature-based goals.
