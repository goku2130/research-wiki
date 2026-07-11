---
id: en:reinforcement-learning-from-human-feedba
type: web
title: Reinforcement learning from human feedback
url: https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

Reinforcement Learning from Human Feedback (RLHF) is a machine learning technique designed to align an intelligent agent's behavior with human preferences. It addresses the challenge of explicitly defining a reward function in classical reinforcement learning, especially for tasks where human preferences are complex and difficult to quantify directly.

**Core Problem:**
The primary problem RLHF solves is the difficulty in specifying a reward function that accurately reflects nuanced human preferences for complex tasks. While humans can easily judge the quality of an agent's output (e.g., whether text is helpful or harmful), it is challenging to programmatically define a reward signal that captures these subjective evaluations. Traditional reinforcement learning struggles with sparse or noisy reward functions, and prior attempts to incorporate human feedback were often narrow or failed on complex tasks.

**Method/Recipe Step by Step:**

1.  **Initial Model Training (Supervised Learning):**
    *   A base model, typically a pre-trained autoregressive language model, is initialized.
    *   This model is then fine-tuned in a supervised manner on a relatively small dataset of prompt-response pairs created by human annotators. This initial step provides the model with a foundational understanding of language and task-specific behavior.

2.  **Human Feedback Collection:**
    *   The initialized model generates multiple responses to a given prompt.
    *   Human annotators are presented with these responses and asked to rank them according to their preferences. While ranking is common, other forms of feedback like numerical scores or direct edits have been explored. The Elo rating system can be used to score outputs based on these rankings.

3.  **Reward Model Training:**
    *   A separate "reward model" is trained. This model is also typically initialized from the same pre-trained base model.
    *   The final layer of this model is replaced with a randomly initialized regression head, allowing it to output a single numerical "reward" score for any given prompt and response.
    *   This reward model is trained on the human preference comparison data collected in the previous step. Its objective is to learn to predict human preferences, assigning higher scores to preferred responses and lower scores to less preferred ones.

4.  **Reinforcement Learning Policy Optimization:**
    *   The trained reward model then serves as the reward function for a reinforcement learning policy.
    *   This policy, also often initialized from the same pre-trained base model, is optimized using an algorithm like Proximal Policy Optimization (PPO).
    *   The policy learns to generate responses that maximize the reward signal provided by the reward model, thereby aligning its behavior with the learned human preferences.

**Key Formulas:**

The reward model is trained to minimize the following cross-entropy loss function:

$$
\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}}E_{(x,y_{w},y_{l})}\left[\log(\sigma(r_{\theta}(x,y_{w})-r_{\theta}(x,y_{l})))\right]
$$

This can also be expressed as:

$$
\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}}E_{(x,y_{w},y_{l})}\log \left[\frac{e^{r_{\theta}(x,y_{w})}}{e^{r_{\theta}(x,y_{w})}+e^{r_{\theta}(x,y_{l})}}\right]
$$

Where:
*   $K$ is the number of responses ranked by labelers.
*   $r_{\theta}(x,y)$ is the output of the reward model for prompt $x$ and response $y$.
*   $y_w$ represents the preferred (winning) response.
*   $y_l$ represents the less preferred (losing) response.
*   $\sigma$ is the sigmoid function.
*   $E$ denotes the expectation over the collected comparison data.

**Key Quantitative Results and Numbers:**

*   RLHF has been shown to require relatively small amounts of comparison data to be effective.
*   Increasing the amount of data tends to be less effective than proportionally increasing the size of the reward model.
*   When learning from pairwise comparisons under the Bradley–Terry–Luce model, the maximum likelihood estimator (MLE) for linear reward functions converges if the comparison data is generated under a well-specified linear model.
*   In offline data collection for policy training, a pessimistic MLE incorporating a lower confidence bound as the reward estimate is most effective. K-wise comparisons are asymptotically more efficient than converting them to pairwise comparisons for prediction.
*   In online scenarios with pairwise comparisons under the Bradley–Terry–Luce model, an optimistic MLE with an upper confidence bound can lead to sample-efficient algorithms for minimizing regret.
*   In applications like Atari games, RLHF can lead to superior performance over traditional RL with score metrics, sometimes surpassing human performance, because human preferences can contain more useful information.
*   In text-to-image models, KL regularization in RLHF helps stabilize training and produces significantly higher quality image outputs compared to training without it.

**Stated Limitations:**

*   Sourcing high-quality human preference data is an expensive process, despite RLHF not requiring massive amounts of data.
*   If human preference data is not carefully collected from a representative sample, the resulting model may exhibit unwanted biases.
*   A key challenge in RLHF, particularly when learning from pairwise comparisons, is the non-Markovian nature of its optimal policies, meaning the best course of action often depends on past events and decisions, requiring memory-dependent strategies.
