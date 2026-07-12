---
id: medium:inside-the-rlhf-engine-a-deep-dive-into-
type: web
title: 'Inside the RLHF Engine: A Deep Dive into SFT, Reward ...'
url: https://medium.com/foundation-models-deep-dive/inside-the-rlhf-engine-a-deep-dive-into-sft-reward-models-and-rl-fine-tuning-60978b291d55
retrieved: '2026-07-12'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

The article "Inside the RLHF Engine: A Deep Dive into SFT, Reward Models, and RL Fine-Tuning" describes the three-stage Reinforcement Learning from Human Feedback (RLHF) pipeline used to align Large Language Models (LLMs) with human preferences.

**Core Problem:**
The fundamental problem addressed by RLHF is aligning the behavior of Large Language Models (LLMs) with human preferences and instructions, moving beyond simply predicting the next token to generating responses that are helpful, harmless, and honest.

**Method/Recipe Step by Step:**

The RLHF pipeline consists of three distinct, sequential stages:

1.  **Supervised Fine-Tuning (SFT):**
    *   **Purpose:** To adapt a general-purpose pre-trained LLM to a specific domain and style, enabling it to generate responses that are more aligned with desired outputs. This stage lays the groundwork for subsequent RL fine-tuning.
    *   **Methodology:** A pre-trained LLM is fine-tuned using a dataset of high-quality, human-curated demonstrations. These demonstrations typically consist of prompt-response pairs that exemplify the desired behavior, style, and content. The fine-tuning process involves training the LLM to predict the next token in the human-generated responses given a prompt.

2.  **Reward Model (RM) Training:**
    *   **Purpose:** To create a model that can automatically score the quality of an LLM's response based on human preferences, thereby providing a scalable reward signal for reinforcement learning.
    *   **Methodology:**
        *   An LLM (often the SFT model or a copy of it) generates multiple responses to a given prompt.
        *   Human annotators then rank or provide preference judgments for these generated responses (e.g., "Response A is better than Response B").
        *   This human preference data is used to train a separate neural network, the Reward Model (RM). The RM learns to predict a scalar reward score for a given prompt-response pair, reflecting the human preference.
        *   The training objective for the RM is typically a pairwise ranking loss, where the model is optimized to assign a higher score to the preferred response over the dispreferred one.

3.  **Reinforcement Learning (RL) Fine-Tuning:**
    *   **Purpose:** To further fine-tune the SFT model using the learned Reward Model as a dynamic reward signal, optimizing the LLM's policy to generate responses that maximize the RM's score. This is the stage where the LLM truly learns to align with human preferences through trial and error.
    *   **Methodology:**
        *   The SFT model is treated as the policy to be optimized.
        *   A Reinforcement Learning algorithm, commonly Proximal Policy Optimization (PPO), is used.
        *   The SFT model generates responses to new prompts.
        *   The trained Reward Model evaluates these generated responses and provides a reward signal.
        *   The PPO algorithm then updates the SFT model's parameters to increase the likelihood of generating responses that receive higher rewards from the RM.
        *   To prevent the model from drifting too far from the initial SFT model and generating nonsensical text, a Kullback-Leibler (KL) divergence penalty is often incorporated into the reward function. This penalty discourages large deviations from the SFT model's original probability distribution over tokens.

**Key Formulas (Implicit in description):**

While specific LaTeX formulas are not provided in the text, the description implies the following conceptual objectives:

*   **SFT Loss (Cross-Entropy Loss):** The SFT model is trained to minimize the negative log-likelihood of the human-provided tokens given the prompt and previous tokens.
    *   $L_{SFT} = - \sum_{i=1}^{N} \log P(y_i | x, y_{<i}; \theta_{SFT})$
    where $P(y_i | x, y_{<i}; \theta_{SFT})$ is the probability of the $i$-th token in the human response, given the prompt $x$ and preceding tokens $y_{<i}$, parameterized by $\theta_{SFT}$.

*   **Reward Model Loss (Pairwise Ranking Loss):** The RM is trained to maximize the difference in scores between preferred and dispreferred responses.
    *   $L_{RM} = - \log(\sigma(r(x, y_w) - r(x, y_l)))$
    where $r(x, y)$ is the score assigned by the RM to response $y$ for prompt $x$, $y_w$ is the preferred response, $y_l$ is the dispreferred response, and $\sigma$ is the sigmoid function.

*   **RL Objective (PPO with KL Penalty):** The RL fine-tuning aims to maximize the expected reward while staying close to the SFT policy.
    *   $J_{RL} = E_{(x,y) \sim \pi_{RL}} [R_{RM}(x,y) - \beta D_{KL}(\pi_{RL}(\cdot|x) || \pi_{SFT}(\cdot|x))]$
    where $R_{RM}(x,y)$ is the reward from the Reward Model, $\pi_{RL}$ is the current RL policy, $\pi_{SFT}$ is the SFT policy, $D_{KL}$ is the KL divergence, and $\beta$ is a coefficient controlling the strength of the KL penalty.

**Key Quantitative Results and Numbers:**
No specific quantitative results or numbers are provided in the text. The article focuses on the methodological description of the pipeline.

**Stated Limitations:**
The article does not explicitly state limitations of the RLHF process itself. However, it implicitly highlights the dependency on the "quality and coherence of each step," suggesting that issues in any stage (e.g., low-quality SFT data, inconsistent human preferences for RM training, or unstable RL optimization) could negatively impact the final model's performance. The reliance on human feedback also implies potential scalability and bias issues, though these are not explicitly discussed as limitations in this particular excerpt.
