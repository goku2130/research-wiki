---
id: ojs:pdf-beyond-prompt-engineering-a-reinforc
type: web
title: '[PDF] Beyond Prompt Engineering: A Reinforced Token-Level Input ...'
url: https://ojs.aaai.org/index.php/AAAI/article/view/34586/36741
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

Beyond Prompt Engineering: A Reinforced Token-Level Input Refinement for Large Language Models (RTLIR) addresses the challenge of irrelevant information in Large Language Model (LLM) inputs, which can consume computational resources, lower processing efficiency, and compromise output quality. Existing prompt engineering methods are often context-dependent, rely on expert experience, and can introduce biases, limiting their broad applicability.

The core problem RTLIR tackles is the need for an efficient and automated method to refine large-scale text data input for LLMs, improving processing speed and output quality while maintaining the original text structure.

**Method/Recipe Step by Step:**

1.  **Input Preprocessing:** The input text (T) is preprocessed through tokenization and stop word removal, converting it into a sequence of token IDs ($ID_{seq} = \{ID_1, ID_2, \ldots, ID_i\}$).
2.  **Embedding Conversion:** These token IDs are then converted into corresponding embedding vectors ($v_{seq} = \Phi_{embed}(ID_{seq})$), where $v_{seq} = \{v_1, v_2, \ldots, v_i\}$ and $v_i$ is the embedding vector for $ID_i$. This establishes the initial state for the reinforcement learning agent.
3.  **State Definition:** Each state ($s \in S$) in the Markov Decision Process (MDP) represents the current context of the text being processed. It concatenates the embedding of a token ($v_i$) and its corresponding action ($a_i$), defined as $s = \mathrm{CONCATENATE}[v_1, v_2, \ldots, v_i, a_1, a_2, \ldots, a_i]$. Initially, all actions ($a_i$) are set to 1 (keep), meaning all tokens are retained by default.
4.  **Action Definition:** Each action ($a \in A$) is a binary decision: either retain (1) or delete (0) the currently evaluated token ID.
5.  **Reward Design:**
    *   **Immediate Reward:** This evaluates the impact of keeping or removing a single token. It is calculated based on the cosine similarity between the current token's embedding and a target embedding, adjusted by a threshold $\beta$.
    *   **Terminal Reward:** This evaluates the overall effect after the entire text processing is completed, considering the change in information quality before and after input refinement. It is defined as:

$$
r_t(s_t,a_t)=\frac{\log p(\hat{v}_{seq},a_t)-\log p(v_{seq},a_t)}{\log p(v_{seq}^*,a_t)-\log p(v_{seq},a_t)+1}
$$

        where $v_{seq}$, $\hat{v}_{seq}$, and $v_{seq}^*$ represent the original, refined, and optimal input refinement text sequence embeddings, respectively.
6.  **Policy Learning (Q-learning):** RTLIR aims to find a policy ($\pi$) that maximizes the information value of the output while minimizing irrelevant token IDs. This is formulated using Q-learning:

$$
Q(s,a;\theta)=\mathbb{E}\left[r_t+\gamma\text{max}_{a^{\prime}}Q(s^{\prime},a^{\prime};\theta)\mid s,a\right]
$$

    where $\theta$ represents network parameters, $r_t$ is the reward, and $\gamma$ is the discount factor.
7.  **Value Decomposition:** To separate the intrinsic value of the state from the added value of a specific action, value decomposition is used:

$$
Q(s,a)=V(s)+\mathcal{A}(s,a)-\frac{1}{\|\mathcal{A}\|}\sum_{a^{\prime}}\mathcal{A}(s,a^{\prime})
$$

    where $V(s)$ is the best performance under state $s$ without considering specific actions, and $\mathcal{A}(s,a)$ is the additional value of taking action $a$.
8.  **Prioritized Experience Replay:** To accelerate learning, a priority experience replay mechanism is adopted. Interaction data is stored as tuples $(s_t, a_t, r_t, s_{t+1})$. Each tuple is prioritized based on the absolute value of its Time Difference (TD) error:

$$
\mathbf{p}=\|r_t+\gamma\text{max}_{a_{t+1}}Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\|+\epsilon
$$

    where $\mathbf{p}$ is the priority and $\epsilon$ is a small positive constant.
9.  **Gradient Descent:** Network parameters ($\theta$) are continuously adjusted using gradient descent to minimize the Mean Squared Error (MSE) between the predicted Q-value and the target Q-value:

$$
L(\theta)=\mathbb{E}\left[\left(y_t-Q(s_t,a_t;\theta)\right)^2\right]
$$

$$
y_t=r_t+\gamma\text{max}_{a^{\prime}}Q(s_{t+1},a^{\prime};\theta^{-})
$$

**Key Quantitative Results and Numbers:**

RTLIR improves LLM performance in various input scenarios and tasks, with an average accuracy increase of **6%**. Specific improvements across different LLMs and languages include:

*   **Qwen2-1.5B:**
    *   English: 5.15% ± 1.09% ↑
    *   Chinese: 7.45% ± 1.11% ↑
    *   Spanish: 5.75% ± 1.10% ↑
    *   German: 1.30% ± 1.10% ↑
    *   Japanese: 2.65% ± 1.11% ↑
    *   Korean: 1.75% ± 1.12% ↑
    *   Average: 3.25% ± 0.42% ↑
*   **Gemma-2B:**
    *   English: 15.00% ± 1.50% ↑
    *   Chinese: 0.20% ± 1.12% ↑
    *   Spanish: 6.70% ± 1.11% ↑
    *   German: 5.20% ± 1.12% ↑
    *   Japanese: -5.15% ± 1.11% ↓ (decrease)
    *   Korean: 0.15% ± 1.12% ↑
    *   Average: 2.51% ± 0.42% ↑
*   **Llama-2-7B:**
    *   English: 14.10% ± 1.05% ↑
    *   Chinese: 15.60% ± 1.09% ↑
    *   Spanish: 9.65% ± 1.08% ↑
    *   French: -2.30% ± 1.12% ↓ (decrease)
    *   German: 18.75% ± 1.08% ↑
    *   Japanese: 12.90% ± 1.11% ↑
    *   Korean: 8.55% ± 1.12% ↑
    *   Average: 11.03% ± 0.41% ↑
*   **vicuna-7B-v1.3:**
    *   English: 15.75% ± 1.06% ↑
    *   Chinese: 5.55% ± 1.11% ↑
    *   Spanish: 4.60% ± 1.11% ↑
    *   French: -2.40% ± 1.11% ↓ (decrease)
    *   German: 12.30% ± 1.08% ↑
    *   Japanese: 6.45% ± 1.11% ↑
    *   Korean: 3.30% ± 1.12% ↑
    *   Average: 6.51% ± 0.41% ↑
*   **Llama-3-8B:**
    *   English: 4.85% ± 1.06% ↑
    *   Chinese: 9.05% ± 1.11% ↑
    *   Spanish: 7.60% ± 1.09% ↑
    *   French: 5.00% ± 1.11% ↑
    *   German: 10.55% ± 1.08% ↑
    *   Japanese: 6.20% ± 1.11% ↑
    *   Korean: 2.25% ± 1.12% ↑
    *   Average: 6.48% ± 0.42% ↑

**Stated Limitations:**

*   **Full text removal case:** If the model's operation results in the removal of all text content ($\hat{v}_{seq} = \phi$), the original text information is retained by default. This situation is stated to be "extremely rare."
*   **Original text optimal case:** Even if the original text can be output correctly, RTLIR still performs input refinement operations to extract valuable learning information and further optimize the model's adaptability to specific tasks, rather than skipping intervention.
*   The paper mentions some instances where RTLIR led to a decrease in performance for certain language models and languages (e.g., French for Qwen2-1.5B, Gemma-2B, Llama-2-7B, vicuna-7B-v1.3, and Japanese for Gemma-2B).
