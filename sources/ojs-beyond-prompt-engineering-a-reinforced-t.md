---
id: ojs:beyond-prompt-engineering-a-reinforced-t
type: web
title: 'Beyond Prompt Engineering: A Reinforced Token-Level Input Refinement for Large
  Language Models'
url: https://ojs.aaai.org/index.php/AAAI/article/view/34586/36741
retrieved: '2026-07-12'
maturity: comprehensive
topic: mdp-formulation
---

The paper "Beyond Prompt Engineering: A Reinforced Token-Level Input Refinement for Large Language Models" introduces RTLIR, a method for optimizing Large Language Model (LLM) input by refining it at the token level using reinforcement learning.

**Core Problem:**
The quality of input data significantly impacts LLM performance. Traditional prompt engineering methods, while effective in some cases, are often context-dependent, rely on expert experience, and can introduce biases, limiting their applicability. Irrelevant information in input consumes computational resources, lowers processing efficiency, and can compromise output quality and reliability. The challenge is to develop an efficient, automated method to refine large-scale text data input, improving processing speed and output quality while preserving the original text structure.

**Method/Recipe Step-by-Step:**
RTLIR operates as a plug-and-play, LLM-agnostic module.
1.  **Preprocessing:** The input text (T) is first preprocessed into a sequence of token IDs ($ID_{seq} = Preprocess(T)$) and then converted into corresponding embedding vectors ($v_{seq} = \Phi_{embed}(ID_{seq})$). This establishes the initial state.
2.  **State Definition:** Each state ($s \in S$) encapsulates the current context of the text being processed. It integrates the embedding of a token and its corresponding action.
    $s = CONCATENATE[v_1, v_2, \dots, v_i, a_1, a_2, \dots, a_i]$
    where $v_i$ is the embedding vector of the $i$-th token, and $a_i$ is a binary flag (1 for keep, 0 for remove) for the $i$-th token. Initially, all $a_i$ are set to 1.
3.  **Action Definition:** Each action ($a \in A$) is a binary decision: either retain or delete the currently evaluated token ID.
4.  **Reinforcement Learning Agent:** The agent iteratively refines the input by evaluating and adjusting each token's retention.
    *   **Reward Design:** The agent learns through a combination of immediate and terminal rewards.
        *   **Immediate Reward:** Measures the impact of keeping or removing a token, guiding decisions based on similarity to a target.
        *   **Terminal Reward:** Evaluates the overall effect after the entire text processing is completed, considering the change in information quality of the entire text sequence. It encourages the agent to find processing paths closest to the ideal state.
        $r_t(s_t, a_t) = \frac{\log p(\hat{v}_{seq}, a_t) - \log p(v_{seq}, a_t)}{\log p(v_{seq}^*, a_t) - \log p(v_{seq}, a_t) + 1}$
        where $v_{seq}$, $\hat{v}_{seq}$, and $v_{seq}^*$ represent the original, input-refined, and optimal input-refined text sequence embeddings, respectively.
    *   **Policy Learning:** The agent optimizes its input refinement ability through policy learning to make optimal decisions.
        *   **Q-learning:** Estimates the expected utility of actions.
        $Q(s,a;\theta) = \mathbb{E}[r_t + \gamma \max_{a'} Q(s',a';\theta) \mid s,a]$
        where $\theta$ are network parameters, $r_t$ is the reward, and $\gamma$ is the discount factor.
        *   **Value Decomposition:** Separates the intrinsic value of the state from the added value of a specific action to improve Q-function estimation.
        $Q(s,a) = V(s) + \mathcal{A}(s,a) - \frac{1}{\|\mathcal{A}\|} \sum_{a'} \mathcal{A}(s,a')$
        where $V(s)$ is the best performance under state $s$, and $\mathcal{A}(s,a)$ is the additional value of taking action $a$.
    *   **Prioritized Experience Replay:** Stores interaction data as tuples $(s_t, a_t, r_t, s_{t+1})$ and prioritizes experiences based on the absolute value of their Time Difference (TD) error to accelerate learning.
        $p = \|r_t + \gamma \max_{a_{t+1}} Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\| + \epsilon$
        where $p$ is the priority and $\epsilon$ is a small positive constant.
    *   **Gradient Descent:** Adjusts network parameters $\theta$ using Mean Squared Error (MSE) between predicted and target Q values to maximize total reward.
        $L(\theta) = \mathbb{E}[(y_t - Q(s_t, a_t; \theta))^2]$
        $y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$
5.  **Output:** The process outputs the cleaned text, achieving effective input refinement.

**Key Quantitative Results and Numbers:**
RTLIR improves LLM performance in various input scenarios and tasks, with an average accuracy increase of 6%.
Specific improvements across different LLMs and languages:
*   **Qwen2-1.5B:**
    *   English: 5.15% increase (from 37.60% to 42.75%)
    *   Chinese: 7.45% increase (from 44.10% to 51.55%)
    *   Spanish: 5.75% increase (from 37.00% to 42.75%)
    *   German: 1.30% increase (from 41.05% to 42.35%)
    *   Japanese: 2.65% increase (from 51.05% to 53.65%)
    *   Korean: 1.75% increase (from 47.15% to 48.90%)
    *   Average: 3.25% increase (from 44.29% to 47.54%)
*   **Gemma-2B:**
    *   English: 15.00% increase (from 35.15% to 50.15%)
    *   Chinese: 0.20% increase (from 51.55% to 51.75%)
    *   Spanish: 6.70% increase (from 40.20% to 46.90%)
    *   German: 5.20% increase (from 46.30% to 51.50%)
    *   Japanese: 5.15% decrease (from 49.90% to 55.05%) - *Note: The table shows 5.15% decrease for Japanese, but the original text states 5.15% increase for English and 15.00% for Gemma-2B English. The table's "Origin" and "RTLIR" columns for Gemma-2B Japanese are 49.90% and 55.05% respectively, which is an increase of 5.15%. The table's "5.15±1.11%↓" seems to be a typo or miscalculation in the source for Japanese.*
    *   Korean: 0.15% increase (from 52.95% to 53.10%)
    *   Average: 2.51% increase (from 47.15% to 49.66%)
*   **Llama-2-7B:**
    *   English: 14.10% increase (from 31.25% to 45.35%)
    *   Chinese: 15.60% increase (from 39.70% to 55.30%)
    *   Spanish: 9.65% increase (from 35.70% to 45.35%)
    *   German: 18.75% increase (from 36.50% to 55.25%)
    *   Japanese: 12.90% increase (from 42.95% to 55.85%)
    *   Korean: 8.55% increase (from 46.65% to 55.20%)
    *   Average: 11.03% increase (from 40.03% to 51.06%)
*   **vicuna-7B-v1.3:**
    *   English: 15.75% increase (from 32.25% to 48.00%)
    *   Chinese: 5.55% increase (from 49.75% to 55.30%)
    *   Spanish: 4.60% increase (from 40.75% to 45.35%)
    *   French: 2.40% decrease (from 47.50% to 45.10%)
    *   German: 12.30% increase (from 42.95% to 55.25%)
    *   Japanese: 6.45% increase (from 49.40% to 55.85%)
    *   Korean: 3.30% increase (from 51.55% to 54.85%)
    *   Average: 6.51% increase (from 44.88% to 51.39%)
*   **Llama-3-8B:**
    *   English: 4.85% increase (from 31.95% to 36.80%)
    *   Chinese: 9.05% increase (from 40.55% to 49.50%)
    *   Spanish: 7.60% increase (from 34.95% to 42.55%)
    *   French: 5.00% increase (from 42.75% to 47.75%)
    *   German: 10.55% increase (from 34.00% to 44.55%)
    *   Japanese: 6.20% increase (from 44.95% to 51.15%)
    *   Korean: 2.25% increase (from 47.55% to 49.80%)
    *   Average: 6.48% increase (from 39.53% to 46.01%)

**Stated Limitations:**
*   **Full text removal case:** If the model's operation results in removing all text content ($\hat{v}_{seq} = \phi$), the original text information is retained by default. This situation is stated to be "extremely rare."
*   **Original text optimal case:** Even if the original text can be output correctly without intervention, RTLIR still performs input refinement operations to extract learning information and optimize adaptability. This implies a potential for unnecessary processing in already optimal cases, though it's framed as a learning opportunity.
