---
id: cameronrwolfe:proximal-policy-optimization-ppo-the-key
type: web
title: 'Proximal Policy Optimization (PPO): The Key to LLM Alignment (Cameron Wolfe)'
url: https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo
retrieved: '2026-07-11'
maturity: comprehensive
topic: ppo-for-llms
---

# Summary: Proximal Policy Optimization (PPO) for LLM Alignment

### Core Problem
The primary challenge addressed is **language model alignment**: ensuring that large language models (LLMs) generate text that is helpful, truthful, and non-toxic, rather than simply predicting the next token. While Supervised Fine-Tuning (SFT) is a starting point, it is insufficient for full alignment. 

Traditional reinforcement learning (RL) algorithms face significant hurdles in this context:
*   **Deep Q-Learning (DQL):** Effective for discrete action spaces but fails to generalize to continuous action spaces.
*   **Vanilla Policy Gradient (VPG):** Suffers from poor data efficiency and lack of robustness. Specifically, performing multiple optimization steps on the same trajectory often leads to "destructively large policy updates" that damage model performance.

### Method: RLHF and the Path to PPO
The source describes the implementation of **Reinforcement Learning from Human Feedback (RLHF)**, which utilizes PPO to align LLMs.

#### The RLHF Recipe
1.  **Data Generation:** Generate multiple outputs for a set of prompts using the LLM.
2.  **Human Ranking:** Human annotators rank or score these responses based on specific alignment criteria (e.g., avoiding hallucinations).
3.  **Reward Modeling:** Train a reward model to predict human preference scores based on the LLM's responses.
4.  **Policy Optimization:** Use PPO to finetune the LLM (the policy) to maximize the scores predicted by the reward model.

#### RL Framework for LLMs
To apply RL, the LLM process is mapped to RL components:
*   **Policy:** The language model.
*   **State:** The current textual input sequence.
*   **Action:** The next token predicted by the model.
*   **Reward:** The quality score provided by the reward model after a full sequence is produced.

#### Technical Evolution: TRPO to PPO
PPO is an extension of **Trust Region Policy Optimization (TRPO)**. TRPO improves upon VPG by finding the largest possible policy update that improves performance without damaging it. This is achieved by updating the policy under a constraint based on the **Kullback-Leibler (KL) Divergence**, which measures the distance between the old and updated policies.

The practical implementation of TRPO involves:
1.  Approximating the objective and constraint using a **Taylor expansion**.
2.  Solving the approximate objective via **Lagrangians and duality**.
3.  Utilizing the **conjugate gradient algorithm** to avoid the computational intractability of inverting large matrices.
4.  Applying post-processing to ensure the KL divergence constraint is met and the objective is improved.

PPO is presented as a more robust, simpler, and more data-efficient alternative to TRPO, making it the preferred choice for aligning models like InstructGPT and ChatGPT.

### Key Formulas
The source emphasizes the use of **KL Divergence** to compare two probability distributions, $p$ and $q$, and to penalize reward hacking or excessively large policy updates.

The KL divergence is formulated as the expected difference in log probabilities between two distributions:

$$
\text{KL}(p \parallel q) = \mathbb{E}_{x \sim p} [\log p(x) - \log q(x)]
$$

### Quantitative Results and Applications
While the source does not provide specific numerical benchmark improvements, it notes the following:
*   **Adoption:** PPO was the selected algorithm for the RLHF implementation used by OpenAI to align **InstructGPT** and **ChatGPT**.
*   **Utility:** PPO is described as satisfying four critical needs: general applicability (discrete and continuous), data efficiency, robustness (minimal hyperparameter tuning), and simplicity of implementation.

### Stated Limitations
*   **RLHF General Limitations:** The curation of human preference data is expensive, and RL can be data inefficient.
*   **DQL:** Limited to simple environments; fails in continuous action spaces.
*   **VPG:** Requires massive amounts of data to eliminate noise in policy gradient estimates.
*   **TRPO:** The analytical update rule is difficult to work with, requiring complex approximations (Taylor expansions and conjugate gradients) to be computationally feasible.
