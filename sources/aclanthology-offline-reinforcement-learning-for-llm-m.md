---
id: aclanthology:offline-reinforcement-learning-for-llm-m
type: web
title: Offline Reinforcement Learning for LLM Multi-step Reasoning
url: https://aclanthology.org/2025.findings-acl.464.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

## OREO: Offline Reinforcement Learning for LLM Multi-step Reasoning

This paper introduces Offline Reasoning Optimization (OREO), an offline Reinforcement Learning (RL) algorithm designed to enhance the multi-step reasoning capabilities of Large Language Models (LLMs). OREO addresses limitations of existing offline RL methods like Direct Preference Optimization (DPO) in multi-step reasoning tasks, particularly the reliance on paired preference data and ineffective credit assignment for sparse rewards.

### Core Problem

The core problem OREO addresses is the difficulty of improving LLM multi-step reasoning using offline RL, specifically when:
1.  **Paired preference data is unavailable:** Traditional methods like DPO require pairwise comparisons of preferred and dispreferred responses, which are not naturally present in multi-step reasoning datasets that typically offer independent trajectories with sparse success/failure rewards.
2.  **Ineffective credit assignment for sparse rewards:** DPO treats all tokens uniformly, making it challenging to assign credit or blame to specific tokens or steps within a long reasoning chain when only a final, sparse reward (success or failure) is provided. This is crucial as the correctness of multi-step reasoning often hinges on a few key tokens.
3.  **Costly online data collection:** Many effective RL algorithms for LLMs (e.g., PPO) necessitate expensive online data generation or environmental interaction, limiting their practical application.

### Method/Recipe Step by Step

OREO builds on maximum entropy RL, specifically Path Consistency Learning (PCL), to jointly learn a policy model ($\pi_0$) and a value function ($V_0$) by optimizing the soft Bellman Equation.

1.  **MDP Definition:**
    *   **State ($s_t$):** For non-interactive tasks, $s_t = (x_0, \dots, x_L, y_0, \dots, y_{L-1})$, where $(x_0, \dots, x_L)$ is the input prompt and $(y_0, \dots, y_{L-1})$ is the generated token sequence up to step $t-1$. For interactive tasks, $s_{t+1}$ includes the next observation after an action.
    *   **Action ($a_t$):** Generating a new token.
    *   **Transition Function ($f$):** Deterministically updates the state, $s_{t+1} = f(s_t, a_t) = s_t \| a_t$ (concatenation) for non-interactive tasks.
    *   **Reward Function ($r(s_t, a_t)$):** Non-zero only at the terminal step ($T$), indicating the correctness of the reasoning chain or task success (sparse reward).
    *   **KL-regularization:** An entropy term is introduced to encourage exploration and keep the learned policy close to a reference policy ($\pi_{\text{ref}}$).

2.  **Learning Objective:** OREO optimizes the policy and value function by enforcing the telescoped version of the soft Bellman Equation, which connects the value of a state to the future rewards and policy entropy.
    *   **Value Network Loss ($\mathcal{L}_V(\phi)$):** Uses Mean Squared Error (MSE) to train a separate value network $V_0$ to predict the expected future KL-regularized reward.
    *   **Policy Objective ($\mathcal{L}_{\theta}(\theta)$):** Optimizes the policy model $\pi_0$ based on the learned value function and a regularization term.

3.  **Loss Variants:**
    *   **Token-level OREO (Standard):** The primary objective, where an action is a single token.
    *   **Step-level OREO:** An action is an entire reasoning step (e.g., multiple tokens). The probability of an action $a = (t_1 t_2 \dots t_n)$ is $\pi(\mathbf{a}) = \prod_{i=1}^{N} B_i\| t_i \| t_1 \dots t_n$.
    *   **Response-level OREO:** Mimics DPO by enforcing consistency only at the initial state, treating the entire response as a single action.

4.  **Iterative OREO:** When additional resources are available, OREO can be extended to a multi-iteration framework. After each training iteration, a new dataset is collected using the updated policy model to generate responses, which is then used for further training.

5.  **Test-Time Search with Value Function:** The learned value function can be leveraged to guide search at inference time.
    *   **Math Reasoning:** Step-level beam search is performed, where at each step, $B$ candidate partial trajectories are maintained. For each candidate, $B$ potential next reasoning steps are generated, and the $B$ resulting candidates with the highest values are retained.
    *   **Embodied Agent Control:** For tasks with unknown environment dynamics, $K$ actions are sampled at each step, and the action with the highest value is selected.

### Key Formulas in LaTeX

1.  **Optimal Policy Objective (KL-regularized):**

$$
\pi^* = \arg \max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T-1} \left( r(s_t, a_t) - \beta \log \frac{\pi(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} \right) \right]
$$

    (Note: The provided text's formula for $\pi^*$ appears to have a typo, missing the reward and KL terms. The above is a standard interpretation of KL-regularized RLHF objective.)

2.  **Value Function ($V^*(\pi)$):**

$$
V^*(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T-1} \left( r(s_t, a_t) - \beta \log \frac{\pi(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} \right) \right]
$$

    (Note: The provided text's formula for $V^*(\pi)$ also appears to have a typo. The above is a standard interpretation.)

3.  **Soft Bellman Equation (Theorem 1):**

$$
V^*(\mathbf{s}_t) - V^*(\mathbf{s}_{t+1}) = r_{\text{task}}(\mathbf{s}_t, \mathbf{a}_t) - \beta \log \frac{\pi^*(\mathbf{a}_t|\mathbf{s}_t)}{\pi_{\text{ref}}(\mathbf{a}_t|\mathbf{s}_t)}
$$

    (Derived from combining equations 14 and 15 in Appendix B, and applying the combined reward term.)

4.  **Telescoped Soft Bellman Equation (for OREO learning objective):**

$$
V_{\phi}(\mathbf{s}_t) = R_t - \beta \sum_{i=t}^{T-1} \log \frac{\pi_{\theta}(\mathbf{a}_i|\mathbf{s}_i)}{\pi_{\text{ref}}(\mathbf{a}_i|\mathbf{s}_i)}
$$

    where $R_t = \sum_{i=t}^{T-1} r(\mathbf{s}_i, \mathbf{a}_i)$.

5.  **Value Network Loss ($\mathcal{L}_V(\phi)$):**

$$
\mathcal{L}_V(\phi) = \frac{1}{T} \sum_{t=0}^{T-1} \left( V_{\phi}(\mathbf{s}_t) - R_t + \beta \sum_{i=t}^{T-1} \log \frac{\pi_{\theta}(\mathbf{a}_i|\mathbf{s}_i)}{\pi_{\text{ref}}(\mathbf{a}_i|\mathbf{s}_i)} \right)^2
$$

6.  **Policy Objective ($\mathcal{L}_{\theta}(\theta)$):**

$$
\mathcal{L}_{\theta}(\theta) = \frac{1}{T} \sum_{t=0}^{T-1} \left( V_{\phi}(\mathbf{s}_t) - R_t + \beta \log \frac{\pi_{\theta}(\mathbf{a}_t|\mathbf{s}_t)}{\pi_{\text{ref}}(\mathbf{a}_t|\mathbf{s}_t)} \right) + \text{sg} \left[ \beta \sum_{i=t+1}^{T-1} \log \frac{\pi_{\theta}(\mathbf{a}_i|\mathbf{s}_i)}{\pi_{\text{ref}}(\mathbf{a}_i|\mathbf{s}_i)} \right]^2 + \alpha \mathcal{L}_{\text{reg}}
$$

    where $\mathcal{L}_{\text{reg}} = \frac{1}{T} \sum_{t=0}^{T-1} \text{KL} \left[ \pi_{\theta}(\cdot|\mathbf{s}_t) \| \pi_{\text{ref}}(\cdot|\mathbf{s}_t) \right]$. (Note: The provided text's formula for $\mathcal{L}_{\theta}(\theta)$ has several typos, including $\pi_{\phi}(\mathbf{s}_t)$ instead of $\pi_{\theta}(\mathbf{a}_t|\mathbf{s}_t)$ and missing the KL divergence definition. The above is a corrected interpretation based on standard PCL objectives and the paper's context.)

7.  **Response-level OREO Objective ($\mathcal{L}_{\text{resp}}(\theta)$):**

$$
\mathcal{L}_{\text{resp}}(\theta) = \left( V_{\phi}(\mathbf{s}_0) - R_0 + \beta \sum_{i=0}^{T-1} \log \frac{\pi_{\theta}(\mathbf{a}_i|\mathbf{s}_i)}{\pi_{\text{ref}}(\mathbf{a}_i|\mathbf{s}_i)} \right)^2 + \alpha \mathcal{L}_{\text{reg}}
$$

    (Note: The provided text's formula for $\mathcal{L}_{\text{req}}(\theta)$ has typos. The above is a corrected interpretation.)

8.  **Explicit Advantage Function ($A_{\phi}$):**

$$
A_{\phi} = V_{\phi}(s_j) - V_{\phi}(s_i)
$$

9.  **Implicit Advantage Function ($A_{\theta}$):**

$$
A_{\theta} = \sum_{k=i}^{j-1} \beta \log \frac{\pi_{\theta}(\mathbf{a}_k|\mathbf{s}_k)}{\pi_{\text{ref}}(\mathbf{a}_k|\mathbf{s}_k)}
$$

    (Note: The provided text's formula for $A_{\theta}$ has typos. The above is a corrected interpretation based on the DPO derivation.)

### Key Quantitative Results and Numbers

*   **Mathematical Reasoning (Qwen-2.5-Math 1.5B):**
    *   OREO achieves **5.2% relative improvement** over SFT on GSM8K.
    *   OREO achieves **10.5% relative improvement** over SFT on MATH.
    *   OREO achieves **52.5% accuracy** on the MATH dataset using only the original training set (with a 1.5B model).
*   **Mathematical Reasoning (DeepSeekMath 7B):**
    *   OREO delivers **3.6% relative gain** on GSM8K.
    *   OREO delivers **5.1% relative gain** on MATH.
*   **Embodied Agent Control (ALFWorld, MiniCPM-2B-dpo-bf16):**
    *   OREO achieves a **17.7% relative improvement** over the baseline in unseen environments.
*   **Comparison of OREO variants (Qwen-2.5-Math 1.5B):**
    *   Response-level OREO accuracy: **74.5% on GSM8K** and **52.5% on MATH**.
    *   Token-level and Step-level OREO perform comparably, with step-level slightly outperforming on GSM8K.
*   **Iterative OREO:** Shows steady and consistent improvements over three iterations, outperforming baselines which show saturation.
*   **Test-Time Search with Value Function (MATH500 subset):**
    *   Beam search with $B=7$ provides an **11.4% relative improvement** in GSM8K.
    *   Beam search with $B=7$ provides a **17.9% relative improvement** in MATH.
    *   In ALFWorld, sampling 5 actions and choosing the one with the largest value gains significant improvement in success rates.

### Stated Limitations

*   **Computational Resources:** Due to limited computational resources, some experiments (ablation studies, iterative OREO, test-time search) were conducted using 1.5B models. Future work plans to scale experiments to larger models.
*   **Task Scope:** The method has primarily been evaluated on mathematical reasoning and embodied agent tasks. Future work aims to extend OREO to a wider variety of tasks, such as coding and web browsing, to assess its effectiveness in different domains.
