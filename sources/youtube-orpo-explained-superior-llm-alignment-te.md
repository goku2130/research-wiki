---
id: youtube:orpo-explained-superior-llm-alignment-te
type: web
title: 'ORPO Explained: Superior LLM Alignment Technique vs. DPO/RLHF'
url: https://www.youtube.com/watch?v=32S_xBt2IXw
retrieved: '2026-07-12'
maturity: comprehensive
topic: dpo-and-preference-optimization
---

The video "DeepSeek's GRPO (Group Relative Policy Optimization) | Reinforcement Learning for LLMs" by Julia Turc explains Group Relative Policy Optimization (GRPO), a reinforcement learning (RL) algorithm used in Large Language Model (LLM) training, particularly by DeepSeek for reasoning models. GRPO is presented as a more efficient alternative to Proximal Policy Optimization (PPO) and Reinforcement Learning with Human Feedback (RLHF).

**Core Problem:**
The core problem GRPO addresses is the efficient fine-tuning of LLMs for reasoning tasks without explicit reference answers, leveraging implicit signals like binary correctness. Traditional RL methods like PPO, especially in the context of RLHF, involve training an additional reward model and a separate critic model (value function), which adds significant computational overhead and cognitive load for practitioners. In scenarios where intermediate rewards are scarce or unreliable, such as in language tasks, the complexity of these methods may not be justified. GRPO aims to simplify this process while maintaining or improving performance, particularly for tasks with a clear "correct" answer.

**Method/Recipe Step by Step:**

1.  **Pre-training:** Start with a randomly initialized model and pre-train it by predicting the next token on a large text corpus. This builds world knowledge.
2.  **Instruction Fine-tuning:** Further fine-tune the model using instruction-response pairs to make it more conversational.
3.  **Reinforcement Learning (Reasoning Fine-tuning):** Apply GRPO to train the model using implicit signals, specifically a binary correctness signal for reasoning tasks (e.g., math problems).
    *   **Agent (LLM) Interaction:** The LLM (policy $\pi_\theta$) acts as the agent, predicting tokens (actions $a_t$) given the current state ($s_t$, which includes the prompt and previously generated tokens).
    *   **Environment Feedback:** The environment (external world, including verification heuristics) provides a reward ($R$) at the end of an episode (e.g., after the LLM generates a complete answer). For reasoning tasks, this reward is typically binary (1 for correct, 0 for incorrect) based on deterministic verification.
    *   **Trajectory Collection:** A trajectory ($\tau$) consists of all states visited and actions performed during one episode.
    *   **Baseline Calculation (GRPO specific):** For a given instruction, GRPO samples an additional group of trajectories (e.g., 4 to 8) and collects their final rewards. The baseline ($B$) is the average reward of this group.
    *   **Advantage Calculation:** The advantage ($A_t$) for each action is calculated as the final reward ($R$) minus the mean reward of the group ($B$), normalized by the standard deviation of the group's rewards.
    *   **Loss Calculation (GRPO specific):** GRPO uses a modified PPO loss. It calculates the PPO loss and then subtracts an additional regularization term based on the KL Divergence between the current policy ($\pi_\theta$) and a reference policy ($\pi_{ref}$). The reference policy is the state of the model before the reasoning fine-tuning stage.
    *   **Policy Update:** Perform gradient ascent on the GRPO loss to update the model parameters ($\theta$), increasing the probability of "good" actions and decreasing the probability of "bad" actions.

**Key Formulas (in LaTeX):**

*   **Policy Gradient Update (General):**

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) R
$$

    where $\theta$ are model parameters, $\alpha$ is the learning rate, $\pi_\theta(a_t | s_t)$ is the probability of action $a_t$ in state $s_t$ under policy $\pi_\theta$, and $R$ is the reward.

*   **Return with Discount Factor:**

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}
$$

    where $G_t$ is the return for action $a_t$, $R_{t+k+1}$ is the reward at time step $t+k+1$, and $\gamma$ is the discount factor ($0 \le \gamma \le 1$).

*   **GRPO Advantage:**

$$
A_t = \frac{R - \text{mean}(R_{\text{group}})}{\text{std}(R_{\text{group}})}
$$

    where $R$ is the final reward of the current trajectory, and $\text{mean}(R_{\text{group}})$ and $\text{std}(R_{\text{group}})$ are the mean and standard deviation of rewards from the sampled group of trajectories.

*   **PPO Loss (Clipped Surrogate Objective):**

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

    where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the ratio of new to old probabilities, $\hat{A}_t$ is the advantage estimate, and $\epsilon$ is a hyperparameter for clipping.

*   **GRPO Loss:**

$$
L^{GRPO}(\theta) = L^{CLIP}(\theta) - \beta \text{KL}(\pi_\theta || \pi_{\text{ref}})
$$

    where $L^{CLIP}(\theta)$ is the PPO clipped loss, $\beta$ is a coefficient, and $\text{KL}(\pi_\theta || \pi_{\text{ref}})$ is the KL Divergence between the current policy $\pi_\theta$ and the reference policy $\pi_{\text{ref}}$.

**Key Quantitative Results and Numbers:**

*   DeepSeek's R1 model, which uses GRPO, was allegedly trained "much more cheaply" than competitors like OpenAI's 01 model, while being a "serious competitor."
*   Experiments with "process supervision" (intermediate rewards) showed "a little bit of Delta" on the GSM 8K dataset (a middle school math questions dataset) compared to "outcome supervision" (single final reward). However, the video states that these small gains "don't justify the overhead."

**Stated Limitations:**

*   **Computational Overhead of Actor-Critic Methods:** Actor-critic methods (like PPO) require training an additional critic model (value function), which doubles memory requirements and computational resources.
*   **Cognitive Load of Actor-Critic Methods:** These algorithms are "just harder to understand" for practitioners.
*   **Inaccuracy of Intermediate Rewards:** In language domains, intermediate rewards, if available, often come from reward models which "are inaccurate anyway."
*   **Variance in Policy Gradient Methods:** Policy gradient methods can suffer from variance, where large updates based on noisy data can lead to model degeneration. GRPO addresses this by simplifying the baseline and adding a regularization term.
*   **Oversimplifications in GRPO:** GRPO makes simplifying assumptions, such as removing the state value function and using a simple group-based baseline, which could introduce more variance compared to PPO. The KL divergence regularization term is included to compensate for these oversimplifications.
