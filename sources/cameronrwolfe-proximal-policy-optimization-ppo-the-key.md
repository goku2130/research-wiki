---
id: cameronrwolfe:proximal-policy-optimization-ppo-the-key
type: web
title: 'Proximal Policy Optimization (PPO): The Key to LLM Alignment (Overview)'
url: https://cameronrwolfe.substack.com/p/ppo-llm
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Proximal Policy Optimization (PPO) for LLM Alignment

### Core Problem
The primary objective in LLM post-training is aligning models with human preferences. While reinforcement learning (RL) is the primary tool for this, basic policy gradient methods—such as Vanilla Policy Gradient (VPG)—suffer from two critical flaws:
1. **High Variance:** Gradient estimates can be highly volatile, requiring a massive number of sampled trajectories to achieve accuracy, which is computationally expensive.
2. **Unstable Updates:** VPG lacks a mechanism to constrain the size of policy updates, meaning a single large, inaccurate update can destabilize and significantly harm the policy.

### Method and MDP Formulation
PPO is a policy gradient algorithm designed to reduce variance and enforce a "trust region" to limit how much a policy changes in a single update. For LLMs, PPO utilizes a **Markov Decision Process (MDP)** formulation rather than a bandit formulation.

#### Step-by-Step Recipe:
1. **Environment Mapping:** The RL framework is mapped to the LLM context:
    * **Policy ($\pi_\theta$):** The LLM itself.
    * **Initial State:** The input prompt.
    * **Action ($a_t$):** Each individual token predicted by the LLM.
    * **State ($s_t$):** The running completion (prompt plus all tokens generated up to time $t$).
    * **Trajectory:** The entire generated completion from the prompt to the stop token (e.g., `<eos>`).
    * **Reward:** A scalar value provided by a reward model or a verifier.
2. **Trajectory Sampling:** A batch of prompts is sampled, and the LLM generates completions based on its current stochastic policy.
3. **Reward Assignment:** Rewards are computed for these completions. The objective is to maximize the expected cumulative reward, often incorporating a discount factor $\gamma$ to prioritize immediate rewards over distant ones.
4. **Gradient Estimation:** The algorithm estimates the "policy gradient"—the gradient of the RL objective with respect to the policy parameters $\theta$. This is done by approximating the theoretical expectation with a sample mean across the generated trajectories.
5. **Policy Update:** Instead of computing the gradient directly, a loss function is formulated such that its gradient equals the policy gradient. Gradient ascent is then performed using automatic differentiation (e.g., via PyTorch's `loss.backward()`).

### Key Formulas
The policy's probability of taking an action is denoted as:

$$
\pi_\theta(a_t | s_t)
$$

To determine the relative utility of an action compared to the average, PPO utilizes the **Advantage Function**:

$$
A(s, a) = Q(s, a) - V(s)
$$

Where:
* $V(s)$ (**Value Function**): The expected cumulative reward starting from state $s$ following policy $\pi_\theta$.
* $Q(s, a)$ (**Action-Value Function**): The expected cumulative reward starting from state $s$, taking action $a$, and then following policy $\pi_\theta$.

### Quantitative Results and Status
The source does not provide specific benchmark numbers or percentage improvements. However, it notes that PPO served as the "default RL algorithm in LLM post-training for years" due to its effectiveness in alignment, maintaining a "long reign" despite the rapid pace of AI research. It is only recently that alternative algorithms, such as GRPO, have been adopted for specific tasks like LLM reasoning.

### Stated Limitations
* **Resource Intensity:** PPO carries high compute and memory overhead, which restricts experimentation to those with extensive computational resources.
* **Complexity:** The algorithm is described as complicated and "packed with nuanced implementation details."
* **Expertise Requirement:** Successful implementation requires both a deep theoretical understanding of the algorithm and substantial practical domain knowledge.
