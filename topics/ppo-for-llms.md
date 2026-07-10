---
title: PPO for LLM fine-tuning (RLHF)
updated: '2026-07-10'
sources:
- https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf
---

Proximal Policy Optimization (PPO) is utilized within the Reinforcement Learning from Human Feedback (RLHF) framework to align large language models with human preferences [S1]. This process optimizes a policy to maximize a reward signal while maintaining stability relative to a reference model [S1].

## Reward Model
The reward model is a neural network trained to quantify how well a model's output aligns with human preferences [S1]. This model is trained on human pairwise preferences utilizing the Bradley-Terry model [S1].

## RL Objective and KL Penalty
The reinforcement learning objective is designed to maximize the reward $R(x,y)$ while preventing the policy from deviating excessively from the starting point [S1]. This is achieved by incorporating a KL divergence penalty into the objective function: $\mathbb{E}[R(x,y) - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})]$ [S1]. The $\beta$ term controls the strength of the penalty, ensuring the current policy $\pi_\theta$ does not drift too far from the reference model $\pi_{\text{ref}}$ [S1].

## PPO Clipping
To maintain training stability, PPO employs a clipping mechanism [S1]. This mechanism restricts the magnitude of policy updates, ensuring they remain within a specific trust region to prevent destabilizing updates to the model [S1].

**Key takeaways**
* RLHF uses PPO to align LLMs with human preferences [S1].
* Reward models are neural networks trained on pairwise preferences via the Bradley-Terry model [S1].
* A KL penalty is used in the RL objective to prevent the policy from drifting from the reference model [S1].
* PPO clipping limits policy updates to a trust region for increased stability [S1].

## References
- [S1] [Training language models to follow instructions with human feedback (InstructGPT)](https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf)
