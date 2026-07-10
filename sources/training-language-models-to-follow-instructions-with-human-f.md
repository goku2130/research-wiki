---
title: Training language models to follow instructions with human feedback (InstructGPT)
url: https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf
retrieved: '2026-07-10'
topic: ppo-for-llms
---

- Introduces RLHF as a framework to align LLMs with human preferences using PPO.
- Defines the reward model as a neural network trained on human pairwise preferences using the Bradley-Terry model.
- Formulates the RL objective with a KL penalty: `E[R(x,y) - beta * KL[pi_theta || pi_ref]]` to prevent the policy from drifting too far from the reference model.
- Details the PPO clipping mechanism to limit policy updates within a trust region.
