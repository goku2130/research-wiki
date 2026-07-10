---
title: Deriving the DPO Loss from First Principles
url: https://huggingface.co/blog/garg-aayush/derive-dpo-loss
retrieved: '2026-07-10'
topic: dpo-and-preference-optimization
---

- The DPO loss is a binary cross-entropy objective: L_DPO = -log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x))).
- This formulation avoids the need for sampling from the LM during fine-tuning, making it computationally lightweight and stable compared to PPO-based RLHF.
- The Bradley-Terry model depends only on the difference of rewards, meaning adding any function f(x) that depends only on the prompt does not change the preference probabilities.
