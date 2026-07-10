---
title: 'Relative Preference Optimization: Enhancing LLM Alignment through Contrasting
  Responses'
url: https://arxiv.org/html/2402.10958
retrieved: '2026-07-10'
topic: dpo-and-preference-optimization
---

- IPO (Identity Preference Optimization) addresses the overfitting challenge in DPO, where the model might overfit to the preference dataset by pushing the reward gap too large.
- IPO uses a margin-based squared hinge loss: max(0, β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)) - 1)^2 to improve out-of-distribution generalization.
- ORPO (Odds Ratio Preference Optimization) combines the language modeling loss and preference optimization into a single monolithic objective without a reference model.
- KTO (Kahneman-Tversky Optimization) aligns models using individual samples based on Prospect Theory, eliminating the need for paired preference data.
