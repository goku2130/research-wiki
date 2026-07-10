---
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model
  (HTML)'
url: https://arxiv.org/html/2305.18290v3
retrieved: '2026-07-10'
topic: dpo-and-preference-optimization
---

- The Bradley-Terry model is used to model the probability of preference: P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l)).
- The optimal policy π*(y|x) is derived as π_ref(y|x) exp(r*(x,y)/β) / Z(x).
- By substituting the closed-form policy back into the Bradley-Terry model, the reward difference cancels out the partition function Z(x), allowing the preference probability to be expressed directly as a function of the policy π_θ.
