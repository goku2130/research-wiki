---
id: arxiv:2404.18922
type: paper
title: 'DPO Meets PPO: Reinforced Token Optimization for RLHF'
url: https://arxiv.org/html/2404.18922v4
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

**Core Problem**
Classical Reinforcement Learning from Human Feedback (RLHF) formulates alignment as a contextual dueling bandit, assigning a single sentence-level reward to evaluate the entire response. This creates a fundamental mismatch with Proximal Policy Optimization (PPO), which is designed for multi-step Markov decision processes (MDPs) requiring dense, step-wise feedback. In standard open-source implementations, the sparse sentence reward is distributed only to the final token, while intermediate tokens receive zero learned reward. This structural misalignment causes training instability, sample inefficiency, and sub-optimal alignment, as PPO cannot effectively leverage the autoregressive, token-by-token nature of large language model generation.

**Method & Step-by-Step Recipe**
The authors introduce Reinforced Token Optimization (RTO), which reformulates RLHF as a token-wise MDP and bridges Direct Preference Optimization (DPO) with PPO. The practical implementation follows a structured two-phase recipe:
1. **Offline Preference Processing:** Train a DPO model on an offline preference dataset $\mathcal{D}$ containing prompt-response pairs $(x, y^w, y^l)$ to obtain the policy $\pi_{\text{dpo}}$. This step implicitly extracts a token-level reward signal without training an explicit reward model.
2. **Dense Reward Construction:** During generation, compute a token-wise reward $r_{\text{rto}}$ for each step $h$. The reward combines the DPO-derived implicit signal with a KL-regularization term relative to a reference policy $\pi_{\text{ref}}$. At the terminal token ($h=H$), a sentence-level MLE reward $r_{\text{MLE}}$ is appended to penalize extreme lengths.
3. **PPO Policy Update:** Use the constructed dense reward $r_{\text{rto}}$ to perform standard PPO updates on the current policy $\pi$. Because every token receives a non-trivial reward signal, PPO optimizes the policy with fine-grained feedback rather than sparse terminal signals.

**Key Mathematical Formulations**
The preference model under the MDP formulation is expressed as:
$$\mathbb{P}(\tau^1 \succ \tau^2) = \sigma \left( \sum_{
