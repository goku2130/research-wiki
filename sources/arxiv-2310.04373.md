---
id: arxiv:2310.04373
type: paper
title: Confronting Reward Model Overoptimization with Constrained RLHF
url: https://arxiv.org/abs/2310.04373
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-model-overoptimization
---

The paper addresses reward model overoptimization in reinforcement learning from human feedback (RLHF), where maximizing a reward model (RM) beyond a critical threshold—termed a *proxy point*—paradoxically degrades ground-truth evaluation performance. This phenomenon, rooted in Goodhart’s Law, is exacerbated in composite RMs, which combine multiple component rewards to capture distinct aspects of text quality. The authors demonstrate that correlation between component RMs significantly shifts proxy points, making fixed weighting schemes brittle and highly dependent on training duration.

To mitigate this, the authors propose a constrained RLHF framework that dynamically modulates RM weights using Lagrange multipliers. The method proceeds sequentially. First, proxy points are identified by training agents to maximize individual RMs, collecting reward and evaluation scores, and fitting a degree-10 polynomial surface via kernel density estimation to locate the joint maximum. Alternatively, a gradient-free Nelder-Mead optimizer dynamically adjusts constraint thresholds during a single training run. Second, the alignment task is reformulated as a constrained Markov decision process (CMDP) where the negative KL divergence from the base policy $\pi_0$ serves as the task reward, and each component RM $r_i$ is constrained to a threshold $\theta_i$ at its proxy point:
$$\max_{\pi} v_0^{\pi} \quad \text{s.t.} \quad v_i^{\pi} \geq \theta_i, \ i = 1, \dots, N.$$
Lagrangian relaxation converts this into a min-max game:
$$\max_{\pi} \min_{\boldsymbol{\mu} \geq 0} v_0^{\pi} + \sum_{i=1}^N \mu_i (v_i^{\pi} - \theta_i).$$
The multipliers $\boldsymbol{\mu}$ are learned online via gradient descent-ascent. To stabilize training, multipliers are bounded using a sigmoid function, yielding mixed advantages:
$$A_{\mu}^{\pi}(s, a) = \left( N - \sum_{i=1}^{N} \sigma(\mu_i) \right) A_0^{\pi}(s, a) + \sum_{i=1}^{N} \sigma(\mu_i) A_i^{\pi}(s, a).$$
Practical enhancements include using low-momentum SGD for multiplier updates and substituting value estimates with sum-of-rewards-to-go for constraint violation calculations.

Experiments on dialogue generation using GPT-2 and the DailyDialog dataset employ two component RMs: METEOR ($r^{\text{met}}$) and intent classification ($r^{\text{int}}$). The identified proxy points are $\theta^*_{\text{meteor}} = 0.23$ and $\theta^*_{\text{intent}} = 0.48$. Training uses 128,000 steps with a batch size of 64, learning rate $10^{-6}$, discount factor $\gamma=0.99$, and GAE $\lambda=0.95$. Ground-truth performance is measured via a composite evaluation metric averaging lexical scores (SACREBLEU, ROUGE2, BLEU) and diversity scores (unique-3, vocab size, max length):
$$\text{eval\_score} = \frac{1}{2} \left( \frac{x_s + x_r + x_b}{3} + \frac{x_u + x_v + x_m}{3} \right).$$
Results show that constrained variants, particularly $\xi$-PPO (equality constraints) and $\mu$-PPO (inequality constraints), outperform standard PPO and PPO-SAT. $\xi$-PPO achieves the highest final evaluation score while successfully enforcing thresholds. The Nelder-Mead variant (NM-PPO) converges to optimal thresholds within a single 256,000-step run, significantly reducing computational overhead compared to multi-run hyperparameter searches.

The authors acknowledge several limitations. All methods require at least minimal access to ground-truth evaluation metrics to identify proxy points or validate constraints. Primal-dual policy optimization via gradient descent-ascent only guarantees convergence of averaged iterates to a saddle point, not the final policy or multipliers. The Nelder-Mead search may settle in local optima, and its efficacy depends on the size of the feasible threshold region. Furthermore, the approach is currently validated only on a two-RM dialogue setting; broader testing across domains and higher-dimensional composite RMs is required. Finally, the framework does not eliminate the need for external evaluation, leaving open the challenge of fully proxy-free alignment.
