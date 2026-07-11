---
id: arxiv:1809.02923
type: paper
title: Deep reinforcement learning in iterated amplification
url: https://arxiv.org/abs/1809.02923
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

**Core Problem**
The paper addresses one-dimensional stochastic convex optimization formulated as $\min_{\ell \leq x \leq u} H(x) = \mathbb{E}_{\xi}[h(x, \xi)]$. In many practical settings (e.g., censored inventory demand, customer preference surveys), the exact realization of the random variable $\xi$ and the objective value $h(x, \xi)$ are unobservable. The decision maker only receives comparative feedback: whether $\xi$ is greater than, less than, or equal to a chosen decision point. The central challenge is to construct an unbiased stochastic gradient estimate using solely these binary comparisons to drive an iterative optimization procedure.

**Method/Recipe Step by Step**
The authors propose the Comparison-Based Algorithm (CBA), executed as follows:
1. **Initialization:** Set iteration counter $t=1$, initial point $x_1 \in [\ell, u]$, step sizes $\{\eta_t\}$, maximum iterations $T$, and select density functions $f_-(x,z)$ and $f_+(x,z)$ satisfying support and integrability conditions (C1)–(C3).
2. **Primary Sampling & Comparison:** Draw $\xi_t$ from the underlying distribution. If $\xi_t = x_t$, resample until $\xi_t \neq x_t$. Record the sign of $\xi_t - x_t$.
3. **Auxiliary Sampling:**
