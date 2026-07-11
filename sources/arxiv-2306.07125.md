---
id: arxiv:2306.07125
type: paper
title: 'H2O: Heavy-Hit Oracle for Generative Inference of Large Language Models'
url: https://arxiv.org/abs/2306.07125
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

This study investigates the mechanisms by which recurrent neural networks (RNNs) learn latent temporal features and how time-aware representations emerge during training. Addressing the scarcity of research on how artificial networks internalize time, the authors isolate temporal learning from long-term memory demands by introducing Temporal Flipflop (TF) automata. These are state-independent timed automata where state transitions depend on a hidden binary temporal variable, allowing direct control over time-awareness complexity.

The experimental recipe proceeds systematically. First, two temporal dependence structures are defined: periodic timing, where the temporal variable $\Theta(t)$ alternates based on a fixed period $P$ (simulating time-of-day phases), and relative timing, where $\Theta(t)$ activates when a clock exceeds a threshold $\tau$ since the last non-null input. Second, supervised datasets are constructed by sampling input sequences $u(t)$ uniformly from a binary alphabet $\Sigma=\{a,b\}$ and computing target outputs $y(t)$ via the automaton’s transition rule. Crucially, explicit time is withheld from the network; the RNN must infer temporal structure solely from input-output pairings. Third, Vanilla RNNs (with corroborating results from GRUs and LSTMs) are trained via stochastic gradient descent to minimize binary cross-entropy loss. Performance is continuously monitored using Time-Independent (TI) accuracy (on symbol $b$) and Time-Dependent (TD) accuracy (on symbol $a$). Finally, dynamical systems analysis is applied at each training iteration: principal component analysis (PCA) characterizes hidden state geometry, a fixed-point detection algorithm identifies input-dependent equilibria, and the spectral radius of the Jacobian matrix $J$ at these points tracks stability.

The core mathematical framework relies on the time-dependent transition rule:

$$
\Delta(c, s, t) = \delta_{\Theta(t)} = \begin{cases} \delta_0(c, s), & \Theta(t) = 0 \\ \delta_1(c, s), & \Theta(t) = 1 \end{cases}
$$

The RNN dynamics follow the discrete-time update:

$$
h(t + 1) = \tanh(W_{hh}h(t) + W_{uh}u(t) + b_h), \quad y(t) = \sigma(W_{hy}h(t) + b_y)
$$

Fixed-point stability is governed by the largest eigenvalue modulus $|\lambda_{\max}|$ of the Jacobian evaluated at the equilibrium; local stability requires $|\lambda_{\max}| < 1$.

Quantitatively, training on $T=60$ sequences with $N_h=32$ hidden units reveals a universal three-phase learning trajectory. Phase 1 spans over half the training duration, during which TI accuracy reaches perfection while TD accuracy stagnates near 50% (random guessing). Phase 2 triggers a rapid TD accuracy surge from $<65\%$ to $>90\%$, directly coinciding with a dynamical bifurcation where $|\lambda_{\max}|$ crosses unity. Across 15 trials, TD accuracy crosses the 75% threshold precisely at $|\lambda_{\max}| \approx 1$, confirming that fixed-point destabilization enables sustained periodic orbits. Post-training PCA reveals hidden states organizing into two ring-like manifolds encoding the previous input symbol and time modulo $P=10$. For relative-timing tasks ($\tau=5, p=0.2$), learning plateaus resolve when a second stable fixed point for the null symbol emerges at iteration $89 \pm 18$ (mean $\pm$ std over 15 seeds), with full convergence averaging $452 \pm 103$ iterations. Extending sequence length to $T=180$ increases the $|\lambda_{\max}|$ value at the 75% accuracy threshold, indicating stricter oscillation precision is required for longer horizons.

The authors acknowledge several limitations. The dynamical systems framework may not generalize to all sequence tasks, particularly those requiring state-dependent transitions. Preliminary experiments indicate that state-dependent timed automata impose longer temporal dependencies, rendering high-granularity fixed-point tracking computationally infeasible. Consequently, broader application of this analysis necessitates more efficient fixed-point tracking methodologies, such as modifying the learning process to stabilize equilibria. Despite these constraints, the work establishes a rigorous empirical link between training plateaus, gradient pathologies, and bifurcations in RNN dynamics.
