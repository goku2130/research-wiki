---
id: arxiv:2510.08049
type: paper
title: 'A Survey of Process Reward Models: From Outcome Signals to Process Supervisions
  for Large Language Models'
url: https://arxiv.org/abs/2510.08049
retrieved: '2026-07-10'
maturity: comprehensive
topic: rl-for-reasoning
---

The survey identifies a fundamental misalignment in LLM reasoning alignment: conventional outcome reward models (ORMs) provide only a single, coarse signal for final answers, failing to capture stepwise progress, diagnose intermediate errors, or enable adaptive computation allocation. Process Reward Models (PRMs) resolve this by evaluating and steering reasoning at the step or trajectory level through an iterative closed loop.

The PRM development pipeline follows a three-stage recipe. First, process data is generated via human annotation (high-fidelity but resource-intensive), automated supervision (symbolic verification, Monte Carlo Tree Search, or self-evolution for scale), or semi-automated hybrid pipelines that blend curated seeds with automated expansion. Second, PRMs are built using four architectural paradigms: discriminative scoring, generative verification-then-judging, implicit inference from weak signals, and structural innovations like graph-based or hierarchical reward designs. Third, PRMs are deployed for test-time scaling (re-ranking, verification-guided decoding, and search) or reinforcement learning, where they replace sparse outcome signals with dense stepwise rewards for policy optimization.

Mathematically, discriminative PRMs map an input $x$ and partial solution $s$ to a scalar score $r_{\text{PRM}}(x, s) = f_{\text{PRM}}(x, s)$ (Eq. 1). Training employs pointwise losses such as binary cross-entropy $\mathcal{L}_{\text{BCE}} = -[y \log \sigma(r_{\text{PRM}}) + (1-y) \log (1-\sigma(r_{\text{PRM}}))]$ (Eq. 2) or MSE $\mathcal{L}_{\text{MSE}} = (y - \sigma(r_{\text{PRM}}))^2$ (Eq. 3), and pairwise preference losses $\mathcal{L}_{\text{pair}} = -\log \sigma(r_{\text{PRM}}(x, s_1) - r_{\text{PRM}}(x, s_2))$ (Eq. 5), where $p(s_1 \succ s_2) = \sigma(r_{\text{PRM}}(x, s_1) - r_{\text{PRM}}(x, s_2))$ (Eq. 4). Generative PRMs operate in two stages, producing a critique chain $v_{\text{critic}}$ before scoring: $r_{\text{PRM}}(x, s) = h_{\text{score}}(v_{\text{critic}}(x, s), s)$ (Eq. 6), optimized via $\mathcal{L} = \mathcal{L}_{\text{critic}}(v_{\text{critic}} \| v_{\text{ref}}) + \lambda \mathcal{L}_{\text{reward}}(r_{\text{PRM}} \| y)$ (Eq. 7). When leveraging answer logits, the reward is computed as $r_{\text{PRM}} = \frac{\exp(p_{\text{yes}})}{\exp(p_{\text{yes}}) + \exp(p_{\text{no}})}$ (Eq. 8).

Quantitatively, the survey references foundational datasets and scales, including PRM800K for human-validated multi-hop steps and medical reasoning pipelines that seed ~8,000 curated examples before automated MCTS expansion. The provided text focuses on methodological taxonomy and design spaces rather than reporting extensive empirical benchmark scores or performance deltas.

Stated limitations center on fidelity–scalability trade-offs and signal reliability. Human annotation remains prohibitively expensive and limited in scale. Automated pipelines, while enabling massive data generation, must carefully mitigate error propagation, verifier tool limitations, and potential misalignment with human reasoning preferences. Implicit PRMs reduce label dependency but rely on weaker outcome feedback or consistency constraints, risking reward mis-specification. Furthermore, test-time scaling incurs computational overhead from candidate sampling and PRM inference, while RL integration requires careful KL-constrained objectives to prevent reward hacking and ensure stable credit assignment across long trajectories.
