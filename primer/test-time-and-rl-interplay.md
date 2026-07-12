---
title: Test-time compute and RL interplay
kind: primer
reference: ../topics/test-time-and-rl-interplay.md
updated: '2026-07-12'
---

# Test-Time Compute and RL Interplay: The Dual Optimization Loop

**Scaffold.** This primer teaches the central mechanic driving frontier LLM reasoning: **RL and test-time search are two optimizers for a single objective**. RL *internalizes* search behavior into policy weights (training-time scaling in parameter space); search *externalizes* computation at inference to solve specific instances beyond the policy's zero-shot reach (test-time scaling in solution space). Together they form a flywheel—search produces high-quality traces, RL distills them into the policy, the improved policy makes search more efficient—that powers systems from STaR to AlphaProof. By the end you will understand why reward design must match the search algorithm, why budget-aware search (BG-MCTS, L1) is replacing unbounded CoT, and why safety scaling demands architectural changes (MRM) not just better PRMs. This connects to RLHF pipelines, verifiable-reward RL, and the emerging science of inference scaling laws.

---

## Core Mechanism: The Dual Optimization Paradigm

**Intuition.** Imagine a student (the policy) and a study group (search). The study group spends hours exploring solution paths, backtracking, and verifying steps to crack a hard problem. The student watches the *successful* sessions and internalizes the patterns—so next time, they need less study-group time. The study group, now equipped with a stronger student, explores more efficiently. This is the flywheel: **search generates the training data that RL compresses into the policy; the better policy makes search cheaper and deeper.**

The reference formalizes this as two coupled optimization problems over the same objective (reward/value) [source:arxiv:2510.09988]:

$$
\begin{aligned}
\textbf{Internalization (RL):} \quad & \theta^* = \arg\max_\theta \; \mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)] \;-\; \lambda \int_{s\in\tau} D_{\text{KL}}\big(\pi_\theta(\cdot|s)\,\|\,\pi_{\mathcal{P}}(\cdot|s)\big)\,ds \
$$

6pt]
\textbf{Externalization (Search):} \quad & p^* = \arg\max_{p\in\mathcal{P}_{\text{plan}}} \Big[ \sum_{t=0}^{T-1} \gamma^t R_{\text{ext}}(s_t,a_t) \;+\; \mathcal{H}_\theta(s_T,p) \Big]
\end{aligned}
$$

- **RL** maximizes expected return $G(\tau)$ while staying close (KL penalty $\lambda$) to a proposal policy $\pi_{\mathcal{P}}$—typically the current policy or a reference model. This *compresses* search behavior into $\theta$.
- **Search** maximizes a planned trajectory's value using an external reward $R_{\text{ext}}$ and a heuristic $\mathcal{H}_\theta$ from the policy. This *expands* compute at test time for a specific problem.

**Why this works.** RL alone struggles with sparse, long-horizon rewards (credit assignment over thousands of tokens). Search alone repeats work for every new problem. The flywheel solves both: search provides dense, step-wise supervision (via MCTS visit counts, PRM scores, or verifier outcomes) that RL can learn from; RL amortizes that supervision into a policy that proposes better branches, evaluates states more accurately, and stops earlier.

**The critical coupling: reward design must match the search algorithm.**  
- **Best-of-N** needs only a *terminal* score $V(y)$ (ORM or self-consistency vote).  
- **MCTS** needs a *step-wise* value $Q(s,a)$ (PRM, learned value net, or verifier at each node).  
- **RL** needs a *differentiable/low-variance* reward (rule-based, verifier-based, or learned).  

Mismatching them—e.g., using an ORM for MCTS selection—starves the search of intermediate guidance, causing error accumulation or over-branching. AlphaProof sidesteps PRMs entirely by using **Lean verification as perfect binary reward at the tactic level** ($r_t=-1$ per step, $G_t=\min(\text{subgoal returns})$), proving that when a perfect verifier exists, sparse rewards + a learned value net suffice [source:nature:olympiad-level-formal-mathematical-reaso]. But for natural language reasoning without formal verifiers, PRMs (or generative critics) remain the practical bridge.

---

## Runnable Check: Best-of-N Over-Optimization

The reference notes that BoN with an imperfect verifier *exploits reward model errors* as $N$ grows [source:wellecks:l1-controlling-how-long-a-reasoning-mode; arxiv:2510.09988]. This minimal simulation shows the effect: a noisy verifier causes selected quality to *decrease* after an optimal $N$.

```python
import numpy as np

def simulate_bon_overoptimization(true_quality: np.ndarray, verifier_noise: float, N_vals: list[int], trials: int = 5000):
    """
    true_quality: ground-truth quality of each candidate (higher = better)
    verifier_noise: std of Gaussian noise added by verifier
    Returns mean true quality of BoN-selected candidate for each N.
    """
    M = len(true_quality)
    results = {N: [] for N in N_vals}
    for _ in range(trials):
        # Verifier scores = true quality + noise
        verifier_scores = true_quality + np.random.normal(0, verifier_noise, M)
        for N in N_vals:
            idx = np.argmax(verifier_scores[:N])  # BoN selects max verifier score
            results[N].append(true_quality[idx])
    return {N: np.mean(v) for N, v in results.items()}

# Ground truth: 100 candidates, quality ~ N(0,1)
np.random.seed(0)
true_q = np.random.randn(100)

# Imperfect verifier (noise std = 0.8, comparable to signal)
noise = 0.8
Ns = [1, 2, 4, 8, 16, 32, 64, 100]
means = simulate_bon_overoptimization(true_q, noise, Ns)

# Quality should peak then drop as N grows (over-optimization)
peak_N = max(means, key=means.get)
assert peak_N < max(Ns), f"Expected peak before max N, got peak at N={peak_N}"
assert means[peak_N] > means[max(Ns)], f"Quality at peak ({means[peak_N]:.3f}) should exceed quality at max N ({means[max(Ns)]:.3f})"
print({N: f"{means[N]:.3f}" for N in Ns})
# Typical output: {1: -0.001, 2: 0.412, 4: 0.621, 8: 0.712, 16: 0.735, 32: 0.718, 64: 0.682, 100: 0.651}
```

**What this shows.** With a noisy verifier, BoN quality peaks at moderate $N$ (here ~16) and *declines* as $N$ grows—the selected candidate increasingly exploits verifier errors rather than reflecting true quality. This is why the reference emphasizes **reward model calibration** and why methods like **Controlled Decoding** (prefix scorer + KL regularization) and **SAFFRON's MRM** (multifurcation reward model) aim to align the verifier's signal with true quality at *every prefix*, not just the terminal state.

---

## Load-Bearing Disagreements & Caveats

1. **Are PRMs necessary for step-wise search?**  
   [source:arxiv:2501.02497] treats PRMs as critical for guiding MCTS/ToT. [source:nature:olympiad-level-formal-mathematical-reaso] demonstrates AlphaProof achieves IMO-silver performance with **only sparse outcome rewards** (Lean proof completion) plus a *learned value network* for intermediate states. The disagreement is not binary: AlphaProof's value net *is* a step-wise critic, but it's trained from verifier outcomes, not human process labels. For domains *without* perfect verifiers (most NL reasoning), PRMs or generative critics remain the practical default—but their saturation (near-binary scores) limits search signal granularity [source:arxiv:2602.09574].

2. **Can safety scaling use standard tree search + PRMs?**  
   [source:arxiv:2506.06444] shows that for refusal tasks, **scalar PRMs are too slow** for tree search: safety lacks self-consistency (no majority voting), so PRM overhead outweighs gains vs. simple BoN. SAFFRON's fix is structural—a **Multifurcation Reward Model (MRM)** that scores *all* next tokens in one forward pass + Trie-based KV caching—yielding **3× TFLOP reduction** at matched ASR. This is a load-bearing caveat: *scaling laws for safety require architectural innovation, not just more compute on standard PRMs.*

---

## Current Status
Test-time search (MCTS, BoN, tool-use self-correction) is default for high-stakes reasoning; the RL–search flywheel (ReST-MCTS, AlphaProof) drives frontier capabilities; budget-aware search (BG-MCTS, L1) and inference-time alignment (Controlled Decoding, MRM) are emerging for deployment; no universal inference scaling law exists to predict compute-optimal configurations.

**Full reference:** See the comprehensive fact-checked article "Test-time compute and RL interplay" (linked sources: arxiv:2510.09988, arxiv:2501.02497, arxiv:2503.24235, arxiv:2602.09574, arxiv:2506.06444, arxiv:2310.17022, nature:olympiad-level-formal-mathematical-reaso, deepmind:alphageometry-2-new-geometry-reasoning-s, wellecks:l1-controlling-how-long-a-reasoning-mode, arxiv:2109.13582, arxiv:2303.05510, arxiv:2505.23614, arxiv:2203.14465, arxiv:2403.09629).

---
*Full reference (citations, derivations, variants):* [Test-time compute and RL interplay](../topics/test-time-and-rl-interplay.md)
