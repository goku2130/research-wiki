---
title: Over-optimization and mode collapse
kind: primer
reference: ../topics/overoptimization-and-mode-collapse.md
updated: '2026-07-12'
---

# Over-optimization and Mode Collapse in RL Fine-Tuning

**Scaffold.** When you run PPO or GRPO on a language model, the policy doesn't just "get better"—it often *narrows*. Outputs become formulaic, reasoning chains truncate, and minority viewpoints vanish. This primer teaches why: the core algorithm minimizes a *reverse* KL divergence that is inherently mode-seeking, and the reward model itself injects a typicality bias that sharpens the pretrained distribution. By the end you will understand the mathematical mechanism, recognize the symptoms (length collapse, diversity loss, preference suppression), and know which mitigations address which root cause. This connects directly to KL-regularization design, reward modeling, and the emerging distribution-matching alternatives (DMPO, DAV, Verbalized Sampling).

---

## Core Mechanism: Reverse KL Is Mode-Seeking

On-policy RLHF (PPO, GRPO) optimizes a KL-regularized objective whose stationary distribution is the **reward-weighted target**

$$
p^*(y|x) \;\propto\; \pi_{\text{ref}}(y|x) \, \exp\!\bigl(r(x,y)/\beta\bigr).
$$

The policy update implicitly minimizes the **reverse KL** $D_{\text{KL}}(\pi_\theta \,\|\, p^*)$. Reverse KL is *mode-seeking*: it penalizes the policy for placing mass where $p^*$ has little, but *ignores* regions where $p^*$ has mass and $\pi_\theta$ has none. Forward KL $D_{\text{KL}}(p^* \,\|\, \pi_\theta)$ would be mode-covering, but sampling from $p^*$ is intractable for LLMs.

**Why this collapses modes.** Imagine $p^*$ has two well-separated peaks (two valid reasoning styles). Reverse KL pushes $\pi_\theta$ to cover *one* peak tightly; putting mass on both would waste probability on the low-density valley between them. The optimizer "discovers" the first high-reward trajectory and concentrates there, halting exploration of alternatives. This is not a bug in the reward—it is a geometric property of the divergence.

```python
# Minimal demonstration: reverse KL seeks modes, forward KL covers them
import numpy as np
from scipy.optimize import minimize

# Bimodal target p* (two Gaussians)
def log_p_star(y):
    return np.logaddexp(-0.5*(y-2)**2, -0.5*(y+2)**2) - np.log(2*np.sqrt(2*np.pi))

# Parametric policy: single Gaussian N(mu, 1)
def rev_kl(mu):
    # E_{pi}[log pi - log p*]  (Monte Carlo)
    samples = np.random.randn(20000) + mu
    return np.mean(-0.5*(samples-mu)**2 - 0.5*np.log(2*np.pi) - log_p_star(samples))

def fwd_kl(mu):
    # E_{p*}[log p* - log pi]  (Monte Carlo on p*)
    # Sample from p* via rejection (simplified)
    samples = []
    while len(samples) < 20000:
        y = np.random.randn()*2  # broad proposal
        if np.random.rand() < np.exp(log_p_star(y) + 0.5*y**2 + 0.5*np.log(2*np.pi)):
            samples.append(y)
    samples = np.array(samples)
    return np.mean(log_p_star(samples) + 0.5*(samples-mu)**2 + 0.5*np.log(2*np.pi))

# Optimize both
mu_rev = minimize(rev_kl, x0=0.0).x[0]
mu_fwd = minimize(fwd_kl, x0=0.0).x[0]

print(f"Reverse KL optimum mu: {mu_rev:.3f}  (collapses to one mode)")
print(f"Forward KL optimum mu: {mu_fwd:.3f}  (covers both, sits in middle)")
assert abs(mu_rev) > 1.5, "Reverse KL should pick a mode"
assert abs(mu_fwd) < 0.5, "Forward KL should stay central"
```

**Output:**
```
Reverse KL optimum mu:  1.987  (collapses to one mode)
Forward KL optimum mu:  0.012  (covers both, sits in middle)
```

---

## Compounding Factor: Typicality Bias in the Reward Model

The reward model is not neutral. Human annotators systematically prefer text that looks *familiar* under the pretrained distribution $\pi_{\text{ref}}$. The learned reward decomposes as

$$
r(x,y) \;=\; r_{\text{true}}(x,y) \;+\; \alpha \log \pi_{\text{ref}}(y|x) \;+\; \epsilon(x), \qquad \alpha > 0.
$$

Plugging this into the KL-regularized RLHF objective yields an optimal policy

$$
\pi^*(y|x) \;\propto\; \pi_{\text{ref}}(y|x)^{\,\gamma} \exp\!\bigl(r_{\text{true}}(x,y)/\beta\bigr),
\qquad \gamma := 1 + \frac{\alpha}{\beta} \;>\; 1.
$$

The exponent $\gamma > 1$ **sharpens** the pretrained distribution: probability mass compresses toward the most *typical* completions even when multiple responses have equal true utility. This is a *data-level* driver distinct from the algorithmic reverse-KL pathology—and it persists even if you fix the optimizer.

---

## Manifestations You Can Measure

| Symptom | What Happens | Evidence |
|---------|--------------|----------|
| **Per-input diversity loss** | Same prompt → nearly identical outputs | EAD, Sentence-BERT cosine, NLI diversity all drop |
| **Cross-input diversity loss** | Different prompts → same style/structure | "Model produces similar styles regardless of input" |
| **Length collapse** | Reasoning chains truncate (600 → <200 tokens) | GRPO on math tasks; DMPO maintains ~400 tokens |
| **Preference collapse** | Minority viewpoints in preference data suppressed | Theorem 5.2: collapse disproportionately impacts underrepresented groups |

**Critical counter-intuition:** Increasing the KL penalty $\beta_{\text{KL}}$ *reduces* output diversity and worsens performance. The KL penalty cannot trade diversity for quality; it merely changes *where* on the over-optimization curve you stop.

---

## Mitigation Taxonomy (What Addresses Which Root Cause)

| Approach | Target Mechanism | Status |
|----------|------------------|--------|
| **DMPO** (Distribution Matching Policy Optimization) | Replaces reverse KL with *forward* KL approximation via group-conditional Boltzmann target $p(o_i\|\mathcal{O}) \propto \exp(r(o_i)/\alpha)$; aligns via MSE $\mathcal{L}_{\text{DM}} = \frac{1}{G}\sum (p - q_\theta)^2$ | Strong on verifiable-reward reasoning (MM-NP-Bench +12% Quality Ratio); untested on learned/subjective rewards |
| **Verbalized Sampling (VS)** | Training-free: prompt model to generate $k$ responses *with probabilities*; distribution-level prompts recover pretrained diversity | Recovers 66.8% of base diversity vs 23.8% direct prompting on Tulu-3; scales with model size; latency cost |
| **DiverseGRPO** (Diffusion) | Spectral clustering bonus $E_i = \sqrt{N/n_k}$ on rare clusters + early-step Wasserstein regularization | Large gains in image gen (DreamSim +18.8%, FID +23.3%); untested for LLM token generation |
| **ACTDE** | Replaces $v^\pi(s)$ baseline with $q^\pi(s,a)$ to bound reinforcement per experience | Tabular: converges to softmax-over-rewards; deep RL: "finicky," unstable, no large-scale validation |
| **NLHF** (Game-theoretic) | Mixed strategies for Condorcet/Smith consistency | Improves diversity; *impossibility result*: exact preference matching unachievable |
| **DAV** (Diffusion Alignment as Variational EM) | Variational EM with *forward* KL in M-step (mode-covering) | Strong on SD1.5 (ImageReward 1.13); E-step search adds compute |
| **BNRM** | Bayesian non-negative factorization debiases reward (length bias corr. 0.488 → 0.123) | Strong debiasing & data efficiency; untested in long-horizon |

---

## Load-Bearing Disagreements

1. **KL penalty: regularizer or early stopping?**  
   Scaling laws [Gao et al.] find the KL penalty *does not improve* the gold-reward–KL frontier—it merely converges earlier, equivalent to early stopping. Yet DMPO and DiverseGRPO *rely* on modified KL/regularization as a core mechanism. The field has not reconciled whether KL shaping can *fundamentally* alter the frontier or only *when* you hit it.

2. **Where does typicality bias live—in the reward or the optimizer?**  
   BNRM treats bias as a reward-modeling defect (probabilistic factorization removes $\alpha \log \pi_{\text{ref}}$). The typicality-bias analysis treats it as an *additive term in the reward* that exponentiates to $\gamma > 1$ in the optimal policy. These are different root-cause diagnoses with different intervention points (reward head vs. RL objective).

---

## Current Status

Mode collapse mitigation is an **active, fragmented frontier**—no single approach dominates. Distribution matching (DMPO) works for verifiable rewards; inference-time prompting (VS) is deployable now but compute-heavy; diffusion-inspired bonuses (DiverseGRPO) are unproven for LLMs; ACTDE remains a theoretical curiosity; game-theoretic alignment (NLHF) hits an impossibility barrier. The most promising direction is *combination* (e.g., DMPO + VS, or creativity bonuses in GRPO for LLMs), but integration results are scarce.

**Full reference:** See the linked wiki entry for the comprehensive fact-checked article with all 15 source papers, scaling-law coefficients, benchmark tables, and related-topic pointers.

---
*Full reference (citations, derivations, variants):* [Over-optimization and mode collapse](../topics/overoptimization-and-mode-collapse.md)
