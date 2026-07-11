---
id: arxiv:2607.07435
type: paper
title: 'RLVP: Penalize the Path, Reward the Outcome'
url: https://arxiv.org/abs/2607.07435
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# RLVP: Penalize the Path, Reward the Outcome

### Core Problem
Agents operating in real-world, costly, and irreversible environments (e.g., placing phone calls) face two primary challenges that standard Reinforcement Learning from Verifiable Rewards (RLVR) cannot address. First, **deployability** depends on the *path* taken, not just the outcome; agents must respect outcome-neutral constraints (e.g., business hours, authentication) that outcome-based rewards cannot express. Second, these agents must learn **online** with high sample efficiency. In group-relative RL, learning stalls in "all-fail" or "all-success" groups because the group-relative advantage—which is equivalent to within-group variance—collapses to zero.

### Method and Recipe
RLVP introduces a **verifiable path channel** that supplements the sparse outcome reward with a per-action signal. The total reward is defined as:

$$
R = O + \beta\Phi
$$

where $O$ is the outcome reward, $\beta$ is a scaling factor, and $\Phi$ is the process term. The authors explain that the group's reward variance, which drives the policy gradient, decomposes as:

$$
\text{Var}_{G}(R) = \text{Var}_{G}(O) + \beta^{2}\text{Var}_{G}(\Phi) + 2\beta\text{Cov}_{G}(O, \Phi)
$$

When $\text{Var}_{G}(O) = 0$ (all-fail/all-success), learning depends entirely on the variance of the process signal $\text{Var}_{G}(\Phi)$.

#### 1. Penalizing the Path (for Deployability)
To enforce constraints, RLVP uses a deterministic rule engine to attach a penalty $-\lambda$ to specific bad actions. To avoid the "inaction trap" (where the agent stops acting to avoid penalties), the authors provide four design rules:
1. **Penalize commission, not omission:** Target specific verifiable bad actions, not a general lack of progress.
2. **Maintain outcome primacy:** The outcome reward must remain the primary driver to compel the agent to keep acting.
3. **Pair with fulfillment credits:** Every penalty should be paired with a credit $+\beta$ for the corresponding compliant action.
4. **Ensure reachability and un-gameability:** Use a small number of scripted compliant demonstrations to ensure the desired behavior is reachable early in training and use verifiable predicates rather than learned proxies.

#### 2. Rewarding Verified Progress (for Sample Efficiency)
The same channel can be used as a dense **potential** by paying a credit $+\beta$ for verified progress (e.g., reducing remaining goals in a proof). This is "reachability-gated"; it only provides a gradient if the policy can actually reach intermediate states with differing levels of progress.

### Key Quantitative Results
*   **Harm Reduction (TerminalBench, Qwen3-4B):** RLVP reduced harmful actions roughly sixfold compared to outcome-only training. Violations per episode dropped from $3.71 \pm 0.52$ to $0.66 \pm 0.63$, while task success remained statistically equal ($0.122 \pm 0.076$ for outcome-only vs. $0.097 \pm 0.060$ for RLVP).
*   **Sample Efficiency (miniF2F Algebra):**
    *   **4B Scale:** The aligned potential reached a 0.9 success threshold in $4.4 \pm 0.5$ iterations, compared to $7.0 \pm 0.7$ for outcome-only.
    *   **30B Scale:** The aligned potential reached the threshold in $5.4 \pm 1.0$ iterations. Outcome-only training using AdamW was significantly slower, taking $19.2 \pm 1.9$ iterations ($\approx 3.6\times$ slower).
*   **Learning at 0% Success (SWE-smith, 8B):** In a regime where the agent solved zero instances, outcome-only RL remained flat (productive actions $\approx 1.5$). The process reward drove a steady increase in productive actions to between $6.5$ and $11.4$ across seeds.

### Stated Limitations
*   **Success Rate Regime:** The strongest harm reduction results were obtained when task success was near the floor.
*   **Manual Specification:** Penalizable targets were manually identified based on domain rules rather than automated discovery.
*   **Scale Variance:** The "inaction trap" sweep showed high variance at larger model scales.
*   **Reachability Gate:** Dense potentials are useless if the base policy cannot reach intermediate states; for example, in software repair, the potential was vacuous because all rollouts scored $\Phi=0$.
