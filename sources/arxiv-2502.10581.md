---
id: arxiv:2502.10581
type: paper
title: Do We Need to Verify Step by Step? Rethinking Process vs Outcome Supervision
url: https://arxiv.org/abs/2502.10581
retrieved: '2026-07-10'
maturity: comprehensive
topic: reward-modeling
---

**Core Problem**
The paper investigates the statistical necessity of process supervision versus outcome supervision in reinforcement learning (RL) for complex, multi-step reasoning tasks. Process supervision provides fine-grained, step-level reward signals, while outcome supervision only delivers cumulative rewards after trajectory completion. Conventional wisdom holds that outcome supervision is fundamentally more challenging due to trajectory-level coverage requirements, which has driven substantial investment in collecting step-by-step feedback. The authors formally ask whether this perceived statistical gap is inherent or merely a consequence of algorithmic design, aiming to resolve the theoretical relationship between the two paradigms.

**Methodological Recipe**
The authors propose a theoretical transformation pipeline that converts outcome-supervised data into a format compatible with process-supervised algorithms. The procedure operates as follows: (1) partition the outcome-supervised dataset into two disjoint subsets; (2) apply least-squares regression to the first subset to estimate a per-step reward function $r_h(s_h, a_h)$ from the available cumulative rewards; (3) impute the missing step-level rewards for the second subset using the learned reward model; and (4) feed the augmented dataset into any standard offline RL algorithm designed for process supervision. This transformation is mathematically enabled by a novel Change of Trajectory Measure Lemma, which bridges trajectory-level return distributions and step-level distribution shifts under Markov dynamics, allowing trajectory-level supervision to be analyzed through state-action-level concentrability.

**Key Formulas & Theoretical Machinery**
The analysis formalizes coverage conditions using the Markov Decision Process tuple $(\mathcal{S}, \mathcal{A}, H, P, r)$. The authors contrast trajectory concentrability, $C_{\text{traj}} = \sup_{\pi} \sup_{\tau} \frac{\rho^\pi(\tau)}{\rho^{\pi_{\text{ref}}}(\tau)}$, with state-action concentrability, $C_{\text{sa}} = \sup_{\pi} \sup_{h, s, a} \frac{\rho^\pi_h(s, a)}{\rho^{\pi_{\text{ref}}}_h(s, a)}$. The central technical contribution is the Change of Trajectory Measure Lemma, which establishes that trajectory-level distribution shifts can be bounded using only state-action concentrability:
$$\mathbb{E}_{\tau \sim \pi} \left[ \left( \sum_{h=1}^H f_h(s_h, a_h) \right)^2 \right] \leq C_{\text{sa}} \cdot \mathbb{E}_{\tau \sim \pi'} \left[ \left( \sum_{h=1}^H f_h(s_h, a_h) \right)^2 \right] + \text{poly}(H) \cdot \log(\cdot)$$
This inequality demonstrates that controlling the second moment of state-action values under a reference policy implicitly controls prefix and suffix variances across the trajectory, up to polynomial horizon factors. The authors further prove that for any policy, its advantage function serves as an optimal process reward model, whereas Q-functions may yield suboptimal results. In preference-based RL settings governed by the Bradley-Terry model, the transformation similarly maps trajectory-level preferences to per-step rewards.

**Quantitative Results & Bounds**
The theoretical analysis demonstrates that outcome supervision requires no more samples than process supervision up to polynomial factors in the horizon $H$. Specifically, the sample complexity scales with $C_{\text{sa}}$ rather than $C_{\text{traj}}$, which can be exponentially larger in practice. For instance, classical algorithms like Fitted Q-Iteration require $\Omega(H/\epsilon^2)$ samples to achieve $\epsilon$-optimality, and the proposed transformation preserves this scaling. In preference learning (e.g., Direct Preference Optimization), the shift from trajectory-level to state-action-level concentrability represents a potential exponential improvement in sample efficiency. Lower bound analysis confirms that $\Omega(1/\epsilon^2)$ samples are necessary even in simple bandit settings, aligning with the derived upper bounds and confirming statistical equivalence under coverage assumptions.

**Stated Limitations**
The equivalence holds strictly under standard data coverage assumptions, particularly the bounded state-action concentrability coefficient. The authors acknowledge that polynomial factors in the horizon $H$ remain in the theoretical bounds, noting that removing them is an open direction for future research. The primary analysis utilizes all-policy coverage for tractability, deferring single-policy coverage extensions to supplementary materials. Furthermore, the paper explicitly states that any empirically observed performance gaps between paradigms likely stem from algorithmic limitations or implementation constraints rather than inherent statistical disadvantages. Consequently, the findings suggest that outcome supervision remains statistically viable and computationally scalable when paired with appropriately designed algorithms that leverage state-action-level coverage.
