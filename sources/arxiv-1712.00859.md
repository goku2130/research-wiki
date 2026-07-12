---
id: arxiv:1712.00859
type: paper
title: On the Geometry of Nash and Correlated Equilibria with Cumulative Prospect
  Theoretic Preferences
url: https://arxiv.org/abs/1712.00859
retrieved: '2026-07-12'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Summary: On the Geometry of Nash and Correlated Equilibria with Cumulative Prospect Theoretic Preferences

## Core Problem
In non-cooperative game theory under Expected Utility Theory (EUT), the set of correlated equilibria ($C_{EUT}$) is a convex polytope, and all Nash equilibria (NE) lie on its boundary (Property P). This paper investigates whether these geometric properties persist when players possess preferences based on Cumulative Prospect Theory (CPT), which accounts for behavioral features like reference points and non-linear probability weighting. Specifically, the authors examine the convexity, connectedness, and boundary properties of the set of CPT correlated equilibria ($C_{CPT}$) and CPT Nash equilibria.

## Method and Recipe
The authors extend the definitions of equilibrium to CPT and analyze the resulting sets using the following steps:

1.  **CPT Value Definition**: A prospect $L = \{(p_1, z_1); \dots; (p_t, z_t)\}$ is evaluated using a reference point $r$, a value function $v^r(\cdot)$, and probability weighting functions $w^\pm(\cdot)$. The CPT value $V^r(L)$ is calculated by ordering outcomes $z_{a_1} \geq z_{a_2} \geq \dots \geq z_{a_t}$ and applying decision weights $\pi_j^\pm$:

$$
V^r(L) := \sum_{j=1}^{j_r} \pi_j^+(p, a)v^r(z_{a_j}) + \sum_{j=j_r+1}^t \pi_j^-(p, a)v^r(z_{a_j})
$$

    where $\pi_j^+$ and $\pi_j^-$ are derived from the differences of $w^+$ and $w^-$ applied to cumulative probabilities.

2.  **Equilibrium Formulation**: 
    *   **CPT Correlated Equilibrium ($C_{CPT}$)**: A joint distribution $\mu$ is an equilibrium if for all players $i$ and strategies $s_i, d_i \in S_i$ where $\mu_i(s_i) > 0$:

$$
V_i^{r_i(\mu)}(L(\mu, s_i, s_i)) \geq V_i^{r_i(\mu)}(L(\mu, s_i, d_i))
$$

    *   **CPT Nash Equilibrium**: The intersection of $C_{CPT}$ and the set of product-form distributions $I$.

3.  **Boundary Analysis (Property P)**: To prove that CPT Nash equilibria lie on the boundary of $C_{CPT}$, the authors introduce the "regret" function:

$$
R^r(p, x, y) := V^r(p, x) - V^r(p, y)
$$

    They prove **Lemma 2.7**, stating that if two prospects $(p, x)$ and $(p, y)$ are not similarly ranked or neither dominates the other, there exists a direction $\delta$ such that $R^r(p + \epsilon \delta, x, y) < R^r(p, x, y)$. This is used to show that any completely mixed CPT Nash equilibrium can be perturbed into a distribution that violates the equilibrium conditions, placing it on the boundary.

4.  **$2 \times 2$ Game Characterization**: For $2 \times 2$ games with fixed reference points, the authors analyze the monotonicity of $R_1^{r_1}$ as a function of the marginal probability $p_0^1$. They show that the CPT equilibrium conditions reduce to linear inequalities of the form $\alpha \mu_{00} \leq \mu_{01}$, proving that $C_{CPT}$ is a convex polytope in this specific case.

## Key Quantitative Results
*   **Property (P)**: The authors formally establish that in any finite, non-trivial game, CPT Nash equilibria lie on the boundary of the set of CPT correlated equilibria.
*   **$2 \times 2$ Games**: $C_{CPT}$ is a convex polytope for $2 \times 2$ games with fixed reference points. These are classified into coordination, anti-coordination, and competitive games. Competitive games result in $C_{CPT}$ reducing to a single point (the unique mixed CPT Nash equilibrium).
*   **Disconnectedness**: Unlike EUT, $C_{CPT}$ can be disconnected. In **Example 4.2**, using Prelec's weighting function $w_i(p) = \exp\{-(-\ln p)^{\alpha_i}\}$ with $\alpha_1 = 0.5$ and $\alpha_2 = 1$, the authors demonstrate a game where $C(1, \text{TOP})$ is disconnected, and the resulting $C_{CPT}$ is also disconnected.

## Stated Limitations
*   **Non-Convexity**: $C_{CPT}$ is not guaranteed to be a convex polytope in general games.
*   **Connectedness**: $C_{CPT}$ can be disconnected, complicating the geometry compared to the EUT case.
*   **$2 \times 2$ Constraints**: The proof that $C_{CPT}$ is a convex polytope for $2 \times 2$ games requires the assumption of fixed reference points independent of the joint probability distribution $\mu$.
