---
id: arxiv:2302.08702
type: paper
title: Variational Reformulation of Generalized Nash Equilibrium Problems with Non-ordered
  Preferences
url: https://arxiv.org/abs/2302.08702
retrieved: '2026-07-12'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Variational Reformulation of Generalized Nash Equilibrium Problems with Non-ordered Preferences

### Core Problem
Traditional variational reformulations of Generalized Nash Equilibrium Problems (GNEP) rely on representing player preferences as numerical objective functions. This representation is only possible if preferences are transitive and complete. However, in many real-world scenarios, preferences are non-ordered (incomplete and non-transitive). The core problem addressed in this work is to characterize GNEPs with non-ordered, non-convex, and interdependent preferences using variational inequalities (VI) and quasi-variational inequalities (QVI) without requiring numerical representations of preferences.

### Method and Recipe
The authors propose a reformulation that replaces objective function gradients with operators based on adapted normal cones.

**1. Definition of the Jointly Convex GNEP**
The game $\Gamma'$ is defined by a joint constraint set $\mathcal{X} \subseteq \mathbb{R}^n$ (non-empty, closed, and convex), where the feasible strategy set for player $i$ is:

$$
K_{i}(x)=\left\{z_{i}\in X_{i} \mid\left(x_{-i}, z_{i}\right) \in \mathcal{X}\right\}
$$

**2. Construction of the Variational Operator**
To avoid numerical utility functions, the authors define a multi-valued map $T$ using the following steps:
*   Define $\tilde{P}_i(x) = \text{co}(P_i(x))$, where $P_i$ is the preference map.
*   Utilize the normal cone $\mathcal{N}_P$ defined as:

$$
\mathcal{N}_{P}(x, y)=\left\{y^{*} \in \mathbb{R}^{p} \mid\left\langle y^{*}, z-y\right\rangle \leq 0, \forall z \in P(x, y)\right\}
$$

*   Construct the player-specific operator $T_i$:

$$
T_{i}\left(x_{-i}, x_{i}\right)=\text{co}\left(\mathcal{N}_{\tilde{P}_{i}}\left(x_{-i}, x_{i}\right) \cap S_{i}[0, 1]\right)
$$

    where $S_i[0, 1] = \{x \in \mathbb{R}^{n_i} \mid \|x\| = 1\}$.
*   The global operator is the product: $T(x) = \prod_{i \in \Lambda} T_i(x)$.

**3. Reformulation as VI and QVI**
*   **Jointly Convex Case:** The GNEP is reformulated as a Stampacchia VI problem $VI(T, \mathcal{X})$, seeking $\bar{x} \in \mathcal{X}$ such that there exists $\bar{x}^* \in T(\bar{x})$ satisfying $\langle \bar{x}^*, z - \bar{x} \rangle \geq 0$ for all $z \in \mathcal{X}$.
*   **General Case:** For non-convex interdependent preferences, the problem is reformulated as a QVI problem $QVI(T, K)$, seeking $\bar{x} \in K(\bar{x})$ such that there exists $\bar{x}^* \in T(\bar{x})$ satisfying $\langle \bar{x}^*, z - \bar{x} \rangle \geq 0$ for all $z \in K(\bar{x})$.

**4. Handling Unbounded Strategy Sets**
To ensure solvability when strategy maps are unbounded, the authors employ a coercivity criterion $(\mathcal{C}_x)$:
There exists $\rho_x > 0$ such that for all $y \in K(x) \setminus \bar{B}(0, \rho_x)$, there exists $z \in K(x)$ with $\|z\| < \|y\|$ and $z_i \in P_i(y)$ for all $i \in \Lambda$.

### Key Quantitative Results
*   **Theorem 3.1:** Any solution of $VI(T, \mathcal{X})$ is an equilibrium for the jointly convex GNEP $\Gamma'$.
*   **Theorem 3.2:** A solution for $\Gamma'$ exists if $P_i$ is lower semi-continuous (l.s.c.) with open upper sections, $x_i \notin \text{co}(P_i(x))$ for all $x \in \mathbb{R}^n$, and specific coercivity conditions are met.
*   **Theorem 3.3:** Any solution of $QVI(T, K)$ is an equilibrium for the GNEP $\Gamma$.
*   **Theorem 3.4:** An equilibrium for $\Gamma$ exists if $X_i$ is non-empty, closed, and convex; $K_i$ is closed and l.s.c. with non-empty convex values; $P_i$ is l.s.c. with open upper sections and $x_i \notin \text{co}(P_i(x))$; and the coercivity criterion $(\mathcal{C}_x)$ holds.
*   **Application:** The authors prove the existence of at least one competitive equilibrium for an Arrow-Debreu economy under uncertainty where preferences are non-convex, non-ordered, price-dependent, and interdependent.

### Stated Limitations
The results are contingent upon several technical assumptions:
1.  Preference maps $P_i$ must be lower semi-continuous and possess open upper sections.
2.  The condition $x_i \notin \text{co}(P_i(x))$ must hold for all $x$, ensuring that a player does not prefer their own current strategy.
3.  Existence in unbounded sets requires the satisfaction of the specific coercivity criterion $(\mathcal{C}_x)$.
