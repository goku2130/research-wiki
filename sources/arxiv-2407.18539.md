---
id: arxiv:2407.18539
type: paper
title: Variational Analysis of Generalized Games over Banach spaces
url: https://arxiv.org/abs/2407.18539
retrieved: '2026-07-11'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Summary: Variational Analysis of Generalized Games over Banach spaces

### Core Problem
The authors address the problem of finding equilibrium points for generalized games (abstract economies) defined over infinite-dimensional Banach spaces. Specifically, they focus on games where players' preference maps are **non-ordered**, meaning the associated weak preference relations are not required to be complete or transitive. Previous variational approaches to these games were restricted to finite-dimensional strategy spaces; this work extends the analysis to Banach spaces by reformulating the game as a quasi-variational inequality (QVI) problem.

### Method/Recipe
The authors employ a variational reformulation to prove the existence of equilibrium. The step-by-step approach is as follows:

1.  **Definition of Preference Maps**: A generalized game is defined as $\Gamma = (X_\nu, K_\nu, P_\nu)_{\nu \in \Lambda}$, where $X_\nu$ are Banach spaces, $K_\nu$ are feasible strategy maps, and $P_\nu$ are non-ordered preference maps.
2.  **Introduction of Mid-point Continuity**: To handle non-ordered preferences, the authors introduce "mid-point continuity" (both upper and lower), a weaker condition than standard continuity of preference relations.
3.  **Construction of the Normal Cone Operator**: A normal cone operator $N_P$ is defined to represent the preference map. For a set-valued map $P: X \times Y \rightrightarrows X$, the operator is defined as:

$$
N_P(x, y) = N_{P(x,y)}(x) = (P(x, y) - \{x\})^\circ
$$

4.  **Derivation of the Principal Operator**: Because the unit sphere is not weak compact in infinite-dimensional spaces, the authors use a **partition of unity technique** to construct a principal operator $F: X \times Y \rightrightarrows X^*$ that is norm-weak upper semi-continuous (u.s.c.) with non-empty, convex, weak compact values.
5.  **QVI Reformulation**: The generalized game is reformulated as a QVI problem. The global principal operator is defined as the product of individual operators:

$$
\mathcal{F}(x) = \prod_{\nu \in \Lambda} F_\nu(x)
$$

6.  **Equilibrium Verification**: The authors prove that any solution to the $QVI(\mathcal{F}, K)$ is an equilibrium for the generalized game $\Gamma$.

### Key Formulas
The **normal cone** of a set $C$ at point $x$ is defined as:

$$
N_C(x) = \begin{cases} \{x^* \in X^* \mid \langle x^*, y - x \rangle \leq 0 \text{ for all } y \in C\}, & \text{if } C \neq \emptyset \\ X^*, & \text{otherwise} \end{cases}
$$

The **Quasi-Variational Inequality (QVI)** problem $QVI(T, K)$ seeks to find $\bar{y} \in K(\bar{y})$ such that:

$$
\exists \bar{y}^* \in T(\bar{y}) \text{ satisfying } \langle \bar{y}^*, y - \bar{y} \rangle \geq 0, \forall y \in K(\bar{y})
$$

An **equilibrium** $\bar{x} \in \mathbb{S}(\Gamma)$ is defined by:

$$
\bar{x}_\nu \in K_\nu(\bar{x}) \text{ and } P_\nu(\bar{x}) \cap K_\nu(\bar{x}) = \emptyset \text{ for every } \nu \in \Lambda
$$

### Key Quantitative Results
*   **Theorem 5.1**: For a non-empty closed convex subset $K$ of a Banach space $X$, if $P: X \rightrightarrows X$ is a convex-valued mid-point continuous map where $x \notin P(x)$, then any $\bar{x} \in K$ that solves the variational inequality $VI(F, K)$ is a maximal element for $P$ over $K$.
*   **Theorem 5.3**: A vector $\bar{x}$ is an equilibrium for the generalized game $\Gamma$ if $\bar{x}$ solves $QVI(\mathcal{F}, K)$, provided $P_\nu$ is convex-valued mid-point continuous with $x_\nu \notin P_\nu(x)$ and $K_\nu$ has non-empty closed convex values.
*   **Theorem 5.4 (Existence)**: The set of equilibrium points $\mathbb{S}(\Gamma)$ is non-empty ($\mathbb{S}(\Gamma) \neq \emptyset$) if:
    1.  $P_\nu$ is convex-valued mid-point continuous and $x_\nu \notin P_\nu(x)$.
    2.  $K_\nu$ is a closed lower semi-continuous map with non-empty convex values.
    3.  $K_\nu(C)$ is relatively compact.

### Stated Limitations
The authors note that the construction of the principal operator $F$ in infinite-dimensional Banach spaces must differ from finite-dimensional cases. In finite dimensions, $F$ can be defined as the convex hull of the intersection of the normal cone and the unit sphere ($S_X$). However, this is not applicable in infinite dimensions because the sphere $S_{X^*}$ is not weak compact in $X^*$.
