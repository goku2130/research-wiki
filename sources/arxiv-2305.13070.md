---
id: arxiv:2305.13070
type: paper
title: Towards Preventing Value Lock-In
url: https://arxiv.org/abs/2305.13070
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-hacking
---

**Core Problem**
The study addresses the geometric characterization of the set $\mathcal{A}$ of all possible area $n$-tuples $(\mathcal{A}_1, \dots, \mathcal{A}_n)$ produced by partitioning a convex quadrilateral $ABCD$ with straight lines intersecting two opposite sides, $AB$ and $CD$. Given fixed ratio sequences $(p_1:\dots:p_n)$ and $(p'_1:\dots:p'_n)$ that divide these sides into consecutive segments, the objective is to describe $\mathcal{A}$ as the quadrilateral varies over all convex configurations in the plane, extending the analysis to infinitely many cuts.

**Method and Analytical Recipe**
The authors construct the solution through a systematic vector-algebraic procedure in $\mathbb{R}^n$ (or $\ell^1$ for infinite partitions):
1. Place division points $A_i$ on $AB$ and $D_i$ on $CD$ according to the prescribed ratio tuples.
2. Distinguish two geometric configurations: parallel opposite sides ($AB \parallel CD$) versus intersecting sides meeting at an external point $E$.
3. For the intersecting case, introduce auxiliary scaling parameters $(sp'_0, sp_0, s)$ and define basis vectors $\vec{v}_1 = (p_1, \dots, p_n)$, $\vec{v}_2 = (p'_1, \dots, p'_n)$, and composite vectors $\vec{v}_3, \vec{v}_4$ whose components aggregate partial sums of the ratios.
4. Express each segment area as $\mathcal{A}_i = s(p_0 p'_i + p'_0 p_i + s_i)$ or $\mathcal{A}_i = s(p_0 p'_i + p'_0 p_i + t_i)$, depending on whether vertex $A$ or $B$ lies between $E$ and the opposite vertex, where

$$
s_i = -p_i p'_i + p_i \sum_{j=1}^i p'_j + p'_i \sum_{j=1}^i p_j, \quad t_i = -p_i p'_i + p_i \sum_{j=i}^n p'_j + p'_i \sum_{j=i}^n p_j.
$$

5. Compute discriminants $\Delta$ (for $n=3$) or $\Delta_i$ (for $n \ge 4$) to determine the rank of the linear transformation matrices mapping the parameter space to $\mathbb{R}^n$.
6. Classify the attainable set $\mathcal{A}$ based on whether the discriminants vanish, yielding trihedral angles, planar angles, or specific rays.
7. For infinite partitions, verify absolute summability and reduce the high-dimensional problem to $\mathbb{R}^3$ or $\mathbb{R}^2$ by aggregating tail segments and enforcing hyperplane constraints.

**Key Formulas and Discriminants**
For $n=3$, the primary discriminant is

$$
\Delta = (p_1 + p_2 + p_3)p'_1p'_3p_2 - (p'_1 + p'_2 + p'_3)p_1p_3p'_2.
$$

When $\Delta = 0$, attainable points lie on a plane $\alpha$ defined by

$$
(p_2p'_3 - p_3p'_2)x_1 + (p_3p'_1 - p_1p'_3)x_2 + (p_1p'_2 - p_2p'_1)x_3 = 0,
$$

or, if $p_1:p'_1 = p_3:p'_3$, by

$$
\frac{p_2 + p_3}{p_1}x_1 - \frac{p_1 + 2p_2 + p_3}{p_2}x_2 + \frac{p_1 + p_2}{p_3}x_3 = 0.
$$

For $n \ge 4$, the discriminants generalize to

$$
\Delta_i = (p_{i-1} + p_i + p_{i+1}) p'_{i-1} p'_{i+1} p_i - (p'_{i-1} + p'_i + p'_{i+1}) p_{i-1} p_{i+1} p'_i, \quad 2 \le i \le n-1.
$$

The subspace containing $\mathcal{A}$ is characterized by linearly independent hyperplanes $\alpha_i$ or $\beta_i$ derived from these determinants.

**Quantitative Results**
Proposition 1 establishes that if $\Delta \neq 0$, $\mathcal{A}$ comprises two nonintersecting open trihedral angles sharing a face and an open ray on that face. If $\Delta = 0$, $\mathcal{A}$ collapses to a single open angle in plane $\alpha$. For parallel opposite sides, the areas satisfy the strict proportionality $\mathcal{A}_1 : \mathcal{A}_2 : \mathcal{A}_3 = (p_1 + p'_1) : (p_2 + p'_2) : (p_3 + p'_3)$. In the infinite case ($p, p' \in \ell^1$), Proposition 5 confirms the identical trihedral or planar structure. When $p = p'$, Corollary 7 provides a closed-form recurrence for normalized areas $\tilde{x}_i = x_i/p_i$:

$$
\tilde{x}_i = \tilde{x}_1 + s_i(\tilde{x}_2 - \tilde{x}_1), \quad s_i = \frac{p_1 + p_i}{p_1 + p_2} + \frac{2}{p_1 + p_2} \sum_{j=2}^{i-1} p_j,
$$

constrained by the ratio bound $1 - \frac{p_1 + p_2}{p_1 + 2 \sum_{j=2}^{\infty} p_j} < \frac{\tilde{x}_2}{\tilde{x}_1} < 2 + \frac{p_2}{p_1}$.

**Limitations**
The framework strictly applies to convex quadrilaterals and cuts intersecting exactly two opposite sides. The infinite-segment analysis requires both ratio sequences to belong to $\ell^1$; otherwise, the attainable set is empty. The geometric classification bifurcates entirely based on the vanishing of $\Delta$ or $\Delta_i$, and the reduction to $\mathbb{R}^3$ or $\mathbb{R}^2$ assumes the ability to aggregate tail segments, which may obscure fine-grained distribution details for arbitrary $n$. The results presuppose familiarity with linear algebra and analytic geometry.
