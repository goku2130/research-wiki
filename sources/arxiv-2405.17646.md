---
id: arxiv:2405.17646
type: paper
title: 'SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks'
url: https://arxiv.org/abs/2405.17646
retrieved: '2026-07-12'
maturity: comprehensive
topic: mdp-formulation
---

# Gap Between the Number of Facets of Order and Chain Polytopes

This research investigates the difference in the number of facets between two polytopes associated with a finite poset $P$ of $n$ elements: the order polytope $\mathcal{O}(P)$ and the chain polytope $\mathcal{C}(P)$. While these polytopes share the same Ehrhart polynomial and number of vertices, the number of facets of the order polytope is always less than or equal to those of the chain polytope. The authors define the "gap" of a poset $P$ as:

$$
\text{gap}(P) = f_{n-1}(\mathcal{C}(P)) - f_{n-1}(\mathcal{O}(P))
$$

### Method and Facet Calculation
To determine the gap, the authors utilize a known lemma to calculate the number of facets ($f_{n-1}$) for each polytope based on the properties of the poset $P$:
*   **Order Polytope:** $f_{n-1}(\mathcal{O}(P)) = \max(P) + \min(P) + \text{edges}(P)$, where $\max(P)$ and $\min(P)$ are the numbers of maximal and minimal elements, and $\text{edges}(P)$ is the number of edges in the Hasse diagram.
*   **Chain Polytope:** $f_{n-1}(\mathcal{C}(P)) = c(P) + |P|$, where $c(P)$ is the number of maximal chains and $|P|$ is the number of elements.

The authors introduce the **crossing number** of an element $v \in P$ to bound this gap. The crossing number is defined as $(u(v)-1)(d(v)-1)$, where $u(v)$ is the number of maximal chains above $v$ and $d(v)$ is the number of maximal chains below $v$.

### Key Quantitative Results and Bounds
The central result of the paper is a bound on the gap relative to any maximal antichain $A$ of $P$:

$$
\sum_{v \in A} (u(v)-1)(d(v)-1) \leq \text{gap}(P) \leq \sum_{v \in A} (u(v)d(v)-1)
$$

Using this bound, the authors provide two primary classifications:

1.  **Gap = 0 (X-avoiding posets):** The gap is exactly zero if and only if the poset $P$ is "X-avoiding," meaning it does not contain an X-shaped subposet (five distinct elements $a, b, c, d, e$ such that $a, b \prec c \prec d, e$). This is equivalent to the condition that the crossing number of every element $v \in P$ is zero.
2.  **Gap = 1 (X-orchids):** The gap is exactly one if and only if $P$ is an "X-orchid." An X-orchid is defined as a poset containing a saturated chain $c_1 < \dots < c_k$ (the stalk) where $c_k$ is covered by exactly two elements and $c_1$ covers exactly two elements. In such posets, the elements of the saturated chain have a crossing number of one, while all other elements have a crossing number of zero.

### Limitations and Open Questions
The authors identify two main areas for further research:
*   **Upper Bound Refinement:** They question whether a tighter upper bound for Theorem 10 exists that would yield 0 for X-avoiding posets and 1 for X-orchids, which would simplify the classification proofs.
*   **Generalization:** They suggest exploring whether the notion of crossing numbers can provide bounds for the difference in facets between "order-chain polytopes," which are a class of polytopes that interpolate between the order and chain polytopes.
