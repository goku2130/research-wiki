---
id: arxiv:2105.14339
type: paper
title: Aligning AI with Broad Human Values
url: https://arxiv.org/abs/2105.14339
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-hacking
---

**Core Problem**
The paper introduces and investigates *well-f-coveredness*, a novel graph-theoretic property where every maximal induced forest shares the same order, defined as the forest number $f(G)$. The research aims to characterize graphs satisfying this property, clarify its relationship with the classical concept of well-coveredness, and develop systematic construction rules for generating larger well-f-covered graphs from smaller components.

**Method and Construction Recipe**
The study establishes well-f-coveredness through a sequence of structural operations that preserve the property. The construction protocol proceeds as follows:
1. **Base Verification:** Confirm that all maximal forests in a base graph have identical order. Trivial graphs, forests, cycles $C_n$, and complete graphs $K_n$ ($n \geq 2$) serve as foundational examples.
2. **Bridge and Pendant Edge Manipulation:** Remove or add bridges, isolated vertices, or pendant edges. These operations do not invalidate well-f-coveredness and adjust the forest number by exactly $\pm 1$.
3. **Disjoint Union:** Combine two disjoint well-f-covered graphs $G$ and $H$. The union preserves the property, and the forest number sums linearly.
4. **Path Subdivision:** Replace an edge incident to a degree-2 vertex with a path of length $d \geq 2$. This maintains well-f-coveredness while incrementing the forest number by $d-1$.
5. **Path Connection:** Connect two disjoint well-f-covered graphs via a path of length $d$ with new interior vertices. The resulting graph remains well-f-covered.
6. **Vertex Identification:** Identify a vertex $x$ in $G$ with a vertex $y$ in $H$. This preserves the property only if every maximal forest of $G$ necessarily contains $x$.
7. **Graph Join ($G \vee H$):** Join two disjoint graphs by adding all possible edges between them. This requires strict algebraic alignment between forest numbers and independence numbers, or specific structural constraints (e.g., one component being empty or trivial).

**Key Formulas**
The quantitative relationships governing these constructions are formalized as:

$$
f(G \cup H) = f(G) + f(H)
$$

$$
f(A) = f(G) + f(H) + d - 1
$$

$$
f(G^*) = f(G) + d - 1
$$

$$
f(L) = f(G) - |V(H)| + n
$$

For join operations, the necessary condition is:

$$
f(G) = f(H) = \alpha(G) + 1 = \alpha(H) + 1
$$

$$
f(G \vee H) = n + 1
$$

where $\alpha(G)$ denotes the independence number, $d$ is path length, $n$ is the number of identified vertices along a path, and $|V(H)|$ is the vertex count of a forest component.

**Key Quantitative Results**
The paper provides exact characterizations for boundary forest numbers. A connected graph satisfies $f(G) = 2$ if and only if it is a complete graph $K_n$ ($n \geq 2$) or an empty graph of order 2. A graph satisfies $f(G) = |V(G)| - 1$ if and only if it contains exactly one cycle. Specific families are quantified: cycles $C_n$ have $f(C_n) = n - 1$, while complete graphs have $f(K_n) = 2$ for $n \geq 2$ and $f(K_1) = 1$. The study determines that a wheel graph $W_n$ is well-f-covered if and only if $n \in \{4, 5\}$. Complete bipartite graphs $K_{r,s}$ satisfy the property if and only if $r = s$ or at least one of $r, s$ equals 1.

**Stated Limitations**
The research explicitly notes that well-f-coveredness and well-coveredness are mathematically independent; a graph may satisfy one without the other. Bridge operations preserve well-f-coveredness but can disrupt well-coveredness, highlighting divergent structural sensitivities. Vertex identification is not a reliable construction method unless strict maximality constraints are enforced, as demonstrated by the failure of joining two $K_3$ graphs at a single vertex. The join operation imposes rigid algebraic constraints on independence numbers and forest numbers, limiting its applicability to specific graph classes. Furthermore, the entire analysis is strictly confined to finite, simple graphs, excluding multigraphs, infinite structures, or weighted variants.
