---
id: arxiv:1909.08568
type: paper
title: Fine-Tuning Pre-Trained Language Models with Human Preferences
url: https://arxiv.org/abs/1909.08568
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-hacking
---

**Core Problem**
The paper addresses the historical difficulty of constructing a fundamental polygon for the Riemann surface of the Klein quartic (genus 3, automorphism group $\mathrm{PSL}(2,7)$) and extending this construction to higher levels, specifically $\mathbb{H}^*/\Gamma(11)$. Felix Klein’s original 1878/79 derivation of the 14-sided fundamental region and its side identifications relied on intricate hyperbolic geometry and explicit Möbius transformations. The authors seek to replace this cumbersome approach with an arithmetic method based on Farey fractions and the Farey tessellation modulo $n$, thereby making side pairings immediate and systematic.

**Method/Recipe Step by Step**
1. **Define the Farey Tessellation:** Begin with the Farey map $\mathcal{M}_3$ on the extended upper half-plane $\mathbb{H}^* = \mathbb{H} \cup \mathbb{Q} \cup \{\infty\}$. Vertices are rationals $\frac{a}{c}$, and an edge exists between $\frac{a}{c}$ and $\frac{b}{d}$ if and only if $ad-bc = \pm 1$. Edges are drawn as hyperbolic semicircles or vertical lines.
2. **Quotient by Congruence Subgroup:** For a chosen level $n$, form the regular map $\mathcal{M}_3(n) = \mathcal{M}_3/\Gamma(n)$, where $\Gamma(n)$ is the principal congruence subgroup modulo $n$ of $\mathrm{PSL}(2,\mathbb{Z})$. Vertices are represented by Farey coordinates $[\frac{a}{c}]$ with $(a,c,n)=1$.
3. **Organize Vertices by Graph-Theoretic Distance:** Fix the north pole $N = \frac{1}{0}$. Compute the distance $\delta$ from $N$ to all other vertices using the determinant $\Delta = ad-bc$. Vertices fall into disjoint circuits: distance 1, distance 2, and distance 3 (poles).
4. **Construct Circuits for $n=7$:** Identify the first circuit $S_1(7)$ of vertices with denominator 1. Construct the second circuit $S_2(7)$ using Farey fractions with denominators $\pm 2, \pm 3$, extended by translations $t \mapsto t+k$. Draw hyperbolic circles $C_1, C_2, C_3$ centered at $N$ to contain these circuits.
5. **Extract the Fundamental Polygon:** Connect the poles $\frac{2}{0}$ and $\frac{3}{0}$ along $C_3$ to form a 14-sided polygon. Identify sides by matching identical Farey coordinate sequences on opposite boundary segments, yielding orientable pairings identical to Klein’s.
6. **Generalize to $n=11$:** Define $S(11)$ using fractions with denominators up to $(11-1)/2$. Generate $S_2(11)$ by concatenating translates $S(11)+k$. Construct a fundamental sector $W$ of 20 triangles, rotate it 11 times via $z \mapsto z+k$ to form $W^*$, and pair the 198 boundary edges orientably to close the polygon.

**Key Formulas**
The order of the automorphism group (and dart count) for $\mathcal{M}_3(n)$ is:

$$
\mu(n) = \frac{n^3}{2} \prod_{p|n} \left(1 - \frac{1}{p^2}\right).
$$

The genus of the associated surface follows from the Euler-Poincaré formula:

$$
g(n) = 1 + \frac{n^2}{24} (n - 6) \prod_{p|n} \left(1 - \frac{1}{p^2}\right). \tag{1}
$$

For prime levels $p$, the graph-theoretic distance between distinct vertices $\frac{a}{c}$ and $\frac{b}{d}$ is:

$$
\delta \left( \frac{a}{c}, \frac{b}{d} \right) = \begin{cases} 1 & \text{if } |\Delta| = 1, \\ 2 & \text{if } |\Delta| \neq 0, \pm 1, \\ 3 & \text{if } \Delta = 0, \end{cases}
$$

where $\Delta = ad-bc$. The circuit lengths are $|S_1(p)| = p$ and $|S_2(p)| = p(p-4)$.

**Key Quantitative Results**
For $n=5$, $\mu(5)=60$, yielding 12 vertices and an icosahedral map. For $n=7$, $\mu(7)=168$, producing 24 vertices of valency 7 and the 14-sided fundamental polygon. For $n=11$, $\mu(11)=660$, resulting in 60 vertices, 330 edges, 220 triangular faces, and a genus-26 surface. The constructed polygon for $n=11$ has 198 boundary edges, which are systematically paired to confirm orientability.

**Stated Limitations**
The authors acknowledge that while the Farey coordinate method simplifies Klein’s original side-pairing derivation, the combinatorial and geometric complexity grows rapidly with $n$. Klein himself could not explicitly draw a fundamental region for $\mathbb{H}/\Gamma(11)$ in his 1879 follow-up; the authors successfully construct the Farey map $\mathcal{M}_3(11)$ but note the construction is "somewhat more complicated." The framework is explicitly demonstrated for small primes ($n=5,7,11$), and the paper does not provide a closed-form algorithm for arbitrary composite levels or higher genera beyond these cases.
