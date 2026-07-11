---
id: arxiv:2304.01132
type: paper
title: 'Search-in-the-Chain: Interactively Enhancing Large Language Models with Search'
url: https://arxiv.org/abs/2304.01132
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Almost Sure Limit Theorems for Non-Regular Continued Fraction Algorithms

### Core Problem
In infinite ergodic theory, for a conservative ergodic measure-preserving transformation $T$ of a $\sigma$-finite measure space $(X, \mathcal{B}, \mu)$ where $\mu(X) = \infty$, the Birkhoff sums $S_N g(x) := \sum_{j=1}^N (g \circ T^{j-1})(x)$ typically lack a strong law of large numbers (SLLN). According to Aaronson's theorem, for any norming sequence $(d_N)$, the ratio $S_N g(x)/d_N$ for a non-negative observable $g$ typically satisfies $\limsup_{N \to \infty} S_N g(x)/d_N = \infty$ or $\liminf_{N \to \infty} S_N g(x)/d_N = 0$ for $\mu$-almost every $x \in X$. This paper addresses the specific case where $g \notin L^1(X, \mu)$ (non-integrable) and seeks methods to obtain almost sure asymptotic behavior.

### Method and Recipe
The authors employ a combination of induced maps, additive corrections based on excursion lengths, and trimming of extreme values.

1.  **Induced Map and Wandering Rate**: Fix a set $E \in \mathcal{B}$ with $\mu(E)=1$. Define the first return time $\varphi_E(x) := \inf\{k \geq 1 : T^k(x) \in E\}$ and the wandering rate $w_n(E) := \sum_{k=0}^{n-1} \mu(A_{>k})$, where $A_{>k} = \{x \in E : \varphi_E(x) > k\}$.
2.  **Longest Excursion**: Define $m(N, E, x)$ as the longest excursion out of $E$ beginning within the first $N$ steps:

$$
m(N, E, x) := 1 + \max \{k \geq 1 : \exists \ell \in \{1, \dots, N+1\} \text{ s.t. } T^{\ell+j}(x) \notin E, \forall j = 0, \dots, k-1\}
$$

3.  **Trimming**: To stabilize the sum, the authors use "lightly trimmed sums" $S_N^r$, which remove the $r$ largest entries of the sequence.
4.  **Mixing Requirements**: The set $E$ must induce "rapid $\psi$-mixing," meaning the $\psi$-mixing coefficient $\psi(n)$ of the sequence $(f \circ T_E^{n-1})$ satisfies $\sum_{n \geq 1} \psi(n)/n < \infty$.
5.  **Asymptotic Analysis**: The Birkhoff sums are analyzed by splitting the observable $g$ into a constant part $g_c$ and a non-integrable part $f$. The behavior of $S_N f$ is then studied via the induced map $T_E$ and the norming sequence $b(n)$, the asymptotic inverse of $a(y) := y / \int_{y_0}^y (1 - F(t)) dt$.

### Key Formulas
The primary result for a non-negative observable $g$ satisfying $\mu(g > n) \sim \kappa \mu(A_{>n})$ and $g \equiv c$ on $X \setminus E$ is:

$$
\lim_{N \to \infty} \frac{S_{N+m(N, E, x)} g(x) - \max_{1 \leq k \leq N+m(N, E, x)} (g \circ T^{k-1})(x) - cm(N, E, x)}{N} = c + \kappa
$$

For cases where $\mu(g > n)/\mu(A_{>n}) \to \infty$, the authors use the $W$-trimmed sum $S^W$:

$$
\lim_{N \to \infty} \frac{S_{N+m(N, E, x)}^W g(x) - cm(N, E, x)}{b(\alpha(N))} = 1
$$

where $\alpha(n)$ is the wandering rate indicator and $b(n)$ is the asymptotic inverse of $a(y) := y / \int_{y_0}^y \mu(g > t) dt$.

### Quantitative Results
The authors apply these theorems to two non-regular continued fraction algorithms:

*   **Backward (Rényi type) Continued Fractions**: For the coefficients $\{d_j(x)\}$, the limit is:

$$
\lim_{N \to \infty} \frac{1}{N} \left( \sum_{j=1}^{N+m(N, E, x)} d_j(x) - \max \{ 2m(N, E, x), \max_{1 \leq k \leq N+m(N, E, x)} d_k(x) \} \right) = 3
$$

*   **Even-Integer Continued Fractions**: For the coefficients $\{h_j(x)\}$, the limit for the sum of $2h_j(x)$ (and similarly for $2h_j(x) + \frac{1}{2}(\varepsilon_j - 1)$) is also $3$.
*   **Rapidly Increasing Observables**: For $\ell(t) = t^s$ with $s > 1$, the norming sequence is:

$$
\gamma_N \sim \frac{(\log 2)^s}{s-1} \left( \frac{N}{\log N} \right)^s (\log \log N)^{(1-s)u}
$$

### Stated Limitations
*   **Mixing Constraints**: The results require the induced map $T_E$ to exhibit rapid $\psi$-mixing, which may not hold for all systems.
*   **Observable Constraints**: The observable $g$ must be locally constant on the partition $\mathcal{P}_E$ and satisfy specific tail distribution conditions relative to the wandering rate.
*   **Point Dependency**: The correction term $m(N, E, x)$ is highly dependent on the point $x$, meaning the "time" $N+m(N, E, x)$ varies significantly across the space.
