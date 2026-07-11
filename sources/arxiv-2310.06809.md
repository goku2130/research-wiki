---
id: arxiv:2310.06809
type: paper
title: Teaching Large Language Models to Judge with Self-Play
url: https://arxiv.org/abs/2310.06809
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

**Core Problem**
The paper investigates two classical, unresolved conjectures in analytic number theory: the infinitude of regular primes and the infinitude of non-Wieferich primes. A prime $p$ is regular if it is odd and does not divide the class number of the $p$th cyclotomic field; it is non-Wieferich if $p^2 \nmid 2^{p-1}-1$. Historically linked to Fermat’s Last Theorem via Kummer’s and Wieferich’s theorems, these primes are now studied through the lens of finite multiple zeta values (FMZVs). The central challenge is proving the non-vanishing of FMZVs and their level-$N$ generalizations for non-empty indices. Currently, no single FMZV corresponding to a non-empty index has been rigorously proven to be non-zero, creating a foundational gap in the algebraic theory of these objects.

**Method/Recipe Step by Step**
1. **Ring Construction:** Define the ring $\mathcal{A} = \left( \prod_{p \in \mathcal{P}} \mathbb{Z}/p\mathbb{Z} \right) / \left( \bigoplus_{p \in \mathcal{P}} \mathbb{Z}/p\mathbb{Z} \right)$, which serves as the ambient space for FMZVs.
2. **Local Sum Definition:** For a prime $p$, an index $\boldsymbol{k}=(k_1,\dots,k_r)$, and a residue tuple $\boldsymbol{\alpha} \in (\mathbb{Z}/N\mathbb{Z})^r$, compute the truncated harmonic sum $\zeta_{p,N}^{\boldsymbol{\alpha}}(\boldsymbol{k})$ over integers $m_i$ strictly between $0$ and $p$ satisfying $m_i \equiv \alpha_i \pmod N$.
3. **Color Map Assembly:** Introduce a color map $c: (\mathbb{Z}/N\mathbb{Z})^\times \to (\mathbb{Z}/N\mathbb{Z})^r \cup \{\boxtimes\}$ that assigns a residue tuple to each prime class. Assemble the global value $\zeta_{\mathcal{A},N}^c(\boldsymbol{k})$ by taking the equivalence class $(\zeta_{p,N}^{c(\overline{p})}(\boldsymbol{k}) \bmod p)_{p \in \mathcal{P}}$ in $\mathcal{A}$.
4. **Algebraic Closure:** Generate the $\mathbb{Q}$-vector space $\mathcal{Z}_{\mathcal{A}}(N)$ spanned by all such values. Verify that $\mathcal{Z}_{\mathcal{A}}(N)$ forms a $\mathbb{Q}$-subalgebra of $\mathcal{A}$ by applying the generalized harmonic (stuffle) product and reversal formulas.
5. **Number-Theoretic Bridging:** Relate the algebraic structure to classical arithmetic by expressing special values via Bernoulli–Seki numbers $B_n$ and Fermat quotients $q_p(N) = \frac{N^{p-1}-1}{p}$. Use established congruences (Skula–Dobson–Ichimura and Lerch) to rewrite logarithmic elements in $\mathcal{A}$ as explicit linear combinations of level-$N$ zeta values.
6. **Conditional Non-Vanishing Proofs:** Leverage the infinitude of primes avoiding specific divisibility conditions (regularity or non-Wieferich status) to demonstrate that products or powers of constructed zeta values must be non-zero in $\mathcal{A}$, thereby guaranteeing at least one non-vanishing element per weight.

**Key Formulas**

$$
\zeta_{p,N}^{\boldsymbol{\alpha}}(\boldsymbol{k}) := \sum_{\substack{0 < m_1 < \dots < m_r < p \\ m_i \in \alpha_i \text{ for all } 1 \le i \le r}} \frac{1}{m_1^{k_1} \cdots m_r^{k_r}}
$$

$$
\zeta_{\mathcal{A},N}^c(\boldsymbol{k}) := \left( \zeta_{p,N}^{c(\overline{p})}(\boldsymbol{k}) \bmod p \right)_{p \in \mathcal{P}}
$$

$$
\mathfrak{Z}(k) := \left( \frac{B_{p-k}}{k} \bmod p \right)_{p \in \mathcal{P}}
$$

$$
\zeta_{\mathcal{A}}(k_1, k_2) = (-1)^{k_2} \binom{k_1 + k_2}{k_1} \mathfrak{Z}(k_1 + k_2)
$$

$$
\log_{\mathcal{A}}(2) = - \frac{2N}{N + 1} \sum_{0 \leq j < N/2} \zeta_{\mathcal{A}, 2N}^{[2j]}(1)
$$

**Key Quantitative Results**
The algebraic analysis establishes explicit structural and analytic bounds. The FMZV vector spaces vanish for low weights: $\mathcal{Z}_{\mathcal{A}, 1} = \mathcal{Z}_{\mathcal{A}, 2} = \mathcal{Z}_{\mathcal{A}, 4} = 0$. For the least odd integer $k \geq 3$ such that $p \nmid B_{p-k}$ (denoted $\partial_p$), the paper proves $\partial_p \leq \frac{p-3}{2}$ when $p \equiv 1 \pmod 4$ and $\partial_p \leq \frac{p-5}{2}$ when $p \equiv 3 \pmod 4$, valid for all $p \geq 11$. Regarding Fermat quotients, the least base $N$ not dividing $q_p(N)$ (denoted $\ell_p$) satisfies $\ell_p \leq 4(\log p)^2$ (Lenstra) and $\ell_p \leq (\log p)^{463/252+o(1)}$ (Bourgain et al.). The lower bound for irregular prime density is given by $\#\{p \leq x : p \text{ irregular}\} \geq (1 + o(1)) \frac{\log \log x}{\log \log \log x}$.

**Stated Limitations**
The primary limitation is the current inability to prove that any FMZV for a non-empty index is non-zero; all non-vanishing results remain strictly conditional on Conjecture 1.1 or Conjecture 1.2. The ambient ring $\mathcal{A}$ is explicitly noted as not being an integral domain, which restricts standard algebraic factorization techniques. Furthermore, the infinitude of Wieferich primes is conjectured to have asymptotic density zero, rendering unconditional proofs highly intractable. The author acknowledges that directly resolving the prime distribution conjectures is currently impossible, instead proposing weaker, more immediate problems (e.g., proving the existence of at least one weight or level yielding infinitely many non-dividing primes) as pragmatic research alternatives.
