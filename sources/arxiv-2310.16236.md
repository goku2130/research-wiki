---
id: arxiv:2310.16236
type: paper
title: Query-Efficient Algorithm to Find all Nash Equilibria in a Two-Player Zero-Sum
  Matrix Game
url: https://arxiv.org/abs/2310.16236
retrieved: '2026-07-12'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Query-Efficient Algorithm to Find all Nash Equilibria in a Two-Player Zero-Sum Matrix Game

### Core Problem
The research addresses the query complexity of finding the set of all Nash equilibria $\mathcal{X}_{\star} \times \mathcal{Y}_{\star}$ for a two-player zero-sum matrix game defined by an $n \times n$ input matrix $A \in \mathbb{R}^{n \times n}$. While prior work established that finding a single Nash equilibrium can require $\Omega(n^2)$ queries and finding a strict Pure Strategy Nash Equilibrium (PSNE) can take $O(n)$, this work seeks to characterize the complexity in terms of the solution size. Specifically, it uses the row support size $k_1 := |\bigcup_{x \in \mathcal{X}_{\star}} \text{supp}(x)|$ and column support size $k_2 := |\bigcup_{y \in \mathcal{Y}_{\star}} \text{supp}(y)|$, where $k := \max\{k_1, k_2\}$.

### Method/Recipe
The authors propose a randomized algorithm that reduces the problem of finding all Nash equilibria to finding a strict PSNE in a constructed high-dimensional matrix.

1.  **Candidate Value Identification (`SetV`):** The algorithm first identifies a set $\mathcal{V}$ of candidate values for the game value $V_A^\star$. This is achieved by applying a `FindPsne` subroutine to a large matrix $A^{(k_1, k_2)}$ where entries are the values of the game for all $k_1 \times k_2$ submatrices of $A$.
2.  **Large Matrix Construction:** A modified matrix $A^{(k_1, k_2, \mathcal{V})}$ is implicitly constructed. To ensure the optimal submatrix corresponds to a **strict PSNE**, entries are adjusted using a small $\epsilon$ (the minimum difference between distinct values in $\mathcal{V}$):
    *   If the submatrix $M$ has a Nash equilibrium with full row and column support ($k_{1,M}=k_1$ and $k_{2,M}=k_2$), the entry is $V_M^\star$.
    *   If only row support is full, the value is $V_M^\star + \epsilon/4$.
    *   If only column support is full, the value is $V_M^\star - \epsilon/4$.
    *   Otherwise, the entry is $V_M^\star$.
3.  **Strict PSNE Localization (`FindPsne`):** A randomized subroutine locates the strict PSNE in the large matrix through a logarithmic number of stages. In each stage, it:
    *   Samples a logarithmic number of rows and columns.
    *   Queries the corresponding entries to identify a candidate row $\hat{i}$ and column $\hat{j}$.
    *   Prunes half of the sub-optimal rows and columns based on the median values of the queried column/row.
4.  **Query Oracle:** Because the large matrix is too vast to store, the algorithm uses a query oracle. A single row or column query in the large matrix corresponds to querying $O(nk)$ entries in the original matrix $A$.
5.  **Verification and Iteration:** The algorithm verifies the result using three conditions (C1: $V_{M_1}^\star = V_{M_2}^\star$; C2 and C3: intersection of supports equals the identified indices). Since $k_1$ and $k_2$ are unknown, the algorithm iterates through all pairs $(s_1, s_2)$ starting from $s=1$ until the conditions are met.

### Key Formulas
The Nash equilibrium $(x_\star, y_\star)$ is defined by:

$$
\langle x, Ay_\star \rangle \leq \langle x_\star, Ay_\star \rangle \leq \langle x_\star, Ay \rangle \quad \forall (x, y) \in \triangle_n \times \triangle_n
$$

The value of the game $V_A^\star$ is given by the minimax theorem:

$$
V_A^\star := \max_{x \in \triangle_n} \min_{y \in \triangle_n} \langle x, Ay \rangle = \min_{y \in \triangle_n} \max_{x \in \triangle_n} \langle x, Ay \rangle
$$

### Quantitative Results
*   **Upper Bound:** With probability $1 - \delta$, the algorithm returns the set of all Nash equilibria $\mathcal{X}_{\star} \times \mathcal{Y}_{\star}$ by querying:

$$
O(n k^5 \cdot \log n \cdot \log(n/\delta)) \text{ entries.}
$$

*   **Lower Bound:** For any randomized algorithm, there exists an $n \times n$ matrix with $\min\{k_1, k_2\} = 1$ such that the expected number of queries $\mathbb{E}[\tau(A)]$ is:

$$
\mathbb{E}[\tau(A)] \geq \frac{(n-1)(k+1)}{2} = \Omega(nk)
$$

### Stated Limitations
The upper bound is tight up to a factor of $\text{poly}(k)$. The algorithm is randomized, meaning its success is probabilistic ($1-\delta$), and the query complexity increases significantly as the support size $k$ grows.
