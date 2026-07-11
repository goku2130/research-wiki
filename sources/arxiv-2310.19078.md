---
id: arxiv:2310.19078
type: paper
title: Rejection Sampling GPT (RSGPT)
url: https://arxiv.org/abs/2310.19078
retrieved: '2026-07-11'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

Nonlinear dynamical systems present formidable computational challenges, prompting the need for global linearization techniques that extend beyond the narrow validity of first-order Taylor expansions. While Carleman Linearization (CL) and Koopman operator theory both lift nonlinear dynamics to infinite-dimensional linear systems via finite section approximations, CL suffers from rapidly expanding matrix dimensions and high computational costs, particularly for non-polynomial systems. This study introduces Koopman Spectral Linearization (KSL) as an analytical, spectral-method-based alternative that explicitly approximates the Koopman operator generator using Chebyshev differentiation matrices, offering a computationally efficient pathway to solve nonlinear autonomous ODEs.

The KSL procedure follows a deterministic spectral collocation recipe. First, the autonomous nonlinear system is defined as $d\boldsymbol{x}/dt = \boldsymbol{f}(\boldsymbol{x})$. Second, a scalar observable $g(\boldsymbol{x})$ is introduced, with its temporal evolution governed by the Koopman generator $\mathcal{K} = \nabla g(\boldsymbol{x}) \cdot \boldsymbol{f}(\boldsymbol{x})$. Third, the state space is discretized using Gauss-Lobatto collocation points $\Theta$ across each dimension. Fourth, eigenfunctions are approximated via Lagrange polynomial interpolation over $\Theta$. Fifth, the finite-dimensional approximation matrix $K$ of $\mathcal{K}$ is constructed by combining the vector field evaluations at collocation points with Chebyshev differentiation matrices $D_i$. Sixth, the lifted linear system $d\boldsymbol{y}/dt = K\boldsymbol{y}$ is solved with the initial condition $\boldsymbol{y}(0) = \text{vec}(g(\Theta))$. Finally, the observable at the initial state $x_0$ is recovered by extracting the central element of the solution vector $\boldsymbol{y}(t)$, which corresponds to the spectral expansion coefficients.

The core mathematical machinery relies on the following key formulas:
$$\mathcal{K}g = \nabla g(\boldsymbol{x}) \cdot \boldsymbol{f}(\boldsymbol{x})$$
$$K = \sum_{i=1}^d \text{diag}(\text{vec}(f_i(\Theta)))(\otimes_i D_i)$$
$$\frac{d\boldsymbol{y}}{dt} = K\boldsymbol{y}, \quad \boldsymbol{y}(0) = \text{vec}(g(\Theta))$$
$$g_N(x(t)) = \sum_{j=1}^{N^d} c_j e^{\lambda_j t} (v_j)_{\frac{N^d+1}{2}}$$

Numerical experiments across five benchmark systems demonstrate that KSL achieves exponential convergence with respect to truncation order $N$ and consistently outperforms CL in accuracy and efficiency. At $N=9$, KSL yields an error of $1.99 \times 10^{-7}$ for the 3D Kraichnan-Orszag model in $0.1370$ seconds, whereas CL produces $8.80 \times 10^{-3}$ error in $32.0546$ seconds. For the Lotka-Volterra model, KSL achieves $6.39 \times 10^{-6}$ error in $0.0188$ seconds compared to CL's $1.49 \times 10^{-1}$ error in $0.1525$ seconds. Matrix dimension analysis confirms KSL scales as $N^d$, drastically reducing size relative to CL's $O(d^N)$ growth; for $N=9, d=3$, KSL requires a $729 \times 729$ matrix versus CL's $29523 \times 29523$. KSL also demonstrates superior accuracy for non-polynomial dynamics, where CL's reliance on Taylor polynomialization introduces significant truncation errors at low orders.

The authors identify critical limitations. KSL requires careful selection of the collocation radius $r$; deviations from an optimal range produce a characteristic "bowl/V-shaped" error increase. Without prior knowledge of the state's spatial bounds, estimating $r$ accurately is difficult, potentially limiting precision. Additionally, computing multi-dimensional observables necessitates solving $d$ separate linear systems with distinct initial conditions, unlike CL's single-system formulation. Finally, the tensor-product collocation grid causes matrix dimensions to scale exponentially with state dimension $d$, rendering the method computationally prohibitive for high-dimensional problems unless sparse grid techniques are integrated.
