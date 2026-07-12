---
id: arxiv:1906.08850
type: paper
title: Implicitly Adaptive Importance Sampling
url: https://arxiv.org/abs/1906.08850
retrieved: '2026-07-12'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# Importance Weighted Moment Matching (IWMM)

### Core Problem
Importance sampling (IS) is used to estimate expectations $\mu = \mathbb{E}_p[h(\theta)] = \int h(\theta)p(\theta)d\theta$. The accuracy of the estimator depends heavily on the choice of the proposal distribution $g(\theta)$. If $g(\theta)$ is a poor match for the target $p(\theta)$—particularly in the tails—the estimator can exhibit poor pre-asymptotic behavior, leading to significant bias or infinite variance. Existing adaptive importance sampling (AIS) methods often rely on parametric proposals (e.g., Gaussian or Student-$t$), which struggle with high-dimensional, multimodal, or highly correlated distributions.

### Method
The authors propose **Importance Weighted Moment Matching (IWMM)**, an implicit adaptive method that transforms an existing Monte Carlo sample using affine transformations rather than resampling from a parametric distribution.

#### Target of Adaptation
Instead of matching $p(\theta)$, IWMM targets the optimal proposal distribution for a specific expectation:
*   **Standard IS:** $g_{\mathrm{IS}}^{\mathrm{opt}}(\boldsymbol{\theta}) \propto p(\boldsymbol{\theta})|h(\boldsymbol{\theta})|$
*   **Self-Normalized IS (SNIS):** $g_{\mathrm{SNIS}}^{\mathrm{opt}}(\boldsymbol{\theta}) \propto p(\boldsymbol{\theta})|h(\boldsymbol{\theta}) - \mathbb{E}_p[h(\boldsymbol{\theta})]|$. To simplify sampling, the authors use a "split proposal" approximation:

$$
g_{\mathrm{SNS}}^{\mathrm{split}}(\boldsymbol{\theta}) \propto |h(\boldsymbol{\theta})|p(\boldsymbol{\theta}) + \mathbb{E}_p[h(\boldsymbol{\theta})]p(\boldsymbol{\theta})
$$

#### Affine Transformations
The method applies a transformation $T: \boldsymbol{\theta}^{(s)} \mapsto \mathbf{A}\boldsymbol{\theta}^{(s)} + \mathbf{b} =: \dot{\boldsymbol{\theta}}^{(s)}$. Three levels of moment matching are used:
1.  **$T_1$ (Mean):** $\dot{\boldsymbol{\theta}}^{(s)} = \boldsymbol{\theta}^{(s)} - \overline{\boldsymbol{\theta}} + \overline{\boldsymbol{\theta}}_{w}$
2.  **$T_2$ (Marginal Variance):** $\dot{\boldsymbol{\theta}}^{(s)} = \mathbf{v}_{w}^{1/2} \circ \mathbf{v}^{-1/2} \circ (\boldsymbol{\theta}^{(s)} - \overline{\boldsymbol{\theta}}) + \overline{\boldsymbol{\theta}}_{w}$
3.  **$T_3$ (Covariance):** $\dot{\boldsymbol{\theta}}^{(s)} = \mathbf{L}_{w}\mathbf{L}^{-1}(\boldsymbol{\theta}^{(s)} - \overline{\boldsymbol{\theta}}) + \overline{\boldsymbol{\theta}}_{w}$

Where $\overline{\boldsymbol{\theta}}_w$ is the weighted mean, $\mathbf{v}_w$ is the weighted marginal variance, and $\mathbf{L}_w\mathbf{L}_w^{\mathsf{T}} = \mathbf{\Sigma}_w$ is the weighted covariance matrix.

#### Step-by-Step Recipe
1.  **Initialization:** Start with a sample $\{\boldsymbol{\theta}^{(s)}\}_{s=1}^S$ from a proposal $g$.
2.  **Diagnostic:** Compute importance weights and the Pareto $\hat{k}$ diagnostic.
3.  **Iterative Adaptation:**
    *   Apply $T_1$ repeatedly. If the new $\hat{k}$ is lower than the previous, accept the transformation.
    *   Once $T_1$ no longer reduces $\hat{k}$, attempt $T_2$, and finally $T_3$.
4.  **Stopping Criterion:** Stop when $\hat{k} \le 0.7$ (the threshold for practical convergence) or when $T_3$ fails to further reduce $\hat{k}$.
5.  **SNIS Double Adaptation:** For SNIS, the numerator and denominator are adapted separately using their respective weights and then combined via multiple importance sampling.

### Key Quantitative Results
The method was evaluated using Bayesian leave-one-out cross-validation (LOO-CV):
*   **Outlier Robustness:** In a Gaussian model with a single outlier ($y_{30}=20$), IWMM (PSIS+MM) almost perfectly aligned with analytical results, while standard AIS-Gaussian was biased.
*   **Sample Efficiency:** In a Poisson regression (Roach data), IWMM provided reliable estimates ($\hat{k} < 0.7$) with only $S=1,000$ draws, whereas other methods remained biased even at $S=64,000$.
*   **High Correlation:** In a linear regression with 30 correlated predictors ($\rho=0.8$), PSIS+MM reduced all LOO folds to $\hat{k} < 0.7$ with $S=2,000$, while parametric AMIS failed even at $S=8,000$.
*   **High Dimensionality:** For an Ovarian cancer dataset (3,075 parameters), PSIS+MM reduced the number of problematic folds ($\hat{k} > 0.7$) from 34.8 to 16.2 (at $S=4,000$).
*   **Computational Cost:** In the Ovarian dataset, PSIS+MM required 1,558 seconds, compared to 27,477 seconds for AMIS-$t \times 2$.

### Limitations
*   **Initial Proposal Dependency:** Performance depends heavily on the quality of the starting proposal distribution.
*   **Transformation Simplicity:** Simple affine transformations may be insufficient for targets with complex characteristics, such as differing tail thicknesses, intricate correlation structures, or multiple modes.
