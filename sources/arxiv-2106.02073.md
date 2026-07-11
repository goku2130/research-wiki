---
id: arxiv:2106.02073
type: paper
title: 'Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path'
url: https://arxiv.org/abs/2106.02073
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path

### Core Problem
Neural Collapse (NC) is a phenomenon observed in deep networks trained with cross-entropy (CE) loss, where last-layer features collapse to their class-means, and both classifiers and class-means converge to a Simplex Equiangular Tight Frame (ETF). This paper investigates whether NC emerges under Mean Squared Error (MSE) loss and seeks to provide a theoretical framework to describe the dynamics of this emergence, leveraging the analytical tractability of MSE compared to CE.

### Method and Theoretical Recipe
The authors employ a combination of empirical verification and a theoretical construct called the "central path" to analyze the training dynamics.

**1. Empirical Verification**
The authors tested MSE-trained networks across three architectures (VGG, ResNet, DenseNet) and five benchmark datasets (MNIST, FashionMNIST, CIFAR10, SVHN, STL10) to confirm the occurrence of the four NC properties:
*   **(NC1) Within-class variability collapse:** $\mathbf{\Sigma}_{B}^{\dagger}\mathbf{\Sigma}_{W}\rightarrow\mathbf{0}$.
*   **(NC2) Convergence to Simplex ETF:** Class-means $\mu_c$ become maximally separated.
*   **(NC3) Convergence to self-duality:** Classifiers $w_c$ align with class-means.
*   **(NC4) Simplification to nearest class center:** Decision rule collapses to the nearest-class-mean.

**2. MSE Loss Decomposition**
The authors decompose the MSE loss $\mathcal{L}$ into three components:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{NC1}} + \mathcal{L}_{\mathrm{NC2/3}} + \mathcal{L}_{\mathrm{LS}}^{\perp}
$$

*   $\mathcal{L}_{\mathrm{NC1}}$ captures activation collapse (within-class variance).
*   $\mathcal{L}_{\mathrm{NC2/3}}$ captures the convergence of class-means and classifiers to a Simplex ETF.
*   $\mathcal{L}_{\mathrm{LS}}^{\perp}$ captures the deviation of the actual classifier $W$ from the least-squares (LS) optimal classifier $W_{LS}$.

**3. The Central Path and Renormalization**
The authors define the **central path** $\mathcal{P}$ as the set of pairs $(W_{LS}(\widetilde{H}), \widetilde{H})$ where the classifier is always MSE-optimal for the current features. To analyze dynamics on this path, they introduce **continually renormalized gradient flow**, where features are renormalized to identity within-class covariance:

$$
\mathbf{X} = \mathbf{\Sigma}_{W}^{-\frac{1}{2}} \mathbf{H}
$$

**4. SNR Matrix Analysis**
The dynamics are studied via the **Signal-to-Noise Ratio (SNR) matrix**:

$$
\mathrm{SNR} \equiv \mathbf{\Sigma}_{W}^{-\frac{1}{2}}\overline{\mathbf{M}}
$$

where $\overline{\mathbf{M}}$ is the matrix of globally-centered class-means. By analyzing the Singular Value Decomposition (SVD) of the SNR matrix, the authors derive the evolution of its singular values $\omega_j$.

### Key Formulas
The optimal least-squares classifier $\widetilde{W}_{LS}$ is given by:

$$
\widetilde{\mathbf{W}}_{L S}=\frac{{}}C\widetilde{\mathbf{M}}^{\top}(\widetilde{\mathbf{\Sigma}}_{T}+\widetilde{\mathbf{\mu}}_{G}\widetilde{\mathbf{\mu}}_{G}^{\top}+\lambda\mathbf{I})^{-1}
$$

The components of the loss decomposition are:

$$
\mathcal{L}_{\mathrm{N C 1}}(\widetilde{\mathbf{H}})=\frac{1}{2}\text{t r}\Bigl\{\widetilde{\mathbf{W}}_{\mathit{L S}}\left[\widetilde{\mathbf{\Sigma}}_{\mathit{W}}+\lambda\mathbf{I}\right]\widetilde{\mathbf{W}}_{\mathit{L S}}^{\top}\Bigr\}
$$

$$
\mathcal{L}_{\mathrm{N C 2/3}}(\widetilde{\boldsymbol{H}})=\frac{1}{2C}\|\widetilde{\boldsymbol{W}}_{L S}\widetilde{\boldsymbol{M}}-\boldsymbol{I}\|_{F}^{2}
$$

The closed-form dynamics for the SNR singular values $\omega_j(t)$ are:

$$
c_{1}\text{l o g}(\omega_{j}(t))+c_{2}\omega_{j}^{2}(t)+c_{3}\omega_{j}^{4}(t)=a_{j}+t
$$

where $c_1, c_2, c_3$ are positive constants and $a_j$ depends on the initial $\omega_j(0)$.

### Key Quantitative Results
*   **Performance:** MSE-trained networks achieved test accuracies comparable to CE-trained networks. The median improvement in accuracy from the moment zero training error was achieved to the end of training was $0.962$ percentage points (mean: $1.833$).
*   **Robustness:** Training beyond zero error improved robustness; the median improvement in the robustness measure from the first zero-error epoch to the last epoch was $0.1762$ (mean: $0.9278$).
*   **Convergence:** $\mathcal{L}_{LS}^\perp$ was found to be negligible during training, confirming that the network stays close to the central path.
*   **Asymptotics:** As $t \to \infty$, all non-zero singular values $\omega_j(t)$ grow to infinity at a rate of $\sqrt[4]{t/c_3}$, and the ratio $\frac{\max \omega_j}{\min \omega_j} \to 1$, implying the emergence of NC.

### Stated Limitations
*   **Scaling Heuristics:** The authors noted that datasets with more than ten classes (e.g., CIFAR100, ImageNet) require additional scaling-heuristics for the MSE loss to achieve comparable performance, which were outside the scope of this analysis.
*   **Outlier Behavior:** STL10-ResNet and STL10-DenseNet exhibited slower convergence or inconsistent trends, potentially due to larger image sizes ($96 \times 96$) creating a harder optimization problem.
*   **Theoretical Assumptions:** The analysis assumes that the within-class covariance $\Sigma_W$ remains full-rank throughout training.
