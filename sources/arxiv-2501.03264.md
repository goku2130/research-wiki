---
id: arxiv:2501.03264
type: paper
title: Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization
url: https://arxiv.org/abs/2501.03264
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-model-overoptimization
---

# Summary: Bridge the Inference Gaps of Neural Processes via Expectation Maximization

### Core Problem
Neural Processes (NPs) are computationally efficient models for learning distributions over functions, but they often suffer from under-fitting and suboptimal performance. The authors identify the root cause as **inference suboptimality** stemming from the optimization objective. Specifically, vanilla NPs utilize an approximate Evidence Lower Bound (ELBO) that introduces a "prior approximation gap" because the real functional prior is replaced by an approximate one. This gap means that optimizing the vanilla NP objective does not guarantee the maximization of the target marginal log-likelihood of the meta-dataset.

### Method: Self-normalized Importance weighted Neural Process (SI-NP)
To resolve this, the authors propose the **Self-normalized Importance weighted Neural Process (SI-NP)**, which utilizes a variational expectation maximization (VEM) framework to optimize a surrogate objective that provides an improvement guarantee for the target log-likelihood.

#### Step-by-Step Recipe
The meta-training process follows an iterative VEM algorithm:
1. **Initialization**: Initialize model parameters $\vartheta$ (encoder/decoder) and proposal distribution parameters $\eta$.
2. **E-step #1**: Reset the variational posterior $q_\phi(z) = p(z|\mathcal{D}_{\tau}^T; \vartheta_k)$ based on the current model parameters.
3. **E-step #2 (Optional)**: Update the proposal distribution $q_\eta(z|\mathcal{D}_{\tau}^T)$ by minimizing the KL divergence between the proposal and the exact posterior to stabilize training.
4. **M-step**: Optimize the surrogate function $\mathcal{L}_{\text{SI-NP}}$ to update the model parameters $\vartheta_{k+1}$.
5. **Iteration**: Repeat until convergence, where the difference in log-likelihood between iterations $\mathcal{L}(\vartheta_H) - \mathcal{L}(\vartheta_{H-1})$ approaches zero.

#### Key Formulas
The target objective is to maximize the marginal log-likelihood:

$$
\mathcal{L}(\vartheta) = \ln \left[ \int p(\mathcal{D}_{\tau}^T | z; \vartheta) p(z | \mathcal{D}_{\tau}^C; \vartheta) dz \right]
$$

Because the posterior is intractable, the authors use self-normalized importance sampling (SNIS) to define the SI-NP surrogate objective:

$$
\mathcal{L}_{\text{SI-NP}}(\vartheta; \eta_k, \vartheta_k) = \sum_{b=1}^B \hat{\omega}^{(b)} \left[ \ln p(\mathcal{D}_{\tau}^T | z^{(b)}; \vartheta) + \ln p(z^{(b)} | \mathcal{D}_{\tau}^C; \vartheta) \right]
$$

Where the importance weights are calculated as:

$$
\omega^{(b)} = \exp \left( \ln p(\mathcal{D}_{\tau}^T | z^{(b)}; \vartheta_k) + \ln p(z^{(b)} | \mathcal{D}_{\tau}^C; \vartheta_k) - \ln q_{\eta_k}(z^{(b)} | \mathcal{D}_{\tau}^T) \right)
$$

$$
\hat{\omega}^{(b)} = \frac{\omega^{(b)}}{\sum_{b'=1}^B \omega^{(b')}}
$$

And samples $z^{(b)}$ are drawn from the proposal distribution $q_{\eta_k}(z | \mathcal{D}_{\tau}^T)$.

### Key Quantitative Results
The SI-NP was evaluated against vanilla NP, Conditional NP (CNP), and ML-NP across synthetic and image datasets.

**Synthetic Regression (Average Log-Likelihood):**
SI-NP consistently outperformed vanilla NPs across different kernels:
*   **RBF Kernel**: SI-NP achieved $0.493 \pm 0.007$ vs. NP's $-0.183 \pm 0.03$.
*   **Matern-5/2 Kernel**: SI-NP achieved $0.305 \pm 0.006$ vs. NP's $-0.225 \pm 0.03$.
*   **Periodic Kernel**: SI-NP achieved $-0.532 \pm 0.036$ vs. NP's $-0.611 \pm 0.034$.

**Image Completion (Target Log-Likelihood):**
SI-NP achieved the best performance across all tested image datasets:
*   **MNIST**: $1.02 \pm 0.004$ (SI-NP) vs. $0.73 \pm 0.007$ (NP).
*   **FMNIST**: $0.94 \pm 0.005$ (SI-NP) vs. $0.73 \pm 0.009$ (NP).
*   **SVHN**: $3.50 \pm 0.003$ (SI-NP) vs. $3.07 \pm 0.02$ (NP).
*   **CIFAR10**: $2.60 \pm 0.005$ (SI-NP) vs. $2.03 \pm 0.02$ (NP).

### Stated Limitations
*   **Computational Cost**: SI-NP requires multiple Monte Carlo samples during training, making it more computationally expensive than vanilla NPs.
*   **Optimization Stability**: Updating the proposal distribution (E-step #2) was found to be empirically tricky and potentially unstable, leading the authors to make it optional.
*   **Theoretical Gap**: The authors note that while the functional prior's uncertainty is propagated to the output, the specific mathematical influence of this uncertainty has not yet been studied.
