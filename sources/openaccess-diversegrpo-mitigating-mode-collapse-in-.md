---
id: openaccess:diversegrpo-mitigating-mode-collapse-in-
type: web
title: 'DiverseGRPO: Mitigating Mode Collapse in Image Generation via Diversity-Aware
  GRPO'
url: https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_DiverseGRPO_Mitigating_Mode_Collapse_in_Image_Generation_via_Diversity-Aware_GRPO_CVPR_2026_paper.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# DiverseGRPO: Mitigating Mode Collapse in Image Generation

**DiverseGRPO** is a training paradigm designed to mitigate mode collapse in image generation models trained via Group Relative Policy Optimization (GRPO). In standard GRPO, models often converge toward a narrow subset of high-reward "safe" patterns, resulting in homogenized outputs (e.g., identical facial features or camera angles) and a loss of visual creativity.

### Core Problem
The authors identify two primary drivers of mode collapse in GRPO-based image generation:
1.  **Reward Modeling:** Traditional GRPO relies on single-sample reward signals. This creates "replicator dynamics" where modes with slightly higher average rewards $\bar{r}_k$ grow in weight $w_k$ while others are eliminated, eventually leading to a unimodal distribution.
2.  **Generation Dynamics:** Standard KL regularization is misaligned with the diffusion process. Diversity is primarily determined during early denoising stages; however, because diffusion variance $\sigma^2$ is largest at the start, the KL penalty is effectively weakest when the "diversity budget" is most critical.

### Method
DiverseGRPO addresses these issues through a dual-pronged approach: a distributional creativity bonus and structure-aware regularization.

#### 1. Distributional Creativity Bonus
Instead of scoring images in isolation, DiverseGRPO introduces a reward based on the semantic structure of the generated group for a given prompt:
*   **Distance Calculation:** Pairwise visual distances $d_{ij}$ between images in a group are computed using the DreamSim model.
*   **Spectral Clustering:** An affinity matrix $A$ is constructed using a Gaussian kernel:

$$
A_{ij} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)
$$

    The images are then partitioned into $k$ clusters using the normalized Laplacian matrix $L = D^{-1/2}AD^{-1/2}$ and k-means clustering on the resulting eigenvectors.
*   **Reward Allocation:** An exploration reward $E_i$ is assigned to images in cluster $C_k$, inversely proportional to the cluster size $n_k$:

$$
E_i = \sqrt{\frac{N}{n_k}}
$$

    where $N$ is the total number of samples. The final reward $\mathsf{R}_i$ combines the quality score $Q_i$ and the bonus via a weighting factor $\beta$:

$$
\mathsf{R}_i = Q_i + \beta \cdot E_i
$$

#### 2. Structure-Aware Regularization (SA-Reg)
To correct the regularization mismatch, the authors replace the uniform KL penalty with a stage-dependent Wasserstein constraint. This concentrates the regularization budget on the early, diversity-critical phase:

$$
\mathcal{L}_{\mathtt{reg}}(t) = \begin{cases} \frac{\|\bar{\mathbf{x}}_{t+\Delta t,\theta} - \bar{\mathbf{x}}_{t+\Delta t,\mathtt{ref}}\|^2}{2}, & t \le K \\ 0, & t > K \end{cases}
$$

where $K$ represents the number of early denoising steps. This forces the model to maintain semantic coverage early on while allowing free optimization for reward fidelity in later stages.

### Key Quantitative Results
DiverseGRPO was evaluated across SD3.5-M and Flux.1-dev backbones using reward models such as PickScore and HPSv3. The method established a new Pareto frontier between image quality and diversity:

*   **Semantic Diversity:** Achieved a $13\% \sim 18\%$ improvement in semantic diversity under matched quality scores.
*   **Metric Gains (SD3.5-M/Pickscore):**
    *   **DreamSim:** $+18.8\%$
    *   **FID:** $+23.3\%$ improvement (lower is better)
    *   **BeyondFID:** $+184.2\%$
    *   **SSIM:** $+25.6\%$
*   **Metric Gains (Flux.1-dev/Pickscore):**
    *   **DreamSim:** $+13.9\%$
    *   **FID:** $+9.1\%$ improvement
*   **Efficiency:** Compared to the baseline (KL=0.01), DiverseGRPO reached the same quality in 400 iterations versus 1,280 iterations, while reducing mode collapse by $9\%$.

### Limitations
The authors note that increasing the number of SA-Reg steps $K$ improves diversity but increases computational costs with diminishing returns. Additionally, while increasing the creativity coefficient $\beta$ boosts exploration, the diversity gains tend to level off after $\beta=5$, suggesting a saturation point in the balance between exploration and exploitation.
