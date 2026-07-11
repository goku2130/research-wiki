---
id: arxiv:2305.18723
type: paper
title: Generative models are problematic when they are trained on their own outputs
url: https://arxiv.org/abs/2305.18723
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

**Core Problem**
Diffusion models require hundreds of iterative denoising steps, imposing severe computational burdens for deployment on resource-constrained hardware. While post-training quantization (PTQ) accelerates inference by substituting full-precision arithmetic with integer operations, conventional data-free PTQ frameworks apply shared quantization functions across all generation timesteps and randomly sample timesteps for calibration. This approach ignores the significant variation in activation distributions across timesteps, resulting in substantial quantization errors and marked degradation in image synthesis quality.

**Methodology and Recipe**
The proposed Accurate Post-training Quantization for Diffusion Models (APQ-DM) addresses these issues through a structured pipeline:
1. **Timestep Partitioning:** The $T$ denoising timesteps are divided into $G$ groups. Each group is assigned a distinct quantization function characterized by a scale parameter and clipping bounds.
2. **Differentiable Group Assignment:** To avoid NP-hard combinatorial optimization and overfitting, the method employs a differentiable relaxation. Activations are quantized using all $G$ functions simultaneously, and the outputs are aggregated using learnable importance weights $\sigma_g$ (normalized to unity). After training, the function with the highest weight is selected for each timestep.
3. **Joint Optimization:** Quantization parameters and importance weights are updated jointly. The objective combines the diffusion noise estimation loss with an entropy regularization term that penalizes non-deterministic weight distributions, effectively pushing $\sigma_g$ toward one-hot vectors.
4. **Active Timestep Selection:** Calibration images are generated at optimally selected timesteps rather than randomly. This selection generalizes the Structural Risk Minimization (SRM) principle, balancing the empirical risk of the quantized model against the Maximal Mean Discrepancy (MMD) between the full and sampled timestep distributions. The MMD is approximated via an Upper Confidence Bound (UCB) principle that prioritizes timesteps with fewer prior sampling attempts.
5. **Deployment:** The finalized group-wise quantization functions are applied to the pre-trained diffusion decoder for efficient integer-arithmetic inference.

**Key Formulas**
The group-specific quantization function is defined as:

$$
\hat{x} = s_{g(i)} \cdot \Phi([x/s_{g(i)}], z_{min}^{g(i)}, z_{max}^{g(i)})
$$

The differentiable relaxation sums all group outputs:

$$
\hat{x} = \sum_{g=1}^{G} \sigma_g s_g \cdot \Phi([x/s_g], z_{min}^g, z_{max}^g)
$$

The joint optimization objective minimizes the noise estimation loss $J_d$ and entropy loss $J_e$:

$$
\min_{t, \boldsymbol{x}_0, \boldsymbol{\epsilon}} J = ||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta||_2^2 + \lambda \sum_{g=1}^{G} -\sigma_g^t \log \sigma_g^t
$$

The SRM-based timestep selection criterion maximizes:

$$
\max_t s = \sum_{g=1}^G -\sigma_g^t \log \sigma_g^t + \frac{\eta}{N_t + 1}
$$

where $N_t$ is the sampling count for timestep $t$, and $\lambda, \eta$ are hyperparameters.

**Quantitative Results**
Evaluated on DDIM and Latent Diffusion Models (LDMs) across Cifar-10, CelebA, LSUN-Bedroom, LSUN-Church, and ImageNet, APQ-DM consistently outperforms state-of-the-art PTQ methods (LSQ, PTQ4DM, Q-Diffusion). For 8-bit quantization (W8A8) on LSUN-Bedroom, APQ-DM achieves an Inception Score (IS) of 2.55 and FID of 6.46, surpassing PTQ4DM’s 2.23 IS and 7.48 FID. For 6-bit quantization (W6A6), the advantage widens: on Cifar-10, APQ-DM reaches IS 9.06 and FID 6.57 versus PTQ4DM’s 8.72 IS and 11.28 FID. On class-conditional ImageNet, APQ-DM attains IS 178.64 and FID 11.58, significantly exceeding PTQ4DM’s 140.86 IS and 13.68 FID. Ablation studies confirm that partitioning timesteps into 8 groups yields the optimal accuracy-generalization trade-off, and active timestep sampling drastically reduces FID compared to random or heuristic sampling, particularly with small calibration sets (e.g., 4.49 FID vs. 5.34 FID at 512 calibration images for W8A8).

**Stated Limitations**
The framework acknowledges that while partitioning timesteps reduces quantization error, excessively fine partitions risk overfitting due to the limited size of data-free calibration sets. Additionally, the differentiable search requires joint optimization of quantization parameters and importance weights, introducing a training phase that, while described as having negligible computational overhead compared to full model training, demands careful hyperparameter tuning ($\lambda=0.8, \eta=1.5$) to balance discretization regularization and distribution matching. The method also remains dependent on pre-trained full-precision diffusion decoders and does not address architectural compression beyond quantization.
