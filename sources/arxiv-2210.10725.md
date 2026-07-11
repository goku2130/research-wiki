---
id: arxiv:2210.10725
type: paper
title: Scaling Laws for Reward Model Overoptimization
url: https://arxiv.org/abs/2210.10725
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

**Core Problem**
Deep neural networks (DNNs) are foundational for click-through rate (CTR) prediction but suffer from convergence instability and a lack of guaranteed smoothness. While extensive research optimizes feature-crossing architectures, DNN optimization is consistently neglected. Standard residual connections enhance smoothness but require uniform layer dimensions, conflicting with the tower-like DNN structures common in CTR systems. Furthermore, residual networks rely on Batch Normalization, which is incompatible with industrial online learning environments that utilize indeterminate or single-sample batch sizes. Additionally, shallow layers in trained networks are highly sensitive to noise, and deep networks frequently suffer from rank collapse and dead neurons, severely limiting expressive capacity.

**Methodology**
The proposed Skip Meta Logit (SML) module addresses these challenges through a two-step architectural integration. First, the Skip Logit mechanism flattens the residual structure by directly adding each layer’s output to the final prediction logit, bypassing the need for identical layer dimensions. This design ensures that shallow layers directly influence the output, facilitating stable gradient flow across arbitrary DNN shapes. Second, Meta Tanh Normalization (MTN) stabilizes training under small-batch or online learning conditions. MTN adaptively rescales layer variances using a meta-learner that processes stop-gradient feature maps, combined with a Tanh activation to cap variance and smooth gradients. This prevents activation saturation and mitigates noise sensitivity in early layers.

**Key Formulas**
The Skip Logit output is mathematically defined as $\operatorname{logit} = x_{0} + \sum_{i=1}^{L-1} F(x_{i}, W_{i}^{L-1})$, where $x_i$ denotes the $i$-th layer output and $L$ is the total skip paths. This architecture yields linear variance growth: $\operatorname{Var}(f(x)) = L \times \operatorname{Var}(x)$. The MTN operation is formulated as $x_{norm} = W \odot \text{Tanh}(s(x) \odot x)$, where $W$ is a learnable scale parameter and $s(x)$ is the meta variance learner. To reduce computational overhead, the variance learner operates on a stop-gradient input $x'$: $s(x) = \text{leakyrelu}(w \times x')$. The Tanh activation bounds variance via $\operatorname{Var}(\tanh(x)) \leq 1$, resulting in a stabilized final variance of $\operatorname{Var}(f(x)) \leq \operatorname{Var}(x) + L - 1$. Theoretically, the optimization landscape satisfies $\|\nabla f(A)\|_F^2 \geq 4 \sum_{i=1}^{l} (1 - \gamma_i)^2 \delta_{\min}(\Sigma) \|f(A) - C_{opt}\|_F^2$, proving that under specific spectral constraints, all critical points are global minima.

**Quantitative Results**
Evaluated on Criteo (45M samples) and Avazu (40M samples), SML universally improves state-of-the-art CTR models. DCN-V2 with SML achieved the highest offline AUC of approximately 0.8117 on Criteo. Ablation studies confirm MTN outperforms fixed-weight Tanh scaling by ~0.006%, while the full SML module improves baseline DNN AUC by ~0.01%. Property analysis reveals that without SML, logit variance fades by depth 5, and cosine similarity between samples approaches 1, indicating rank collapse. SML preserves sample distinguishability up to depth 15 and maintains flexible kernel activation up to depth 100. In industrial A/B tests, SML deployment yielded an AUC increase of +0.07% and advertiser value (ADVV) uplift of +3.28% in a vertical ad scenario, alongside +0.15% AUC and +1.13% ADVV in main traffic.

**Stated Limitations**
The theoretical guarantee of a spurious-local-optima-free landscape relies on linear assumptions and specific spectral norm constraints ($\gamma_i < 1$). While SML enables training up to 100 layers, empirical results indicate diminishing performance returns beyond depth 4. The meta-variance learner currently utilizes stop-gradient inputs rather than real-time online statistics, a limitation acknowledged for future work. Additionally, specific industrial deployment metrics and system details are partially redacted in the source.
