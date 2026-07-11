---
id: arxiv:2605.18838
type: paper
title: 'Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling'
url: https://arxiv.org/abs/2605.18838
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

**Core Problem**
Scaling laws accurately forecast validation loss but fail to capture how distinct capabilities interact during model growth. The widely assumed "alignment tax"—the phenomenon where improving reasoning degrades truthfulness in smaller models—is treated as an inevitable consequence of scale. This work investigates whether this tax represents a fundamental physical constraint or an engineerable bottleneck that remains invisible to standard loss curves, which continue to decrease smoothly across regime boundaries.

**Methodology / Recipe Step by Step**
The Capability Coupling Analysis of Phase Emergence (CAPE) framework operationalizes phase diagnosis through a sequential protocol: (1) compute local coupling $\gamma_{12}(N) \equiv \Delta B_2/\Delta B_1$ between consecutive model sizes within a family, and population coupling $r(B_1, B_2)$ across a model panel; (2) identify the critical scale $N_c$ where $\gamma_{12}$ crosses zero; (3) apply engineering levers—including width normalization, data curation, and architectural changes—to shift $N_c$; (4) probe internal representations using TransformerLens to measure per-layer hidden-state cosine similarity and attention head competition; (5) discover governing dynamics via sparse regression (PySINDy) on benchmark trajectories; and (6) execute a zero-retraining intervention by adding a truth-direction vector at the quarter-depth probe layer during inference.

**Key Formulas**
The coupling transition follows the logarithmic law:

$$
\gamma_{12}(N, \mathcal{D}) = \gamma_0(\mathcal{D}) \cdot \log_{10}(N/N_c(\mathcal{D}))
$$

Benchmark trajectories are modeled by the discovered ODE:

$$
\frac{dB_i}{d\log_{10}N} = \sum_j c_{ij}B_j + \sum_{j \le k} d_{ijk}B_j B_k
$$

Dimensionality collapse is quantified via the participation ratio:

$$
d_{\text{eff}} = \frac{(\sum_{i=1}^{5} \lambda_i)^2}{\sum_{i=1}^{5} \lambda_i^2}
$$

Width normalization scales scores as:

$$
B_{\text{norm}} = B / (d_{\mathrm{model}}/d_{\mathrm{ref}})
$$

**Key Quantitative Results**
Below $N_c$, reasoning and truthfulness strongly anticorrelate ($r = -0.989$, $p = 4 \times 10^{-5}$ for Pythia), defining the "alignment tax" phase. Above $N_c$, coupling flips to cooperative ($r > +0.78$). The critical scale varies significantly by architecture: $N_c \approx 0.12\text{B}$ for OPT, $3.5\text{B}$ [2.9B, 13.4B] for Pythia, and $7.0\text{B}$ for Falcon. Width normalization eliminates the anticorrelation across all tested families (e.g., Pythia shifts from $-0.989$ to $+0.963$). Internal analysis reveals 38 of 40 models contain zero competing attention heads (95% CI: 84–99%), indicating the bottleneck resides at the output projection rather than in representational space. A sparse-regression ODE cross-predicts held-out Llama-2 benchmarks at 5.6% MAE. Activation steering at the bottleneck layer corrects 60% of misaligned outputs in the tax phase without retraining. At frontier scales, the cooperative regime persists ($r = +0.72$ across 34 models from 10 labs).

**Stated Limitations**
The framework is currently scoped to tested benchmark pairs, and generalization to other capability axes requires independent validation. While width normalization flips correlations, the authors note that dividing bounded scores by a growing denominator could theoretically induce spurious positive correlation, though direct projection-width measurements confirm the bottleneck. Data curation effects cannot be fully isolated from other generational recipe changes. Frontier benchmark scores are predominantly self-reported. The discovered ODE captures dynamics within a phase but fails across transitions, requiring phase-specific fitting. Finally, the claims would be weakened if a family spanning 0.1B–10B parameters exhibited no coupling sign change, a counterexample not yet observed across the 16 tested families.
