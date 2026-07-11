---
id: arxiv:2305.10193
type: paper
title: Fine-Tuning Can Degrade Pretrained Language Models
url: https://arxiv.org/abs/2305.10193
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

**Core Problem**
Extracting the thermal Sunyaev-Zeldovich (tSZ) Compton-$y$ parameter map from microwave observations is fundamentally challenged by the faint tSZ signal relative to Galactic thermal dust, cosmic infrared background (CIB), and instrumental noise/systematics. Previous Planck PR2 maps (2015) exhibit residual large-scale striping from $1/f$ noise, elevated thermal noise, and significant foreground leakage. This study addresses the need for an updated, higher-fidelity tSZ map by leveraging the Planck PR4 (NPIPE) data release, which incorporates additional repointing data, improved glitch flagging, and advanced destriping to suppress systematic artifacts.

**Method/Recipe Step by Step**
The reconstruction follows a tailored Needlet Internal Linear Combination (NILC) pipeline:
1. Subtract the dipole and dipole-induced quadrupole from the nine PR4 frequency maps (30–857 GHz) using Commander templates.
2. Apply a processing mask (NILC-MASK) excluding the brightest 2% of pixels at 857 GHz along the Galactic ridge to prevent spherical harmonic ringing.
3. Deconvolve native instrumental beams and reconvolve with a common $10'$ Gaussian beam in harmonic space.
4. Define 10 Gaussian-shaped needlet window functions $h_\ell^j$ in harmonic space satisfying $\sum_j (h_\ell^j)^2 = 1$ to localize data across angular scales.
5. Compute needlet coefficients $\gamma^{j,\nu}(p)$ for each frequency and needlet scale.
6. Estimate a local empirical covariance matrix $C_j(p)$ using a Gaussian convolution kernel $K_j(p,p')$ over neighboring pixels.
7. Calculate NILC weights $w_j^\nu(p)$ using the tSZ SED coefficients $g_\nu$ and the inverse covariance matrix to minimize variance.
8. Combine needlet maps across frequencies to obtain scale-dependent estimates $\hat{y}_j(p)$.
9. Transform estimates back to harmonic space and sum over needlet scales to produce the final $y$-map $\hat{y}(p)$.
10. Apply identical weights to half-ring splits (HR1/HR2) to generate noise maps and compute noise-free cross-power spectra.

**Key Formulas**
The tSZ temperature distortion follows $\Delta T_{\mathrm{tSZ}} (\nu , \hat {\mathbf {n}}) = g (\nu) y (\hat {\mathbf {n}})$, where the non-relativistic SED coefficient is $g (\nu) = T _ {\mathrm{CMB}} \left(x \coth \left(\frac {x}{2}\right) - 4\right)$ with $x = h \nu / k T _ {\mathrm{CMB}}$. The Compton-$y$ parameter is $y (\hat {\mathbf {n}}) = \frac {\sigma_ {T}}{m _ {\mathrm{e}} c ^ {2}} \int_ {\text { los }} n _ {\mathrm{e}} k T _ {\mathrm{e}} (\hat {\mathbf {n}}, l) d l$. The data model is $x (\nu , p) = g (\nu) y (p) + n (\nu , p)$. Beam correction uses $a _ {\ell m} ^ {\nu} = x _ {\ell m} ^ {\nu} \times \frac {b _ {\ell} ^ {\text {gauss}}}{b _ {\ell} ^ {\text {PR4}} [ \nu ]}$. Needlet coefficients are $\gamma^ {j, \nu} (p) = \sum_ {\ell , m} h _ {\ell} ^ {j} a _ {\ell m} ^ {\nu} Y _ {\ell m} (p)$, with optimal weights $w _ {j} ^ {\nu} (p) = \frac {\sum_ {\nu^ {\prime}} g _ {\nu^ {\prime}} \left[ C _ {j} ^ {- 1} (p) \right] ^ {\nu \nu^ {\prime}}}{\sum_ {\nu , \nu^ {\prime}} g _ {\nu^ {\prime}} \left[ C _ {j} ^ {- 1} (p) \right] ^ {\nu \nu^ {\prime}} g _ {\nu}}$. Covariance is estimated as $C _ {j} ^ {\nu \nu^ {\prime}} (p) = \sum_ {p ^ {\prime}} K _ {j} (p, p ^ {\prime}) \gamma_ {j} ^ {\nu} (p ^ {\prime}) \gamma_ {j} ^ {\nu^ {\prime}} (p ^ {\prime})$, yielding $\hat{y}(p) = \sum_{\ell,m} \left( \sum_j \hat{y}_{\ell m,j} h_\ell^j \right) Y_{\ell m}(p)$.

**Key Quantitative Results**
The PR4 NILC $y$-map covers 98% of the sky. Residual noise variance is reduced by 6.8% (standard deviation $\sigma = 1.12 \times 10^{-6}$ vs. $1.16 \times 10^{-6}$), with a mean noise power reduction of 6.7% across $\ell=30-2048$. The 1-PDF variance decreases by 17%, and the derived cosmological parameter $\sigma_8 = 0.76 \pm 0.02$ matches PR2 consistency. CIB contamination is reduced by 34.2% over 50% of the sky (and 56.7% over 15% using an independent template). The NILC pipeline achieves a signal-to-noise ratio of 178.2 over $\ell=30-1000$, a factor of three improvement over a tested Harmonic ILC implementation (SNR = 61.2), validating the necessity of spatial localization for tSZ reconstruction.

**Stated Limitations**
Relativistic SZ corrections are deliberately neglected in this release and will be addressed in future work. Residual Galactic thermal dust emission remains significant near the Galactic plane, requiring masking for cosmological power spectrum analyses. Diffuse CIB contamination persists at small angular scales and cannot be fully mitigated via masking due to its all-sky distribution. Additionally, while the current map is optimized for general tSZ studies, future cross-correlation analyses with large-scale structure may require constrained ILC methods to deproject CIB moments, which will inherently increase overall noise variance.
