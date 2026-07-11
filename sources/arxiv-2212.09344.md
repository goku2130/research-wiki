---
id: arxiv:2212.09344
type: paper
title: 'Code as Policies: Language Models as Program Synthesis'
url: https://arxiv.org/abs/2212.09344
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

**Core Problem**
The study investigates the physical drivers of gas-phase metallicity distributions and radial gradients in star-forming galaxies. While massive galaxies typically exhibit negative metallicity gradients and a tight resolved mass-metallicity relation (rMZR), the relative predictive power of local stellar mass surface density ($\Sigma_*$) versus galactocentric radius ($R$) remains unresolved. The research specifically tests whether these local dependencies vary across the galaxy mass-size plane and whether large-scale environmental factors can account for observed variations in metallicity behavior.

**Methodology (Step-by-Step)**
1. **Data Acquisition & Sample Selection:** Integral-field spectroscopy from SDSS-IV MaNGA (DR17) was used. Galaxies from the MPL-10 release were selected with elliptical Petrosian axis ratios $b/a \geq 0.6$ to minimize edge-on projection effects and cross-matched with the GALEX-SDSS-WISE Legacy Catalog for stellar masses.
2. **Spaxel Identification:** Star-forming spaxels were isolated between $0.5\,R_e$ and $2\,R_e$ using the BPT-NII diagnostic diagram and a minimum H$\alpha$ equivalent width of 10. A signal-to-noise ratio $\geq 3$ was required for H$\alpha$, H$\beta$, [OIII]$\lambda$5007, [NII]$\lambda$6583, and [OII]$\lambda\lambda$3727,3729. Galaxies with fewer than 20 qualifying spaxels were discarded.
3. **Metallicity Calibration:** Gas metallicities were computed from DAP emission line fluxes using the Marino et al. (2013) O3N2 calibrator. Supplementary analyses employed the Pilyugin & Grebel (2016) R2 calibrator to test calibrator dependence.
4. **Correlation Analysis & Error Perturbation:** Galaxy-by-galaxy Spearman rank correlation coefficients were calculated: $\rho_\Sigma$ for $\Sigma_*$-metallicity and $\rho_R$ for radius-metallicity. To fairly compare the two, spaxel radii were perturbed with a Gaussian scatter $\Delta_R$ to approximate the measurement uncertainty in $\Sigma_*$.
5. **Subsampling & Environmental Cross-Matching:** The sample was divided into six subsamples across the mass-size plane to analyze two-dimensional metallicity trends. Environmental parameters ($\rho_{3\mathrm{Mpc}}$, $\rho_{9\mathrm{Mpc}}$, $D_{\mathrm{skel}}$, $D_{\mathrm{node}}$) were incorporated from Duckworth et al. (2019) to assess large-scale influences.

**Key Formulas**
Gas metallicity is quantified via the oxygen abundance indicator:

$$
12 + \log(\mathrm{O}/\mathrm{H}) = 8.533 - 0.214 \times \mathrm{O}3\mathrm{N}2,
$$

where the O3N2 ratio is defined as:

$$
\mathrm{O}3\mathrm{N}2 = \log \left(\frac{[\mathrm{OIII}]\lambda5007}{\mathrm{H}\beta} \times \frac{\mathrm{H}\alpha}{[\mathrm{NII}]\lambda6583}\right).
$$

The radius perturbation applied to match $\Sigma_*$ measurement scatter is calculated as:

$$
\Delta_R = 0.1\,\mathrm{dex} \times \frac{\sigma_{\log(r)}}{\sigma_{\log(\Sigma)}},
$$

utilizing observed dispersions $\sigma_{\log(r)} = 0.149\,\mathrm{dex}$ and $\sigma_{\log(\Sigma)} = 0.409\,\mathrm{dex}$.

**Quantitative Results**
The final sample comprises 2,127 galaxies containing 860,214 star-forming spaxels, with 95% exhibiting specific star formation rates above $10^{-11}\,\mathrm{yr}^{-1}$. Analysis reveals that in extended, lower-mass galaxies, gas metallicity correlates more strongly with galactocentric radius ($|\rho_R| > |\rho_\Sigma|$), whereas in higher-mass, compact systems, density and radius are nearly equally predictive. These observational trends are not reproduced by prior "base" or "star formation history" (SFH) models, which consistently predict stronger density-metallicity correlations across the mass-size plane. Bulge-to-total (B/T) ratios strongly correlate with gradient morphology: low B/T extended galaxies exhibit steep negative gradients, while high B/T compact galaxies show flattened gradients. Environmental analysis indicates that dense environments are associated with marginally flatter gradients, but environmental metrics trend predominantly with stellar mass rather than size, rendering large-scale environment insufficient to explain the observed mass-size plane variations.

**Stated Limitations**
The study acknowledges that the O3N2 calibrator implicitly assumes a fixed N/O–O/H relation, which may introduce biases in metallicity gradient measurements. Additionally, MaNGA point-spread function (PSF) beam smearing, particularly for galaxies observed with smaller 19- and 37-bundle IFUs, can artificially flatten measured gradients. While the authors argue that trends persist when controlling for physical size at fixed angular scales, they note that explicit angular size cuts would introduce selection biases. Furthermore, environmental data were restricted to an earlier MaNGA data release (MPL-6), limiting the environmental analysis to 1,726 galaxies (81.1% of the sample). Finally, the inability of existing local relation models to replicate the observed radius-dominated metallicity trends in extended galaxies highlights a gap in current chemical evolution frameworks.
