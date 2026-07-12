---
id: arxiv:2401.06571
type: paper
title: 'CPO: Contrastive Preference Optimization for LLM Alignment'
url: https://arxiv.org/abs/2401.06571
retrieved: '2026-07-12'
maturity: comprehensive
topic: dpo-variants
---

# Dark Matter Search in Dwarf Irregular Galaxies with IceCube Data

### Core Problem
The study aims to detect neutrinos produced by the annihilation of ultra-heavy dark matter (UHDM) in dwarf irregular galaxies (dIrrs). While dwarf spheroidal galaxies (dSphs) are common targets for indirect dark matter detection, they are pressure-supported systems with high kinematic uncertainty, leading to significant systematic errors in their astrophysical J-factors. In contrast, dIrrs are rotationally supported, allowing for more reliable determinations of their dark matter (DM) density profiles and J-factors, making them robust complementary targets for searching for DM in the 1 TeV to 10 PeV mass range.

### Method and Recipe
The researchers employed a 10-year dataset (April 2008 to July 2018) of IceCube muon-track events, totaling 1,134,450 events. The analysis followed these steps:

1.  **Target Selection**: Seven dIrrs were selected: IC10, IC1613, NGC6822, WLM, DDO133, DDO154, and DDO168.
2.  **Signal Modeling**: 
    *   **Mass and Channels**: DM masses were considered from 1 TeV to 10 PeV across six annihilation channels: $b\bar{b}, t\bar{t}, \tau^+\tau^-, \nu_\mu\bar{\nu}_\mu, \mu^+\mu^-,$ and $W^+W^-$.
    *   **DM Profile**: The DM distribution was modeled using a Burkert profile.
    *   **Enhancement**: A boost factor ($B$) was included to account for the signal enhancement from internal DM subhalos.
    *   **Source Extension**: dIrrs were analyzed as extended sources using a source extension $\sigma_s$ corresponding to the virial radius.
3.  **Statistical Analysis**: An unbinned maximum-likelihood-ratio method was used to search for signals beyond the background.
    *   **Likelihood Construction**: The likelihood function $\mathcal{L}(n_s)$ combined signal and background probability density functions (PDFs) for reconstructed direction and muon energy.
    *   **Test Statistic**: A likelihood ratio test statistic (TS) was used to evaluate the significance of putative signals.
    *   **Joint Analysis**: To increase sensitivity, a combined likelihood analysis was performed for the five northern-sky sources.

### Key Formulas
The expected differential neutrino flux for DM annihilation is given by:

$$
\frac{d\Phi_{\nu}}{d E_{\nu}}\left(E_{\nu}\right)=\frac{\left\langle\sigma v\right\rangle}{8\pi m_{\chi}^{2}}\frac{d N_{\nu}}{d E_{\nu}}\left(E_{\nu}\right)J_{\left(\Delta\Omega\right)}\times B
$$

Where $\langle\sigma v\rangle$ is the velocity-averaged annihilation cross section, $m_\chi$ is the DM mass, and $J$ is the astrophysical factor.

The DM distribution follows the Burkert profile:

$$
\rho_{\mathrm{D M}}(r)=\rho_{0}\frac{r_{0}^{3}}{\left(r_{0}+r\right)\left(r_{0}^{2}+r^{2}\right)}
$$

The likelihood function for the number of signal events $n_s$ is:

$$
\mathcal{L}\left(n_{s}\right)=\prod_{k}\prod_{i=1}^{N^{k}}\left[\frac{n_{s}^{k}}{N^{k}}S_{i}^{k}+\left(1-\frac{n_{s}^{k}}{N^{k}}\right)B_{i}^{k}\right]
$$

The test statistic is defined as:

$$
\mathrm{TS}=-2\log\left\lfloor\frac{L\left(n_{s}\right)}{L\left(n_{s}=0\right)}\right\rfloor
$$

### Quantitative Results
The study found no significant neutrino signal excess beyond the background in any of the seven dIrrs. Consequently, 95% confidence level (C.L.) upper limits were set on $\langle\sigma v\rangle$ for DM masses between 1 TeV and 10 PeV.

*   **Strongest Constraints**: The most stringent limits were obtained for the $\chi\chi \to \mu^+\mu^-$ annihilation channel.
*   **Source Performance**: Among individual galaxies, DDO154 provided the strictest exclusion limits.
*   **Joint Analysis**: The joint analysis of the five northern-sky sources yielded constraints approximately 2 to 3 times stronger than those of DDO154 alone.
*   **Hemisphere Sensitivity**: Southern sky sources (WLM and NGC6822) provided weaker limits for DM masses below $\simeq 10$ TeV due to stringent cuts required to suppress atmospheric muon backgrounds. However, they provided stronger constraints for very large DM masses ($> 10^6$ GeV) due to the larger effective area of the southern sky at high energies.

### Stated Limitations
The authors identify several sources of uncertainty:
*   **J-factor Uncertainties**: Statistical errors in the DM profile parameters ($\rho_0, r_0$) lead to a $\sim 75\%$ uncertainty in the J-factor. Systematic uncertainties regarding the choice of the density profile (e.g., Burkert vs. NFW) contribute approximately $0.2$ dex.
*   **Boost Factor**: The largest uncertainty arises from the nature of DM substructures; the inclusion of the boost factor $B$ results in a difference of $\sim 5$ in the final results.
*   **Relative Sensitivity**: The constraints are generally weaker than those derived from dSphs because dIrrs are more distant and possess smaller J-factors.
