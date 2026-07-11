---
id: arxiv:2305.06950
type: paper
title: 'Orca: A Distributed Serving System for Transformer-Based Generative Models'
url: https://arxiv.org/abs/2305.06950
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

**Core Problem**
The study investigates the discovery and characterization of TOI-2498 b, a transiting hot super-Neptune residing near the lower boundary of the "Neptune desert"—a region in mass-period parameter space exhibiting a pronounced dearth of close-orbiting Neptune-sized planets. The primary objective is to constrain the planet's physical properties, internal structure, and photoevolutionary history to better understand the formation mechanisms and atmospheric loss processes that define the boundaries of this planetary population.

**Methodology**
The research employs a sequential observational and analytical pipeline:
1. **Data Acquisition:** Photometry was gathered from TESS (sectors 6 and 33) and ground-based LCOGT telescopes. High-angular-resolution imaging was performed using SOAR speckle imaging and Shane adaptive optics to rule out blended background eclipsing binaries. Radial velocity (RV) spectroscopy was obtained via HARPS.
2. **Stellar Characterization:** Atmospheric parameters ($T_{\text{eff}}$, $\log g$, [Fe/H]) were derived from HARPS spectra using ARES+MOOG equivalent width measurements. Chemical clock abundance ratios provided an independent age estimate. A spectral energy distribution (SED) fit using PHOENIX models and Gaia DR3 parallaxes independently validated the stellar radius and mass.
3. **Joint Modeling:** Flattened TESS/LCOGT light curves and HARPS RV data were combined in a Bayesian framework using `exoplanet`, `starry`, and `PYMC3`. Transit models utilized Keplerian orbits with quadratic limb-darkening and exposure time effects. The RV model fitted for the semi-amplitude $K$, instrument offset, and white noise jitter. Posterior sampling employed a No U-Turn Sampler (NUTS) variant of Hamiltonian Monte Carlo.
4. **Internal Structure & Evaporation:** A coupled system of equations was solved assuming a rocky core and H/He envelope. The planet's evaporation history was simulated using `photoevolver`, integrating stellar XUV evolution tracks (Johnstone et al. 2021), photoevaporation mass-loss rates (Kubyshkina et al. 2018), and envelope structure models (Chen & Rogers 2016).

**Key Formulas**
The internal structure decomposition relies on two explicit definitions:
\[
f_{\text{env}} = \frac{M_{\text{env}}}{M_{p}} = \frac{M_{p} - M_{\text{core}}}{M_{p}}
\]
\[
R_{\text{env}} = R_{p} - R_{\text{core}}
\]
These relations are coupled with rocky core mass-radius relations and an envelope structure model to solve for core mass, core radius, envelope thickness, and envelope mass fraction.

**Quantitative Results**
The host star TOI-2498 has $M_\star = 1.12 \pm 0.02\,M_\odot$, $R_\star = 1.26 \pm 0.04\,R_\odot$, $T_{\text{eff}} = 5905 \pm 12\,\text{K}$, and an age of $3.6 \pm 1.1\,\text{Gyr}$. The planet TOI-2498 b orbits with a period of $3.738252 \pm 0.000004\,\text{days}$, a radius of $6.06^{+0.29}_{-0.27}\,R_\oplus$, a mass of $34.62^{+4.10}_{-4.09}\,M_\oplus$, and a bulk density of $0.86^{+0.25}_{-0.20}\,\text{g cm}^{-3}$. Internal structure modeling yields a rocky core of $25.12 \pm 3.41\,M_\oplus$ surrounded by a gaseous envelope comprising $27 \pm 4\%$ of the total mass. Evaporation simulations indicate the planet formed as a puffy Saturn-sized object ($8-10\,R_\oplus$ with a $30-45\%$ envelope) and has remained largely stable over its $\sim 3.6\,\text{Gyr}$ lifespan, experiencing minimal atmospheric stripping.

**Stated Limitations**
The analysis acknowledges several constraints. The internal structure model explicitly assumes negligible water content and atmospheric metallicity; TOI-2498 b's intermediate size likely places it between dry sub-Neptune and high-water mixed interior evolutionary pathways. The stellar rotation period was not measured, necessitating reliance on age-inferred XUV evolution models rather than direct activity tracking. Transit timing variation (TTV) analysis was limited by the available number of transits, and discrepancies in ground-based TTV offsets were attributed to high airmass effects. Furthermore, the observed boundaries of the Neptune desert are subject to detection biases favoring larger, more massive planets, potentially blurring the lower mass-period boundary and requiring cautious interpretation of the desert's exact limits.
