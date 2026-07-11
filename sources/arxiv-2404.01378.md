---
id: arxiv:2404.01378
type: paper
title: A General Theoretical Paradigm to Understand Learning from Human Preferences
url: https://arxiv.org/abs/2404.01378
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

**Core Problem**
The study addresses the challenge of probing dark matter (DM) substructure at mass scales $\sim 10 - 10^8 M_\odot$ and constraining the primordial power spectrum at short comoving wavelengths ($k \sim 10 - 1000 \text{ Mpc}^{-1}$). While prior work established limits on point-like compact objects (MACHOs/PBHs) via stellar heating in ultra-faint dwarf galaxies (UFDs), this work extends the analysis to finite-size, diffuse DM clumps. The core objective is to determine how gravitational scattering from these substructures would dynamically heat stellar populations, thereby setting observational bounds on clump properties, inflationary models, and DM isocurvature perturbations.

**Methodology**
The analysis proceeds through a sequential computational recipe:
1. **Clump Parametrization:** DM substructure is modeled as truncated Navarro-Frenk-White (NFW) profiles characterized by mass $M_{\mathrm{clump}}$, scale density $\rho_s$, and scale radius $r_s$. These clumps constitute a fraction $f_{\mathrm{clump}}$ of the UFD's total DM and are distributed according to a Dehnen profile.
2. **Migration Dynamics:** The authors compute gravitational heat transfer from clumps to the dominant smooth DM background. This energy loss drives inward migration (dynamical friction), increasing the local clump number density at the stellar scale radius $R_{0,\star}$ over the UFD's lifetime.
3. **Stellar Heating Rates:** Direct and tidal heating rates are calculated for stars interacting with migrating clumps. The formalism adjusts the minimum impact parameter $b_{\min}$ and enclosed mass $M_{\mathrm{enc}}(b_{\min})$ to conservatively account for finite clump sizes, ensuring stars either pass outside the clump or interact only with a subsphere of it.
4. **Survival Thresholds:** Clump longevity is evaluated against host halo heating (HHH) and clump-clump harassment (CCH). A clump is deemed destroyed if the cumulative energy transferred from encounters exceeds its gravitational binding energy $E_b$.
5. **Constraint Synthesis:** Heating limits (which exclude excessive stellar expansion) are combined with survival limits (which exclude clumps destroyed before heating stars) to define the viable parameter space.
6. **Cosmological Mapping:** Surviving clump abundances are translated into primordial power spectrum constraints using the Press-Schechter formalism, top-hat/Gaussian filter functions, and a collapse overdensity $\Delta \approx 200$. Any $P(k)$ yielding predicted clump fractions exceeding observational limits is ruled out.

**Key Formulas**
The theoretical framework relies on several critical expressions:
- NFW density profile and clump mass: $\rho_{\mathrm{NFW}}(r) = 4\rho_s(r/r_s)^{-1}(r/r_s+1)^{-2}$ and $M_{\mathrm{clump}} = 8\pi r_s^3 \rho_s(-1+\log 4)$
- Clump-to-smooth-DM heat transfer: $H_{\mathrm{clump}} = 2\sqrt{2\pi} G^2 (1-f_{\mathrm{clump}}) \rho_{\mathrm{DM}} M_{\mathrm{clump}} \frac{\sigma_{\mathrm{clump}}^2}{(\sigma_{\mathrm{DM}}^2+\sigma_{\mathrm{clump}}^2)^{3/2}} \log\left(\frac{b_{90}^2+R_{\mathrm{clumps}}(t)^2}{b_{90}^2+r_s^2}\right)$
- Direct stellar heating rate: $H_{\star,\mathrm{direct}} = 2\sqrt{2\pi} \frac{G^2 \sigma_{\mathrm{clump}}^2}{(\sigma_\star^2+\sigma_{\mathrm{clump}}^2)^{3/2}} \rho_{\mathrm{clumps}}(t) \frac{M_{\mathrm{enc}}(b_{\min})^2}{M_{\mathrm{clump}}} \log\left(\frac{b_{90}^2+b_{\max}^2}{b_{90}^2+b_{\min}^2}\right)$
- Survival condition: $E_b \geq \Delta E_{\mathrm{tot}}$
- Power spectrum exclusion criterion: $f_{\mathrm{clump},P(k)}(M_{\mathrm{clump}}, z) > f_{\mathrm{clump},\lim}(M_{\mathrm{clump}}, \rho_s(z))$

**Quantitative Results**
Applying the method to the UFD Segue-I ($R_{0,\star}=24.7$ pc), the authors constrain DM clumps across $10 - 10^8 M_\odot$. The resulting bounds on the primordial power spectrum at $k \sim 10 - 1000 \text{ Mpc}^{-1}$ exceed FIRAS $\mu$-distortion limits by orders of magnitude and approach projected PIXIE sensitivity near $k \sim 100 \text{ Mpc}^{-1}$. Clump destruction thresholds are mass-dependent: HHH dominates disruption for $M_{\mathrm{clump}} \gtrsim 10^5 M_\odot$, while CCH governs $M_{\mathrm{clump}} \lesssim 10^5 M_\odot$. For dark matter isocurvature perturbations with a spectral index $n_{\mathrm{iso}}=4$, the amplitude is constrained to $A_{\mathrm{iso}} < 6 \times 10^{-20}$, representing the strongest known limit in this regime.

**Stated Limitations**
The analysis employs several conservative assumptions that may relax with future work. It assumes a monochromatic clump mass distribution and smooth DM dominance ($f_{\mathrm{clump}} \ll 1$). Finite-size scattering effects are treated conservatively by neglecting extra heating from stars passing through clump interiors. Survival calculations rely on specific adiabatic correction factors and ignore potential enhancements from complex orbital dynamics. Theoretical uncertainty persists regarding the choice of filter function (top-hat vs. Gaussian) and multiplicity function (Press-Schechter vs. Sheth-Tormen/Watson). Finally, current bounds do not yet reach the standard $\Lambda$CDM extrapolation from CMB scales; tighter constraints may require numerical simulations or the analysis of less dense host galaxies where clump survival is enhanced.
