---
id: arxiv:2402.03000
type: paper
title: 'RLOO: Reinforcement Learning with Off-Policy Correction'
url: https://arxiv.org/abs/2402.03000
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

**Core Problem**
The development of ultra-wide bandgap (UWBG) optoelectronic devices requires zinc gallium oxide (ZnGa₂O₄) thin films with precise stoichiometric control, high crystallinity, and minimal defect density. Conventional radio-frequency (RF) magnetron sputtering from ceramic ZnO:Ga₂O₃ targets is limited by low deposition rates, poor power efficiency, high costs, and significant stoichiometry drift across varying substrate temperatures. This drift stems from differential elemental sticking probabilities and the high vapor pressure of zinc, which causes preferential desorption before oxidation, making reproducible film growth technically challenging.

**Method/Recipe Step by Step**
1. **Target Preparation:** Utilize a solid zinc target (99.95% purity) and a liquid gallium target (99.999% purity) in a dual-magnetron co-sputtering chamber.
2. **Chamber Conditioning:** Evacuate to a base pressure ≤7×10⁻⁴ Pa. Introduce argon (30 sccm) and oxygen (11–15 sccm) to establish a process pressure of 0.47 Pa.
3. **Power Configuration:** Fix the gallium sputtering power at 100 W. Adjust the zinc sputtering power ($P_{Zn}$) to set the target plasma optical emission spectroscopy (OES) intensity ratio $I_{Ga417}/I_{Zn481}$.
4. **Deposition Control:** Fine-tune $P_{Zn}$ in real-time during deposition to maintain a constant $I_{Ga417}/I_{Zn481}$ ratio, which linearly correlates to the Ga:Zn atomic flux. Modulate oxygen flow to suppress zinc volatilization at elevated temperatures.
5. **Substrate Temperature:** Vary substrate temperature from room temperature (RT) to 800°C. For post-annealing studies, deposit at RT then heat at 700°C in air for 3 hours.
6. **Characterization:** Analyze composition via XPS, structure via XRD, optical constants via spectroscopic ellipsometry, and emission properties via room-temperature photoluminescence (PL).

**Key Formulas**
Optical bandgap determination relies on the Tauc relation for direct allowed transitions:
$$(\alpha h\nu)^2 \propto (h\nu - E_g)$$
where $h\nu$ is the photon energy and $E_g$ is the optical bandgap. The absorption coefficient $\alpha$ is derived from the extinction coefficient $k$ and wavelength $\lambda$:
$$\alpha = \frac{4\pi k}{\lambda}$$
Nanocrystallite sizes are estimated using the Scherrer equation applied to XRD peak broadening.

**Key Quantitative Results**
The co-sputtering protocol enables precise tuning of the Ga:Zn atomic ratio from 0.3 to 5.7. The static deposition rate ranges from 5 to 14 nm/min, exceeding the ~0.4 nm/min rate of RF-sputtered films by at least an order of magnitude. Films deposited at RT are X-ray amorphous, while well-defined cubic spinel ZnGa₂O₄ Bragg peaks emerge at 300°C and intensify up to 800°C. The optical bandgap increases from 4.3 eV to 5.2 eV with higher Ga:Zn ratios and deposition temperatures; specifically, it widens from 4.6 eV to 5.1 eV as temperature rises from RT to 800°C for stoichiometric films. The refractive index at 2.25 eV decreases with increasing Ga:Zn ratio but jumps from ~1.86 to 1.97 between 100°C and 200°C, coinciding with crystallization onset. Photoluminescence spectra consistently exhibit a dominant peak at 3.1 eV (400 nm), attributed to oxygen-vacancy transitions, and a secondary peak at ~2.9 eV (428 nm) linked to self-activated octahedral Ga-O centers. Non-stoichiometric or improperly temperature-controlled films display a broad low-energy emission tail. XPS analysis reveals a consistent 2–5 at.% oxygen deficiency in near-stoichiometric films.

**Stated Limitations**
Stoichiometry control is constrained by the non-linear temperature dependence of atomic sticking coefficients, particularly the sharp desorption of zinc above 400°C. Compensating for this via $P_{Zn}$ adjustment alone is impractical at 700–800°C, as required power densities exceed the safe operational limit of ~0.05 kW/cm², risking target damage. Consequently, oxygen flow modulation is necessary but introduces additional process complexity. Furthermore, post-annealing at 700°C induces surface cracking and nanoscale particle formation, limiting its utility for defect-free applications. The inherent 2–5 at.% oxygen deficiency and potential for phase separation during prolonged thermal treatment also restrict the achievement of perfect stoichiometry and crystalline perfection.
