---
id: arxiv:2406.01672
type: paper
title: 'Obol: A Comprehensive Framework for Detecting Data Contamination in LLMs'
url: https://arxiv.org/abs/2406.01672
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# Summary: Baryonification Extended to Thermal Sunyaev Zel’dovich

### Core Problem
Standard gravity-only $N$-body simulations fail to account for the impact of galaxy formation and feedback (baryonic processes) on the matter field, which introduces significant theoretical systematic errors in weak gravitational lensing (WL) and thermal Sunyaev-Zel’dovich (tSZ) effect observations. While hydrodynamical simulations provide these details, they are computationally expensive. The authors aim to extend the Baryon Correction Model (BCM), or "baryonification," to self-consistently model gas temperature and pressure, enabling fast and accurate predictions of tSZ, WL, and their cross-correlations down to small scales.

### Method/Recipe
The framework transforms gravity-only simulations into baryonified simulations through the following steps:

1.  **Particle Splitting:** To reduce discreteness noise, each particle in the gravity-only simulation is split into $N_c$ particles (representing dark matter, gas, and stars) at the same position, with masses summed to the original particle mass.
2.  **Gas Density Profiling:** An analytical gas density profile $\rho_{BG}$ is applied to haloes:

$$
\rho_{\mathrm{B G}}(r\leq r_{\mathrm{o u t}})=\rho_{0}\frac{1}{(1+r/r_{\mathrm{i n n}})^{\beta_{i}}}\frac{1}{(1+(r/r_{\mathrm{o u t}})^{2})^{2}}
$$

    where $r_{inn}$ and $r_{out}$ are defined by free parameters $\theta_{inn}$ and $\theta_{out}$ relative to $r_{200}$.
3.  **Thermodynamic Modeling:** The gas is assumed to follow a polytropic equation of state. The polytropic index $\Gamma$ is modeled as:

$$
\Gamma=1+\frac{(1+x^{\prime})\ln(1+x^{\prime})-x^{\prime}}{(1+3x^{\prime})\ln(1+x^{\prime})}
$$

    where $x' = c \times \theta_{out}$ ($c$ is halo concentration).
4.  **Non-Thermal Pressure Correction:** To account for turbulence and magnetic fields, the total pressure $P_{BG}$ is corrected to find the thermal pressure $P_{BG,th}$ using a fitting function $f_{hh}$ based on peak height $\nu$:

$$
P_{\mathrm{B G,t h}}(r)=f_{\mathrm{h h}}(r/r_{\mathrm{200m}})\,P_{\mathrm{B G}}(r)
$$

5.  **Temperature Assignment:** The temperature $T_{BG}$ is derived via the ideal gas law:

$$
T_{\mathrm{B G}}(\boldsymbol{r})=\frac{\mu m_{p}}{k_{\mathrm{B}}}\frac{P_{\mathrm{B G,t h}}(\boldsymbol{r})}{\rho_{\mathrm{B G}}(\boldsymbol{r})}
$$

    Particles outside haloes are assigned a constant field temperature $T_{field}$.
6.  **Projection to Observables:** 
    *   **tSZ:** Particles are assigned a "Compton weight" $\Upsilon_{i}={\frac{\sigma_{\mathrm{T}}k_{\mathrm{B}}T_{i}\,m_{i}}{\mu_{\mathrm{e}}m_{\mathrm{p}}m_{\mathrm{e}}c^{2}}}$ and integrated along the line of sight to create Compton-$y$ maps.
    *   **WL:** Convergence maps $\kappa(\theta)$ are created by integrating the excess surface mass density.

### Key Quantitative Results
The model was validated against BAHAMAS hydrodynamical simulations across different AGN feedback strengths ($T_{AGN} = 10^{7.6}, 10^{7.8}, 10^{8.0}$ K):

*   **Power Spectra:** The matter power spectrum suppression $S(k)$ was reproduced at the **percent level**. Electron pressure auto- and cross-power spectra were fitted with **better than 10% accuracy** down to $k = 5\,h\,\mathrm{Mpc}^{-1}$.
*   **Angular Power Spectra:** Down to $\ell = 5000$, the convergence power spectrum was reproduced within **1%** and the tSZ power spectrum within **10%**.
*   **Efficiency:** Generating a $10^\circ \times 10^\circ$ lightcone up to $z=1$ took approximately **2.5 hours using 32 cores**.

### Stated Limitations
*   **Cosmic Variance:** The BAHAMAS simulation box size ($400^{-1}\,h\,\mathrm{Mpc}$) introduces cosmic variance of approximately **10%** at $k \approx 0.1\,h\,\mathrm{Mpc}^{-1}$, which can be comparable to the AGN feedback calibration.
*   **Symmetry Assumptions:** The model assumes smooth, spherically symmetric radial temperature and pressure profiles, ignoring the effects of pressure clumping.
*   **Redshift Dependence:** The redshift scaling of the non-thermal pressure contribution ($\alpha_{nth} = 1.5$) was fixed based on inspection rather than being a fully free parameter.
*   **Computational Cost:** While faster than hydro-simulations, fitting large simulations directly remains computationally demanding.
