---
id: arxiv:2408.08415
type: paper
title: Reinforced Self-Training (RST) for Language Modeling
url: https://arxiv.org/abs/2408.08415
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

The study addresses the poorly understood non-equilibrium dynamics of a unitary Bose gas near a narrow Feshbach resonance, specifically utilizing $^{133}$Cs atoms with a resonance width of 8.3 mG. While rapid quenches to unitarity in broad-resonance systems exhibit prethermalization governed by the Fermi energy $E_F \propto n^{2/3}$, narrow-resonance dynamics are dominated by the many-body Feshbach coupling scale $\alpha\sqrt{n}$. The core challenge is to theoretically and experimentally characterize the creation and subsequent evolution of out-of-condensate atoms and molecules following a quench.

The methodological approach combines reanalysis of time-of-flight (TOF) experimental data with a novel beyond-Hartree-Fock-Bogoliubov (HFB) theoretical framework. Experimentally, the authors extract momentum-resolved distributions and effective temperatures for atomic and molecular species from prior measurements of a quenched $^{133}$Cs BEC. Theoretically, they define one-point correlations $\xi_\sigma(t)$ for condensate wavefunctions and two-point correlations $G_{\alpha\beta}(k)$ for quantum fluctuations using a 4-vector field operator $\hat{\Psi}(k)$. They derive closed equations of motion for these correlations by introducing a critical cross-correlation term $f_{12}$ between atoms and molecules, which is absent in standard HFB approaches. The dynamical matrix $L(k)$ governs the evolution of finite-momentum populations, and its eigenvalues determine the instability and growth rates of non-condensed particles.

The system is governed by the effective Hamiltonian:

$$
\hat{H} = \sum_{\sigma} \int d^3x \hat{\psi}_\sigma^\dagger \left(-\frac{\hbar^2\nabla^2}{2m_\sigma} + \nu_\sigma\right) \hat{\psi}_\sigma + \int d^3x \sum_\sigma \frac{g_\sigma}{2} \hat{\psi}_\sigma^\dagger \hat{\psi}_\sigma^\dagger \hat{\psi}_\sigma \hat{\psi}_\sigma - \left(\alpha \hat{\psi}_1^\dagger \hat{\psi}_1^\dagger \hat{\psi}_2 + h.c.\right),
$$

where $\sigma=1,2$ denotes atoms and molecules, $\nu$ is detuning, $g_\sigma$ are background interaction strengths, and $\alpha$ is the Feshbach coupling. The condensate dynamics follow:

$$
i \partial_t \xi_1 = 2g_1 n_1 \xi_1 + (g_1 \xi_1^2 + g_1 x_1 - 2\alpha \xi_2)\xi_1^* - 2\alpha f_{12}, \quad i \partial_t \xi_2 = (\nu + 2g_2 n_2)\xi_2 + (g_2 \xi_2^2 + g_2 x_2)\xi_2^* - \alpha(x_1 + \xi_1^2).
$$

Two-point correlations evolve via $i \partial_t G_{mn}(k) = \sum_\beta L_{m\beta}(k) G_{\beta n}(k) + L_{n\beta}(k) G_{m\beta}(k)$. The characteristic oscillation frequency emerges from the energy gap $\Delta\mu = 2\mu_1 - \mu_2$, with self-energy corrections $\mu_1 = -2\alpha \text{Re} f_{12}/\xi_1$ and $\mu_2 = -\alpha \text{Re} x_1/\xi_2$.

Quantitative results reveal three distinct evolutionary stages: (I) rapid conversion of the atomic BEC into a molecular BEC (~0.2 ms), (II) partial decay of the molecular condensate generating finite-momentum atom-molecule complexes, and (III) steady-state coherent oscillations persisting for ~2 ms without thermalization. All particle species oscillate at a universal frequency of ~2 kHz, dictated by $\alpha\sqrt{n}$. The decay of the initial condensate transfers interaction energy to excited states, yielding kinetic energy increases of ~10 nK for atoms and ~5 nK for molecules. Theoretical momentum distributions concentrate at low momenta ($k \lesssim k_F/2$), contrasting with the saturation behavior of broad resonances.

Limitations include the theoretical model's assumption of $T=0$, whereas experiments occur at finite temperatures, leading to discrepancies in absolute kinetic energy values and slightly reduced oscillation amplitudes in the data. The framework truncates three-point correlations and decomposes four-point correlations, an approximation justified only within the experimentally accessible window of <3 ms. Furthermore, experimental particle loss is acknowledged but not incorporated into the dynamical equations, though it is noted to be negligible over the observed timeframe. The universality described here is strictly applicable to narrow-resonance bosonic superfluids and does not extend to broad-resonance prethermalization paradigms.
