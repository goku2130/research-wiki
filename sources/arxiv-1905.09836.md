---
id: arxiv:1905.09836
type: paper
title: Learning to Use Tools via Reinforcement Learning
url: https://arxiv.org/abs/1905.09836
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

**Core Problem**
The study investigates how non-minimal couplings of light (pseudo-)scalar and vector fields to the Ricci scalar $R$ alter their cosmological evolution and dark matter production. It specifically quantifies the impact of curvature couplings on three non-thermal generation mechanisms: the misalignment mechanism, stochastic accumulation of superhorizon fluctuations, and direct energy density from inflationary fluctuations, while evaluating resulting observational constraints such as CMB isocurvature bounds and trans-Planckian field excursions.

**Methodology**
The analysis extends the Einstein-Hilbert action with curvature couplings $\xi \phi^2 R$ for scalars and $\frac{\kappa}{6} X_\mu X^\mu R$ for vectors. During inflation (modeled as de Sitter expansion with constant $H_I$), equations of motion are derived for homogeneous fields and quantum fluctuations. The superhorizon evolution is solved analytically using scale-dependent exponents that dictate field dilution or enhancement. Post-inflationary dynamics are matched to standard oscillation regimes once $R \to 0$. For vectors, transverse and longitudinal polarizations are decoupled; mode equations are solved in subhorizon and superhorizon limits, with intermediate regimes connected via numerical matching. Isocurvature power spectra are evaluated at CMB scales and compared against Planck constraints. The stochastic scenario is treated as a random walk process reaching equilibrium, requiring integration over superhorizon modes and tracking the competition between accumulated variance and exponentially decaying initial conditions.

**Key Formulas**
The scalar action and homogeneous equation of motion are:
$$S = \int \mathrm{d}^4x \sqrt{-g} \left( \frac{1}{2} \left( M_{\mathrm{pl}}^2 - \xi \phi^2 \right) R - \frac{1}{2} \partial_\mu \phi \partial^\mu \phi - \frac{1}{2} m_\phi^2 \phi^2 \right), \quad \ddot {\phi} + 3 H \dot {\phi} + \left(m _ {\phi} ^ {2} + \xi R\right) \phi = 0.$$
Evolution exponents are $\alpha_\pm = 3 \pm \sqrt{9 - 48\xi}$ (scalars) and $\beta_\pm \equiv 3 \pm \sqrt{1+8\kappa}$ (vectors). The relic density relative to observed dark matter is:
$$\frac {\Omega_ {\phi}}{\Omega_ {\mathrm{DM}}} \simeq 5 \mathcal {F} (T _ {\star}) \left(\frac {\phi_ {e}}{1 0 ^ {1 2} \mathrm{GeV}} \right) ^ {2} \sqrt {\frac {m _ {\phi}}{\mathrm{eV}}}.$$
Field suppression during inflation follows $\phi_e \simeq \phi_s \mathrm{e}^{-\frac{1}{2}\alpha_- N_{\mathrm{tot}}}$, while the fluctuation power spectrum evolves as $\mathcal {P} _ {\phi} (k, a) = \mathcal {P} _ {\phi} (k, a _ {k}) \left(\frac {k}{a H _ {I}} \right) ^ {\alpha_ {-}}$. The CMB isocurvature power spectrum is:
$$\mathcal {P} _ {\delta_ {\phi}} \left(k _ {\mathrm{CMB}}\right) = \frac {4}{\phi_ {c} ^ {2} \mathrm{e} ^ {\alpha_ {-} N _ {\mathrm{CMB}}}} \left(\frac {H _ {I}}{2 \pi}\right) ^ {2} F (\alpha_ {-}), \quad F(\alpha_-) \equiv \frac {2 ^ {2 - \alpha_ {-}}}{\pi} \Gamma^ {2} \left(\frac {3 - \alpha_ {-}}{2}\right).$$
For vector longitudinal modes, the numerical matching coefficient is fitted as $| f (\kappa) | = 0. 5 0 2 \kappa^ {- 0. 5} - 0. 2 2 4 + 0. 2 6 2 \kappa - 0. 0 4 1 1 \kappa^ {2} + 0. 0 0 6 5 4 \kappa^ {3}$.

**Key Quantitative Results**
Positive scalar couplings $\xi \lesssim \mathcal{O}(0.1)$ significantly weaken isocurvature constraints, permitting viable misalignment production for inflation scales up to $H_I \sim 10^{13}$ GeV regardless of mass. Negative couplings $\xi \lesssim -4$ (or $\xi \lesssim -0.1$ for high-scale inflation) are excluded due to runaway potentials causing trans-Planckian field excursions. Fluctuation-driven production becomes dominant for $\xi \gtrsim 0.03$ and masses $m_\phi \gtrsim 1$ eV at high $H_I$. The stochastic scenario requires extreme inflation ($N_{\mathrm{tot}} \gtrsim 10^9$) and very small couplings $\xi \lesssim 10^{-10}$ to satisfy Planck isocurvature limits. For vectors, $\kappa=1$ reproduces minimal scalar behavior, while $\kappa \lesssim 1$ relaxes isocurvature bounds. Longitudinal vector modes are suppressed for $m_X \lesssim 10^{-4}$ eV (scaling with $H_I$), and minimally coupled vector fluctuation production requires $\kappa \lesssim 10^{-6}$ for $m_X \sim 10^{-6}$ eV. The minimum inflationary e-folds are bounded by $N_{\mathrm{min}} = 61.97 + \frac{1}{2}\ln\left(\frac{H_I}{6.6 \cdot 10^{13}\,\text{GeV}}\right)$.

**Stated Limitations**
The vector theory acknowledges potential unitarity violations and ghost-like kinetic term sign flips for specific momenta during high-curvature epochs, treating $\kappa$ phenomenologically without full stability analysis. Calculations assume instantaneous reheating and exact de Sitter expansion; slow-roll spectral tilts are noted to cause only minor quantitative shifts. Negative $\xi$ bounds rely on conservative assumptions of minimal inflation and de Sitter vacuum initial conditions. The stochastic scenario strictly assumes equilibrium is reached, and longitudinal mode analyses depend on numerical matching in intermediate regimes without closed-form solutions. Full quantum gravity UV completions and vacuum stability under non-minimal couplings remain unresolved.
