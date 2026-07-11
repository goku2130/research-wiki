---
id: arxiv:2406.13479
type: paper
title: Reasoning with Reinforcement Learning
url: https://arxiv.org/abs/2406.13479
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

The core problem addressed in this work is the lensing amplitude anomaly observed in the cosmic microwave background (CMB) by the Planck satellite, where the measured smoothing of acoustic peaks exceeds theoretical predictions derived from the reconstructed lensing potential ($A_L > 1$). The author investigates whether this anomaly, alongside other cosmological tensions such as the Hubble tension, $\Omega_K \neq 0$, and $S_8$ tension, could serve as indirect evidence for extremely low frequency primordial gravitational waves (PGWs) with frequencies $f < 10^{-18}$ Hz. Traditional B-mode polarization detection methods are hindered by galactic dust foregrounds, motivating the exploration of alternative detection channels.

The proposed mechanism relies on the physical coupling between extremely low frequency PGWs and large-scale structure (LSS). As CMB photons propagate from the last scattering surface to the observer, their trajectories are deflected by LSS through weak gravitational lensing, which remaps temperature anisotropies and smooths the acoustic power spectrum. PGWs with wavelengths exceeding the cosmic horizon exhibit effectively frozen amplitudes. The interaction between these frozen PGWs and LSS is hypothesized to enhance the weak lensing effect beyond a simple linear superposition, thereby producing greater spectral smoothing that could resolve the $A_L > 1$ discrepancy. The verification follows a systematic computational recipe: first, modified versions of the CAMB and CLASS Boltzmann codes are employed to incorporate the PGW-LSS coupling; second, CMB temperature maps are numerically generated using $\Lambda$CDM density perturbations and tensor perturbations parameterized by the tensor-to-scalar ratio $r$; third, the resulting CMB temperature power spectrum is calculated; fourth, the process is inverted to infer the value of $r$ required to match the observed spectrum; fifth, if the inferred $r$ falls below the observationally constrained value, the lensing potential is reconstructed from the generated maps to derive the theoretical smoothing amount; and finally, this theoretical smoothing is compared against the smoothing derived from Planck’s reconstructed lensing potential. Consistency between the two would confirm that extremely low frequency PGWs account for the anomaly.

The theoretical framework is built upon a perturbed spacetime metric and the Boltzmann transport equation for photons. The PGW metric perturbation is expressed as:

$$
h_{ij} = \frac{a_0}{a} \left[ (u_i u_j - v_i v_j) h_+ + (u_i v_j + v_i u_j) h_\times \right] \cos(\omega \eta - \mathbf{k} \cdot \mathbf{x})
$$

where $a$ is the scale factor ($a_0=1$), $\eta$ is conformal time, $\mathbf{k}$ is the propagation vector, and $h_+, h_\times$ are polarization amplitudes. The full spacetime line element incorporates the scalar potential $U$ and tensor perturbations $h_{ij}$:

$$
ds^2 = a^2(\eta) \left[ -(1 + 2U)d\eta^2 + (1 - 2U)(dx^2 + dy^2 + dz^2) + h_{ij}dx^i dx^j \right]
$$

Photon deflection is governed by the geodesic equation $\frac{d^2 x^\mu}{d\lambda^2} + \Gamma_{\alpha\beta}^\mu \frac{dx^\alpha}{d\lambda}\frac{dx^\beta}{d\lambda} = 0$, with Christoffel symbols $\Gamma_{\alpha\beta}^\mu = \frac{1}{2} g^{\mu\rho}(\partial_\alpha g_{\rho\beta} + \partial_\beta g_{\rho\alpha} - \partial_\rho g_{\alpha\beta})$. The evolution of the photon distribution function $f$ follows the Boltzmann equation:

$$
\frac{\partial f}{\partial t} + \frac{\partial f}{\partial x^i} \frac{dx^i}{dt} + \frac{\partial f}{\partial p} \frac{dp}{dt} + \frac{\partial f}{\partial n^i} \frac{dn^i}{dt} = C(f)
$$

where the directional derivative term $\frac{\partial f}{\partial n^i}\frac{dn^i}{dt}$ encapsulates weak lensing effects.

Quantitative benchmarks for photon deflection are computed across specific frequency and tensor-to-scalar ratio values. For $r = 0.01$, the root-mean-square deflection angles are $6.93423 \times 10^{-7}$, $3.38361 \times 10^{-6}$, and $6.15404 \times 10^{-6}$ at frequencies $f = 10^{-19}$, $5 \times 10^{-19}$, and $10^{-18}$ Hz, respectively. Scaling to $r = 0.005$ yields deflections of $4.90326 \times 10^{-7}$, $2.39258 \times 10^{-6}$, and $4.35158 \times 10^{-6}$, while $r = 0.001$ produces $2.19273 \times 10^{-7}$, $1.06996 \times 10^{-6}$, and $1.946023 \times 10^{-6}$. Although standard LSS weak lensing induces deflections of approximately 2 arcminutes, the coupling mechanism is proposed to amplify the total deflection beyond linear addition, potentially explaining the excess smoothing.

The study explicitly acknowledges several limitations. The current analysis remains strictly qualitative, and the proposed enhancement of weak lensing requires rigorous quantitative validation using modified Boltzmann solvers to accurately model the PGW-LSS coupling. Furthermore, the theoretical treatment simplifies complex lensing geometries (e.g., modeling a single cluster for illustrative purposes), and a comprehensive numerical verification addressing the lensing amplitude anomaly alongside Hubble, $\Omega_K$, and $S_8$ tensions is deferred to a forthcoming companion publication.
