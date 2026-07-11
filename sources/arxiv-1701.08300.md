---
id: arxiv:1701.08300
type: paper
title: Collapse. What else?
url: https://arxiv.org/abs/1701.08300
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Summary: Collapse. What else?

### Core Problem
The author presents the quantum measurement problem not as a matter of interpretation, but as a serious physical problem indicating that quantum theory is incomplete. Specifically, the theory fails to describe how measurements are performed in principle. The author argues that popular alternatives—such as Many-Worlds and Bohmian mechanics—do not actually solve the problem but instead rely on "effective collapses" triggered by the observer ("I") and assume a "hyper-determinism" based on the physical reality of infinitesimal digits in mathematical real numbers, which the author rejects as non-physical.

### Proposed Method: Modified Schrödinger Equation
To resolve the measurement problem, the author proposes modifying the dynamics of quantum theory by replacing the linear, deterministic Schrödinger equation with a non-linear, non-deterministic stochastic differential equation. This approach describes a form of Brownian motion in Hilbert space.

**Step-by-Step Recipe:**
1. **Introduce Stochasticity:** Generalize the Schrödinger equation into an Itô stochastic differential equation to avoid the superluminal communication associated with deterministic non-linear generalizations.
2. **Define Operators:** Introduce a set of linear operators $L_j$ (e.g., proportional to the positions of all elementary particles).
3. **Evolve the State Vector:** The state vector $|\psi_t\rangle$ evolves according to the stochastic equation, where the non-linear terms drive the system toward an eigenstate of the $L_j$ operators.
4. **Determine Outcome Probabilities:** The probability that the system converges to a specific eigenstate $|l\rangle$ is given by the standard quantum probability $|\langle l|\psi_0\rangle|^2$.
5. **Average for Ensemble Dynamics:** Averaging over all Wiener processes yields a closed-form equation for the density matrix $\rho(t)$, ensuring no faster-than-light communication.

### Key Formulas
The fundamental dynamical evolution is described by the following Itô equation:

$$
\begin{aligned}
|d\psi_t\rangle = &-iH|\psi_t\rangle dt \\
&+ \sum_j \left(2\langle L_j^\dagger \rangle_{\psi_t} L_j - L_j^\dagger L_j - \langle L_j^\dagger \rangle_{\psi_t} \langle L_j \rangle_{\psi_t}\right) |\psi_t\rangle dt \\
&+ \sum_j \left(L_j - \langle L_j \rangle_{\psi_t}\right) |\psi_t\rangle d\xi_j
\end{aligned}
$$

Where the expectation value is defined as:

$$
\langle L_j \rangle_{\psi_t} = \frac{\langle \psi_t | L_j | \psi_t \rangle}{\langle \psi_t | \psi_t \rangle}
$$

The independent complex Wiener processes $d\xi_j$ satisfy:

$$
\mathcal{M}[d\xi_j d\xi_k^*] = \delta_{jk} dt
$$

The resulting evolution of the density matrix $\rho(t)$ follows:

$$
\frac{d\rho(t)}{dt} = -i[H, \rho(t)] - \sum_j \left(L_j^\dagger L_j \rho(t) + \rho(t) L_j^\dagger L_j - 2 L_j \rho(t) L_j^\dagger\right)
$$

### Key Quantitative Results
The author illustrates the efficacy of this model by applying it to a macroscopic measurement apparatus (a pointer). 
* **Scale of System:** A pointer is assumed to consist of approximately $10^{20}$ particles.
* **Collapse Rate:** Because the stochastic non-linear terms act on all particles, the modified equation predicts that a macroscopic pointer will localize approximately $10^{20}$ times faster than an individual particle. This results in a "quasi-instantaneous collapse" for macroscopic objects while leaving microscopic systems essentially unaffected.

### Stated Limitations
The author identifies two primary challenges to this theory:
1. **Environmental Mimicry:** The same stochastic evolution can be derived within standard quantum theory by assuming a coupling between a system and its environment and conditioning the state on continuous measurement outcomes. This makes it difficult to prove that the evolution is a fundamental law rather than a result of environmental decoherence.
2. **Error Correction:** The author notes that advanced quantum information processors utilizing sufficient error correction could potentially hide these stochastic terms, preventing the experimental detection of the modified dynamics.
