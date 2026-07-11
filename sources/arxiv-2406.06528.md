---
id: arxiv:2406.06528
type: paper
title: 'AlpacaEval 2.0: A Generalized Automatic Evaluation for Instruction-tuned LLMs'
url: https://arxiv.org/abs/2406.06528
retrieved: '2026-07-11'
maturity: comprehensive
topic: length-and-format-bias
---

# Phase Estimation via Number-Conserving Operation inside the SU(1,1) Interferometer

### Core Problem
While $\text{SU}(1,1)$ interferometers—which replace linear beam splitters with optical parametric amplifiers (OPAs)—offer superior phase sensitivity compared to passive interferometers, their precision is significantly degraded by photon losses, particularly internal losses. The core challenge is to mitigate the impact of this internal dissipation to maintain high phase estimation accuracy.

### Method and Recipe
The authors propose implementing a non-Gaussian Number-Conserving Operation (NCO) inside the $\text{SU}(1,1)$ interferometer. The theoretical scheme is structured as follows:

1.  **Input State Preparation**: A coherent state $|\alpha\rangle_a$ and a vacuum state $|0\rangle_b$ are used as inputs.
2.  **First Nonlinear Transformation**: The states pass through the first OPA, characterized by the two-mode squeezing operator:

$$
U_{S_1}(\xi_1) = \exp(\xi_1^* ab - \xi_1 a^\dagger b^\dagger)
$$

3.  **Internal Loss Simulation**: Internal photon losses are modeled using fictitious beam splitters (BS) with transmissivity $T_k = \cos^2 \theta_k \in [0, 1]$.
4.  **NCO Implementation**: An NCO operator $U_P = s aa^\dagger + t a^\dagger a$ (where $s^2 + t^2 = 1$) is applied to the output of the first OPA. Two boundary cases are analyzed:
    *   **PA-then-PS**: $U_{P_1} = aa^\dagger$ (Photon Addition then Photon Subtraction).
    *   **PS-then-PA**: $U_{P_2} = a^\dagger a$ (Photon Subtraction then Photon Addition).
5.  **Phase Shifting**: Mode $a$ undergoes a phase shift $U_\phi = \exp[i\phi(a^\dagger a)]$.
6.  **Second Nonlinear Transformation**: The beams are coupled in a second OPA $U_{S_2}$.
7.  **Detection**: Homodyne detection is performed on mode $a$ to measure the quadrature $X = (a + a^\dagger)/\sqrt{2}$.

### Key Formulas
The **phase sensitivity** $\Delta\phi$ is determined via the error propagation equation:

$$
\Delta\phi = \frac{\sqrt{\langle X^2 \rangle - \langle X \rangle^2}}{|\partial\langle X\rangle/\partial\phi|}
$$

The **Quantum Fisher Information (QFI)** for a pure state is given by:

$$
F_j = 4[\langle\Psi_j'|\Psi_j'\rangle - |\langle\Psi_j'|\Psi_j\rangle|^2]
$$

where $|\Psi_j'\rangle = \partial|\Psi_j\rangle/\partial\phi$. This relates to the **Quantum Cramér-Rao Bound (QCRB)**:

$$
\Delta\phi_{QCRB} = \frac{1}{\sqrt{v F}}
$$

(where $v$ is the number of measurements).

The performance is compared against the **Standard Quantum Limit (SQL)** $\Delta\phi_{SQL} = 1/\sqrt{N_j}$ and the **Heisenberg Limit (HL)** $\Delta\phi_{HL} = 1/N_j$, where $N_j$ is the total mean photon number inside the interferometer before the second OPA.

### Key Quantitative Results
*   **Phase Sensitivity**: Both NCO schemes enhance $\Delta\phi$ compared to the standard $\text{SU}(1,1)$ interferometer. The **PS-then-PA** scheme consistently demonstrates superior phase sensitivity over PA-then-PS in both ideal ($T_k=1$) and photon-loss scenarios.
*   **Robustness**: NCO schemes are less affected by photon losses than the standard interferometer. Specifically, NCO schemes can surpass the SQL even under significant internal losses (e.g., $T=0.7$).
*   **QFI and QCRB**: 
    *   In the **ideal case**, the PA-then-PS scheme slightly outperforms PS-then-PA in terms of QFI.
    *   In **high-loss scenarios**, the PS-then-PA scheme demonstrates a greater advantage in QFI and QCRB.
*   **Non-classicality**: Using the negative volume $V_j$ of the Wigner Function (WF) for $\alpha=1$ and $g \in [0.6, 1.2]$, the PS-then-PA scheme showed higher non-classicality ($V_j$ ranging from $0.034$ to $0.030$) compared to PA-then-PS ($V_j$ ranging from $0.009$ to $0.020$).

### Stated Limitations
The authors note that the study focuses primarily on the **ideal case of photon subtraction (PS) and photon addition (PA)**. They acknowledge that in practical experimental implementations, different parameters used to realize these non-Gaussian operations will impact the actual performance, which requires further examination.
