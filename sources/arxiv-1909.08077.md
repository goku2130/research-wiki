---
id: arxiv:1909.08077
type: paper
title: Learning to summarize with human feedback
url: https://arxiv.org/pdf/1909.08077
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-modeling
---

This paper investigates the influence of surface anchoring disorder on the thermal pseudo-Casimir interaction in smectic-A (sm-A) liquid crystalline films.

**Core Problem:**
The core problem is to understand how quenched and annealed surface anchoring disorder modifies the thermal pseudo-Casimir force between confining substrates in smectic-A liquid crystalline films. Specifically, the authors aim to quantify the effect of disorder on the magnitude and nature of this fluctuation-induced interaction.

**Method/Recipe Step-by-Step:**

1.  **Model Setup:** A smectic-A liquid crystal film is confined between two parallel plates at $y=0$ and $y=d$. The sm-A layers are oriented perpendicular to the substrates (bookshelf geometry).
    *   Strong anchoring is assumed at $y=0$, fixing the mean molecular orientation in the z-direction.
    *   Finite-strength anchoring with a random (disordered) component is applied at $y=d$.
    *   The layer displacements $u(\mathbf{r}_{\perp},y)$ in the bulk have a harmonic form.
    *   The surface interaction energy at $y=d$ is of the Rapini-Papoular form, dependent on anchoring strength $W$ and easy axes $\mathbf{e}(\mathbf{r}_{\perp})$.
    *   The disorder fields $e_i(x,z) = \{e_x(x,z), e_y(x,z)\}$ are assumed statistically independent with Gaussian distributions around zero mean values.

2.  **Thermodynamic Free Energy Calculation:**
    *   The thermodynamic free energy is obtained by averaging the sample free energy over all disorder field realizations.
    *   For **quenched disorder** ($F_Q$), the average is taken over the logarithm of the partition function: $F_Q = -k_B T \langle \ln \mathcal{Z} \rangle$.
    *   For **annealed disorder** ($F_A$), the average is taken over the partition function itself: $F_A = -k_B T \ln \langle \mathcal{Z} \rangle$.
    *   The **replica trick** is employed for quenched disorder, considering $k$ replicated fields $u_a$ ($a=1, \dots, k$) and taking the limit $k \to 0$. For annealed disorder, $k=1$.

3.  **Replicated Partition Function and Fourier Diagonalization:**
    *   The disorder-averaged replicated partition function $\langle \mathcal{Z}^k \rangle$ is calculated by performing Gaussian averages over the disorder fields.
    *   Translational invariance in the $xz$-plane allows for Fourier transformation of $u(\mathbf{r}_{\perp},y)$ into $u(\mathbf{p}_{\perp},y)$, where $\mathbf{p}_{\perp}=(p_x, p_z)$ is the in-plane wavevector.
    *   The replicated Hamiltonian $\mathcal{H}^{(k)}$ is diagonalized over the in-plane modes, separating into bulk $h_b^{(k)}$ and surface $h_s^{(k)}$ terms.
    *   Path-integral methods are used to integrate over bulk displacement fields, yielding the replicated partition function in terms of surface fields $u_a(\mathbf{p}_{\perp},d)$ and $\dot{u}_a(\mathbf{p}_{\perp},d)$ weighted by an effective surface Hamiltonian $h_s^{eff}$.

4.  **Disorder-Averaged Free Energy Expressions:**
    *   The determinant of a $2k \times 2k$ matrix $\mathbb{M}^{(k)}$ is computed, which contains the bulk and surface contributions, including disorder.
    *   The free energy for annealed disorder ($k=1$) is given by:
        $\mathcal{F}_A = \frac{k_B T}{2} \sum_{\mathbf{p}_{\perp}} \ln \left| \mathcal{B} \det \left( \mathbb{M} - \frac{\beta W c(\mathbf{p}_{\perp})}{2} \mathbb{P} \right) \right|$
    *   The free energy for quenched disorder ($k \to 0$) is given by:
        $\mathcal{F}_Q = \frac{k_B T}{2} \sum_{\mathbf{p}_{\perp}} \ln | \mathcal{B} \det \mathbb{M} | - \frac{W}{4} \sum_{\mathbf{p}_{\perp}} c(\mathbf{p}_{\perp}) \text{tr} \left( \mathbb{P} \mathbb{M}^{-1} \right)$
        This can be explicitly rewritten as:
        $\mathcal{F}_Q = \frac{k_B T}{2} \sum_{\mathbf{p}_{\perp}} \ln \left| \mathcal{B} (m_{11}m_{33}-m_{13}^2) \right| - \frac{W}{4\ell} \sum_{\mathbf{p}_{\perp}} c(\mathbf{p}_{\perp}) \frac{p_x^2 m_{33} + m_{11}}{m_{11}m_{33}-m_{13}^2}$

5.  **Force Calculation:** The fluctuation-induced force $f(d)$ is calculated by differentiating the free energy with respect to the inter-surface distance $d$:
    $f(d) = -\frac{\partial \mathcal{F}(d)}{\partial d}$.
    The rescaled interaction pressure $\Pi(\tilde{d})$ is defined as:
    $\Pi(\tilde{d}) = \frac{\beta \lambda^3}{A} f(\lambda \tilde{d})$, where $\tilde{d} = d/\lambda$.

**Key Formulas in LaTeX:**

*   **Bulk Hamiltonian (harmonic form):**
    $\mathcal{H}_{\mathrm{b}} = \frac{K}{2} \int_V \mathrm{d}\mathbf{r} [(\partial_x^2 u + \partial_y^2 u)^2 + \frac{B}{K}(\partial_z u)^2]$
*   **Surface Interaction Energy (Rapini-Papoular form):**
    $\mathcal{H}_{\mathrm{s}} = - \frac{W}{2} \int_{\partial V} \mathrm{d}\mathbf{r}_{\perp} (\mathbf{n} \cdot \mathbf{e})^2$
    In terms of $u$:
    $\mathcal{H}_{\mathrm{s}} = \frac{W}{2} \int_{\partial V} \mathrm{d}\mathbf{r}_{\perp} \left[ (\partial_x u)^2 + (\partial_y u)^2 + 2 (e_x \partial_x u + e_y \partial_y u) \right]$
*   **Gaussian Distribution of Disorder Fields:**
    $\mathcal{P}[e_i] = \mathcal{C} \exp\left( - \frac{1}{2} \int \mathrm{d}\mathbf{r}_{\perp} \mathrm{d}\mathbf{r}_{\perp}' e_i(\mathbf{r}_{\perp}) c^{-1}(\mathbf{r}_{\perp} - \mathbf{r}_{\perp}') e_i(\mathbf{r}_{\perp}') \right)$
*   **Free Energy for Annealed Disorder:**
    $\mathcal{F}_A = \frac{k_B T}{2} \sum_{\mathbf{p}_{\perp}} \ln \left| \mathcal{B} \det \left( \mathbb{M} - \frac{\beta W c(\mathbf{p}_{\perp})}{2} \mathbb{P} \right) \right|$
*   **Free Energy for Quenched Disorder:**
    $\mathcal{F}_Q = \frac{k_B T}{2} \sum_{\mathbf{p}_{\perp}} \ln \left| \mathcal{B} (m_{11}m_{33}-m_{13}^2) \right| - \frac{W}{4\ell} \sum_{\mathbf{p}_{\perp}} c(\mathbf{p}_{\perp}) \frac{p_x^2 m_{33} + m_{11}}{m_{11}m_{33}-m_{13}^2}$
*   **Force:**
    $f(d) = -\frac{\partial \mathcal{F}(d)}{\partial d}$
*   **Rescaled Pressure:**
    $\Pi(\tilde{d}) = \frac{\beta \lambda^3}{A} f(\lambda \tilde{d})$
*   **Correlation Function for Spatially Correlated Disorder:**
    $c(p_x, p_z) = \frac{c_0}{\xi_z^4(p_x^4 + \lambda^{-2}p_z^2) + 1}$

**Key Quantitative Results and Numbers:**

*   **Disorder-free case (Fig. 1):** For $\lambda/a_0 = 2$, the rescaled interaction pressure $\Pi(\tilde{d})$ is shown for various $\lambda/\ell$ values ($10^6, 100, 10, 5, 3, 0.5, 0.1$). As anchoring strength decreases (smaller $\lambda/\ell$), the attractive force diminishes, and repulsive interactions can emerge due to boundary condition asymmetry.
*   **Quenched disorder (Fig. 2):** For $\lambda/a_0 = 2$ and $\lambda/\ell = 10$, with uncorrelated quenched surface disorder ($c(\mathbf{p}_{\perp}) = c_0$), increasing the rescaled variance $\beta K c_0 / \ell$ from 0 to $0.2, 0.3, 0.4, 0.5, 0.7, 0.9$ (bottom to top) reduces the attractive pressure and can change it to repulsive.
*   **Correlated quenched disorder (Fig. 3):** For $\lambda/a_0 = 2$, $\lambda/\ell = 3$, and $\beta K c_0 / \ell = 0.3$, increasing the rescaled correlation length $\xi_z/\lambda$ from 0 (uncorrelated) to 0.5 and 1 (top to bottom) diminishes the role of disorder, making the interaction more attractive. The disorder-free case ($c_0=0$) shows the most attractive interaction.
*   **Annealed vs. Quenched Disorder (Fig. 4):** For $\lambda/a_0 = 2$, $\lambda/\ell = 10$, and $\beta K c_0 / \ell = 0.5$, with and without spatial correlations ($\xi_z/\lambda = 0.3$ and 0), annealed disorder leads to a more strongly attractive interaction profile than quenched disorder across all inter-substrate separations, unlike nematic systems where the difference is smaller.

**Stated Limitations:**

*   The formalism for quenched disorder is based on a fixed equilibrium configuration of the order parameter and does not account for possible destabilization of this uniform equilibrium configuration upon increased disorder.
*   For annealed disorder, the factor $(1 - \beta W c(\mathbf{p}_{\perp}))$ must be kept positive for all $\mathbf{p}_{\perp}$-modes to ensure the equilibrium order-parameter profile remains unchanged.
*   The study focuses on the transverse pseudo-Casimir force arising from thermal fluctuations of smectic-A layers in a bookshelf geometry.
*   The analysis assumes small displacements from thermodynamic equilibrium.
