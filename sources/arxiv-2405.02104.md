---
id: arxiv:2405.02104
type: paper
title: 'DeepSeek-RLHF: A Strong and Customizable LLM with Reinforcement Learning from
  Human Feedback'
url: https://arxiv.org/abs/2405.02104
retrieved: '2026-07-11'
maturity: comprehensive
topic: grpo
---

This paper proposes a formalism for new U(1) interactions, focusing on a U boson with axial couplings, and explores its phenomenology in muon beam dump experiments.

**Core Problem:**
The paper addresses the search for new light gauge bosons beyond the Standard Model, specifically those with axial couplings. These bosons, denoted as U, can arise from extra U(1) factors in the gauge group and are particularly interesting due to their potentially enhanced interactions in certain scenarios. The challenge is to identify experimental signatures and design experiments that can effectively probe the parameter space of such particles, especially considering their axion-like behavior in the ultrarelativistic limit.

**Method/Recipe Step-by-Step:**

1.  **Formalism Development:**
    *   Introduce new U(1) interactions involving weak hypercharge (Y), baryon (B), and lepton (L) numbers.
    *   Incorporate a possible axial symmetry generator $F_A$ by assuming two Brout-Englert-Higgs (BEH) doublets.
    *   Model the U boson as interpolating between a generalized dark photon, a dark Z, and an axially coupled gauge boson, depending on the specific U(1) quantum numbers and mixing with the Z boson.

2.  **Mass and Couplings Derivation:**
    *   Analyze mass mixing with the Z boson, which occurs after electroweak symmetry breaking and is independent of quark/lepton mass generation.
    *   Define the (2x2) mass-squared matrix $\mathcal{M}^2$ for the Z_sm and C (extra U(1) gauge field) basis.
    *   Derive the Z-U mixing angle $\xi$ and the U boson mass $m_U$ from this matrix.
    *   Express the U boson couplings to Standard Model fields as a combination of vector and axial components, emphasizing the role of $F_A$ and Z-U mixing.
    *   Specifically, derive axial couplings $g_{A\pm}$ for up-type/neutrinos and down-type/charged leptons, showing their dependence on $\gamma_A$, $\eta$, and $\cos\xi$.

3.  **Axion-like Behavior Analysis:**
    *   Demonstrate that the longitudinal polarization state of a light U boson with axial couplings exhibits enhanced interaction amplitudes, behaving effectively as a pseudoscalar particle.
    *   Quantify this enhancement with an effective pseudoscalar coupling $g_p = g_A \frac{2m_{q,l}}{m_U}$.
    *   Introduce an "invisibility parameter" $r$, which quantifies the reduction in interaction amplitudes when a dark singlet's vacuum expectation value (v.e.v.) contributes significantly to the U boson mass.

4.  **Lifetime and Decay Length Calculation:**
    *   Calculate the partial lifetime for a U boson decaying into $f\bar{f}$ pairs, including invisible decays into neutrino pairs.
    *   Approximate the decay length $l_U$ for an ultrarelativistic U boson, showing its dependence on energy, mass, and axial coupling.

5.  **Muon Beam Dump Experiment Simulation:**
    *   Focus on a purely axially coupled boson (where $F = F_A$, $\gamma_A = 1$, and no Z mixing).
    *   Model the production of U bosons via muon bremsstrahlung $\mu + N \rightarrow \mu + U + X$.
    *   Utilize the Fermi-Weiszäcker-Williams (FWW) approximation to relate the full scattering process to a 2 $\rightarrow$ 2 process $\mu + \gamma \rightarrow \mu + U$.
    *   Calculate the differential cross section $d\sigma/dx$ for U boson production.
    *   Estimate the expected number of events $N_{events}$ in a detector, considering production, survival probability through shielding, and decay within the decay region.
    *   Analyze the shape of the exclusion region in the $m_U$ vs. $\varepsilon_A$ parameter space, identifying boundaries related to minimum mass, very weak couplings (long decay length), and very strong couplings (short decay length).

6.  **Comparison with Other Models:**
    *   Compare the production cross sections and exclusion regions for axially coupled U bosons with those of pseudoscalar and vectorially coupled particles.
    *   Highlight the unique enhancement for axially coupled bosons in muon beam dump experiments due to the muon mass.

**Key Formulas (in LaTeX):**

*   **Mass-squared matrix for Z-U mixing:**

$$
\mathcal{M}^{2} = \frac{1}{4} \Bigg[\begin{array}{c c}{g_{Z}^{2}\left(v_{1}^{2}+v_{2}^{2}\right)}&{g^{\prime\prime}g_{Z}(F_{1}v_{1}^{2}-F_{2}v_{2}^{2})}\\ {\ g^{\prime\prime}g_{Z}(F_{1}v_{1}^{2}-F_{2}v_{2}^{2})}&{g^{\prime\prime2}(F_{1}^{2}v_{1}^{2}+F_{2}^{2}v_{2}^{2}+F_{\sigma}^{2}w^{2})}\\ \end{array}\Bigg]
$$

*   **Z-U mixing angle $\xi$:**

$$
\text{tan}\xi \simeq \frac{g^{\prime\prime}}{g_{Z}} \left(F_{2}\text{sin}^{2}\beta-F_{1}\text{cos}^{2}\beta\right)
$$

*   **U boson mass $m_U$:**

$$
m_{U}\simeq \frac{g^{\prime\prime}\text{cos}\xi}{2} \sqrt{\left(\frac{F_{1}+F_{2}}{2}\right)^{2}\text{sin}^{2}2\beta\ v^{2}+F_{\sigma}^{2}\:w^{2}}
$$

*   **Axial couplings $g_{A\pm}$ (Type II 2HDMs, $\gamma_A=1$):**

$$
g_{A\pm} \simeq \frac{g^{\prime\prime}}{4} \cos\xi (1 \pm \cos 2\beta)
$$

$$
g_{A\,d,e} \simeq \frac{g^{\prime\prime}}{2} \cos\xi \sin^{2}\!\beta
$$

*   **Effective pseudoscalar coupling:**

$$
g_{p} = g_{A} \frac{2m_{q,l}}{m_{U}}
$$

*   **Invisibility parameter $r$:**

$$
\frac{1}{r} = \frac{m_{U}}{m_{U}^{0}} = \frac{\sqrt{v^{2}\sin^{2}2\beta+F_{\sigma}^{2}\,w^{2}}}{v\,\sin2\beta} > 1
$$

*   **Axial coupling $\varepsilon_A$ in terms of $m_U$ and $r$ (for $F=F_A$, $\tan\beta=1$):**

$$
\varepsilon_{A} = \frac{g_{A}}{e} \simeq 6.7\times10^{-6}\ m_{U}(\mathrm{M e V})\ r
$$

*   **Decay length $l_U$:**

$$
l = \beta\gamma\,c\,\tau \simeq \frac{E(\mathrm{M e V})}{r^{2}\,m_{U}(\mathrm{M e V})^{4}}\,\times\,0.72\,\mathrm{m} \simeq \frac{E(\mathrm{M e V})}{m_{U}(\mathrm{M e V})^{2}\,(\varepsilon_{A}/10^{-6})^{2}}\times32.4\,\mathrm{m}
$$

*   **Differential production cross section $d\sigma/dx$ (axially coupled U boson):**

$$
\frac{d\sigma}{d x} = 2 \varepsilon_{A}^{2} \alpha^{3} x \chi \left[ \frac{m_{\mu}^{2} x (2 - x)^{2} - 2 (3 - 3 x + x^{2}) \tilde{u}_{max}}{3 x \tilde{u}_{max}^{2}} + \frac{2 m_{\mu}^{2} (1 - x)}{\tilde{u}_{max} \left(\tilde{u}_{max} + m_{\mu}^{2} x\right)} \right]
$$

    where $\tilde{u}_{max} = - m_U^2(1-x)/x - m_{\mu}^2 x$.
*   **Expected number of events $N_{events}$ (general form):**

$$
N_{\mathrm{events}} \simeq N_{\mu} \frac{\rho}{m_{T}} \left[ \int \frac{d\sigma}{dx} l_{U} \left(1 - e^{-L_{T}/l_{U}}\right) e^{-L_{\mathrm{sh}}/l_{U}} \times \left(1 - e^{-L_{\mathrm{dec}}/l_{U}}\right) dx \Bigg ] \left(B_{ee} + B_{\mu\mu}\right) P_{\mathrm{det}}
$$

**Key Quantitative Results and Numbers:**

*   **Muon enhancement:** The effective coupling of the U boson to muons is enhanced by a factor proportional to $2m_{\mu}/m_U$ compared to electrons. This factor is about 200 times larger for muons than for electrons.
*   **Experimental Setup (Illustrative Example):**
    *   Number of incoming muons ($N_{\mu}$): $10^{20}$
    *   Muon beam energy ($E_0$): 1.5 TeV
    *   Target length ($L_T$): 10 m (Lead target)
    *   Shielding extent ($L_{sh}$): 10 m
    *   Decay region length ($L_{dec}$): 100 m
*   **Probed Coupling Strength:** With the illustrative setup, coupling strengths as low as $10^{-7}$ can be probed across a mass range from 1.1 MeV to 4.7 GeV.
*   **Event Threshold:** Exclusion regions are typically set for $N_{events} = 3$ signal events.
*   **Axial Coupling Limit (very weakly coupled):** For very weakly coupled bosons, $\varepsilon_A < 10^{-7}$, largely independent of $m_U$.
*   **Decay Length Behavior:** $l_U \propto (m_U \varepsilon_A)^{-2}$.
*   **Branching Ratios (for $m_U$ slightly above 1 MeV but less than $2m_{\mu}$):** 60% into neutrino pairs, 40% into $e^+e^-$ pairs.
*   **Branching Ratios (for $m_U \gtrsim 1.5$ GeV):** Predominantly into hadrons (72%), with remaining decays into neutrino pairs (12%), $e^+e^-$ pairs (8%), and muon pairs (8%).
*   **Axial vs. Vector/Pseudoscalar:**
    *   For $m_U \lesssim 20$ MeV, an axially coupled U boson behaves like an axion-like pseudoscalar, with the same differential production cross section.
    *   For $m_U \gtrsim 100$ MeV, the exclusion region for an axially coupled U boson is similar to that of a vectorially coupled boson, as transverse polarizations dominate production.
    *   The lower mass boundary (A) for axially coupled bosons (due to the two BEH doublets) is not present for dark photons or vectorially coupled bosons.

**Stated Limitations:**

*   **Kinetic Mixing:** The analysis disregards a possible kinetic-mixing term between U(1) gauge fields, stating it's not present in an orthogonal field basis or easily removed by diagonalization.
*   **QCD Term:** The QCD term in the covariant derivative is ignored as it's not relevant to the U boson interactions discussed.
*   **Radiative Energy Losses:** For simplicity, negligible radiative energy losses of incoming muons within the target are assumed when calculating the total number of U bosons produced.
*   **FWW Approximation:** The Fermi-Weiszäcker-Williams (FWW) approximation is applicable when beam particles and emitted particles are highly relativistic and collinear.
*   **Simplified Comparison:** The comparison between pseudoscalar, axially, and vectorially coupled particles assumes the same decay lengths for all three, which is stated to be a "simplified illustration for comparative purposes... and does not represent a physically realistic situation for the pseudoscalar and vectorially coupled particles."
*   **Specific 2HDM Types:** While the analysis is general, specific examples like Type II 2HDMs are used for clarity, and the applicability to other types (e.g., Type I) is discussed with specific modifications.
*   **Anomalies:** The spontaneous breaking of the axial U(1) symmetry assumes anomalies are cancelled or irrelevant due to a very small gauge coupling.
