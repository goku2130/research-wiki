---
id: arxiv:2405.07823
type: paper
title: 'TensorRT-LLM: Fast and Easy Way to Build and Deploy LLMs'
url: https://arxiv.org/abs/2405.07823
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

**Core Problem**
Laser powder bed fusion (LPBF) suffers from sub-optimal mechanical properties due to defects like porosity and surface roughness, primarily driven by spatter ejection during laser-material interactions. The multi-physics phenomena governing melt pool instabilities, vapor recoil, and fluid ejection are highly non-linear and occur at microsecond/micrometer scales. High-fidelity numerical simulations capture these dynamics but are computationally prohibitive for large-scale parameter screening, while experimental monitoring lacks the resolution to quantify spatter formation mechanisms or establish reliable process windows.

**Methodology**
The authors constructed a hybrid physics-ML pipeline:
1. **High-Fidelity CFD Setup:** A volume-of-fluid (VOF) model was implemented in OpenFOAM using the `icoReactingMultiphaseInterFoam` solver. Tangential surface tension (Brackbill equation) and recoil pressure (Clausius–Clapeyron) were integrated to resolve keyhole dynamics and spatter ejection.
2. **Validation:** Simulations were calibrated against experimental thermal imaging and compared with FLOW-3D outputs for melt pool dimensions and temperature profiles.
3. **Spatter Tracking & Feature Extraction:** A custom algorithm thresholded the gas volume fraction ($\alpha_g > 0.5$) and applied connected component analysis to isolate spatter droplets. Particle trajectories were tracked across timesteps using velocity-based displacement prediction and $k$-d tree nearest-neighbor assignment. Features (position, velocity, temperature, density, pressure) were sampled from both spatter and melt pool regions.
4. **Machine Learning Classification:** A dataset of 488 samples (50% spatter, 50% melt pool) was split 70/30 for training/testing. Seven classifiers (RF, GB, Bagging, ExtraTrees, LGBM, KNN, SVC) were optimized via GridSearchCV (5-fold CV, ROC-AUC). SHAP values and Partial Dependence Plots (PDPs) quantified feature importance and prediction thresholds.
5. **Process Map Generation:** The OpenFOAM-trained ML model was applied to 281 FLOW-3D simulations (which lack explicit spatter physics) to predict spatter initiation voxels, generating a power-velocity process map correlated with porosity boundaries.

**Key Formulas**
The multiphysics model and tracking algorithm rely on the following governing equations:
- Tangential surface tension force: $F_s = [\sigma \kappa \vec{n} + (\nabla \sigma - \vec{n}(\vec{n} \cdot \nabla \sigma))] \cdot \frac{2 \rho(x)}{\rho_1 + \rho_2} \cdot \frac{|\nabla \rho(x)|}{\rho_2 - \rho_1}$
- VOF advection with compression velocity: $\frac{\partial \alpha_1}{\partial t} = -\vec{\nabla} \cdot (\alpha_1 \vec{u}) - \vec{\nabla} \cdot \vec{u} \alpha_1 (1 - \alpha_1)$
- Recoil pressure boundary condition: $\vec{P}_r = 0.54 P_0 \exp\left[\frac{L_v M (T - T_v)}{R T T_v}\right] \vec{n} |\nabla \alpha_1| \frac{2\rho}{\rho_{\text{metal}} + \rho_{\text{gas}}}$
- Particle displacement tracking: $\delta\mathbf{x}_i = \frac{\Delta t}{N_s} \sum_{s=1}^{N_s} \mathbf{v}_s$
- SHAP feature attribution: $\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$

**Quantitative Results**
Validation showed melt pool dimensions closely matched experimental and FLOW-3D data, with a maximum 7.8% depth deviation at 450 W; all other deviations remained below 4%. The ML classifiers achieved 98–100% training and 97–99% testing accuracy across all metrics. Removing spatial coordinates reduced accuracy to 95–97% (KNN: 90%), yet retained sufficient fidelity for cross-software transfer. Computationally, a 5 μm mesh simulation required 72 hours in OpenFOAM versus 4 hours in FLOW-3D. The resulting process map revealed spatter volume scales as: high power/low speed > high power/high speed > low power/low speed > low power/high speed. A distinct low-spatter regime emerged at scan speeds of 500–700 mm/s across tested power levels.

**Stated Limitations**
The OpenFOAM implementation initially omitted Marangoni convection, requiring external modification. FLOW-3D relies on simplified vapor dynamics and lacks tangential surface tension, preventing direct spatter simulation. Experimental validation was constrained by plume interference and camera exposure limits, obscuring temperatures below 2000°C. Furthermore, the highest classification accuracy depended on spatial features, which are not directly accessible in standard commercial AM simulations, necessitating feature engineering for broader industrial deployment. High computational costs of the high-fidelity solver also restrict direct large-scale parameter screening without ML surrogates.
