---
id: arxiv:2401.06518
type: paper
title: 'Contrastive Preference Optimization: Pushing the Boundaries of LLM Alignment
  in Large-Scale Settings'
url: https://arxiv.org/abs/2401.06518
retrieved: '2026-07-12'
maturity: comprehensive
topic: dpo-variants
---

# Transitional Grid Maps (TGM) for Joint Static and Dynamic Occupancy

### Core Problem
Autonomous agents require accurate environment representations for planning and localization. Standard Occupancy Grid Maps (OGMs) assume a static environment; when dynamic obstacles are present, OGMs either retain "traces" of moved objects (ghosting) or, if confidence is capped (clamped-OGM), lose stability and fail to distinguish between static and dynamic elements. While decoupling these components via separate trackers and maps is common, it typically requires reliable semantic segmentation, which is often unavailable for noisy or sparse sensors (e.g., radar, low-resolution lidar). The core challenge is to jointly infer static and dynamic occupancy without relying on high-level semantic labels.

### Method
The authors propose **Transitional Grid Maps (TGMs)**, a probabilistic framework that models the environment as a set of $N$ cells, where each cell $m_{t,i}$ can exist in one of three states: free ($f$), static ($s$), or dynamic ($d$).

#### Step-by-Step Recipe:
1.  **Bayesian Formulation**: The map is split into a constant static map $m^s$ and a stochastically evolving dynamic map $m_t^d$. The posterior of the map is approximated as the product of its marginals:

$$
p(m_{t}|z_{1:t},x_{1:t})\approx{\prod_{i}}p(m_{t,i}|z_{1:t},x_{1:t})
$$

2.  **Inverse Sensor Model**: New measurements $z_t$ are integrated. The probability of occupancy is split between static and dynamic states proportionally to their priors.
3.  **Random Transition Modeling**: To avoid the NP-hard complexity of general Bayesian network inference, the authors introduce a transition model $\mathcal{T}_{j,i}$ (the event that content moves from cell $j$ to $i$). Transitions are constrained such that:
    *   Transitions between different cells are impossible if either cell is static.
    *   The probability of staying in the same cell is $1$ if the cell is static.
4.  **Analytical Prediction via Convolution**: To avoid marginalizing over $3^N$ possible maps, the dynamic prediction is simplified using 2D convolutions. If transition probabilities depend only on relative position ($T[\Delta x, \Delta y] = \tau_{j,i}$), the prediction for dynamic cells is:

$$
p(m_{t,i}^d | z_{1:t}, x_{1:t}) = M^d[x_i, y_i](\tau_0 + (M^s * K')[x_i, y_i]) + (1 - M^s[x_i, y_i])(M^d * K)[x_i, y_i]
$$

    where $M^s$ and $M^d$ are the estimated static and dynamic maps, $K$ is the transition kernel, $K'$ is its flipped version, and $\tau_0$ is the probability of staying in the same cell.
5.  **SLAM Integration**: The TGM is integrated with scan-matching SLAM. Instead of matching scans to a general grid, the algorithm matches only against the static part of the TGM ($M^s$) to minimize:

$$
x_{t}=\mathop{\text{a r g}\text{m i n}}_{x}{\sum_{k=1}^{K} (1-M_{\mathrm{smooth}}^{s}[T_{x}z_{t}^{k}])^{2}}
$$

### Key Quantitative Results
*   **Computational Efficiency**: The Python implementation achieves an update rate of $0.1\text{s}$ for the TGM and $0.015\text{s}$ for the SLAM component, matching the $10\text{Hz}$ frequency of the lidar sensor.
*   **Experimental Parameters**: 
    *   Priors: $p(m_{t,i}^s) = 0.3$ and $p(m_{t,i}^d) = 0.3$.
    *   Lidar Range: Tested at $100\text{m}$ for general mapping and $25\text{m}$ for challenging SLAM scenarios.
*   **Performance**: In highly dynamic scenarios (e.g., vehicles starting simultaneously at a traffic light), TGM prevented the SLAM algorithm from wrongly matching scans to moving vehicles, maintaining localization accuracy where OGM and c-OGM failed.

### Limitations
*   **Pose Assumption**: The core method assumes known poses, though it can be coupled with external SLAM.
*   **Transition Model**: The transition distribution $p(\tau_{j,i})$ is derived from a maximum speed model rather than being learned from application-specific data.
*   **Lack of Velocity**: The current framework does not provide explicit velocity estimates for dynamic cells.
*   **Class Agnosticism**: The model does not currently incorporate semantic class information to apply different transition models to different types of obstacles.
