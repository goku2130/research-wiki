---
id: arxiv:2402.05402
type: paper
title: 'Inference Scaling Laws: An Empirical Analysis of Compute-optimal Inference'
url: https://arxiv.org/abs/2402.05402
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

The provided source presents a comprehensive survey on full-duplex (FD) network design, detailing the transition from link-level feasibility to large-scale network deployment. The core problem centers on managing inter-node interference (INI)—comprising self-interference (SI), cross-link interference (CLI), and inter-cell interference (ICI)—which emerges when nodes simultaneously transmit and receive. Unlike half-duplex (HD) networks, FD systems require coordinated PHY, MAC, and RRC layer interventions to preserve spectral efficiency gains while ensuring fairness, stability, and backward compatibility with legacy infrastructure.

The methodological framework proceeds through four integrated stages. First, at the PHY layer, analog and digital self-interference cancellation (SIC) must achieve sufficient suppression to enable stable FD operation. Second, MAC protocol design for WLAN and cellular networks must resolve hidden node problems, INI, and fairness trade-offs using contention-based, centralized, or hybrid scheduling mechanisms that leverage signal-to-interference ratio (SIR) maps, busy tones, or RTS/CTS handshakes. Third, cellular user scheduling is formulated as a joint user pairing and power allocation optimization problem. The scheduler maximizes a weighted objective function subject to transmit power bounds and minimum QoS thresholds. Fourth, CLI handling requires standardized measurement protocols (CLI-SRS and CLI-RSSI) followed by mitigation strategies such as cell clustering, coordinated UL/DL configuration, spatial precoding, or advanced architectures like CoMPflex and distributed antenna systems. Finally, system-level validation employs 3D ray-tracing-based simulations incorporating real-world channel models and analog beamforming to evaluate multi-cell, multi-user FD performance.

Key mathematical formulations define the FD channel and optimization objectives. The received signal at a BS node $b$ is $y_{b\leftarrow j}^{Rx} = y_{b\leftarrow j}^{UL} + y_b^{SI} + n_b$, where residual SI power is $I_{BS}^{SI} = G_b^{SI} p_b^{DL}$. The downlink and uplink SINR expressions are:

$$
\text{SINR}_{b\rightarrow i}^{\text{DL}} = \frac{p_b^{\text{DL}} G_{b,i}^{\text{BU}}}{p_j^{\text{UL}} G_{i,j}^{\text{UU}} + \sum_{b' \in \mathbb{B}} p_{b'}^{\text{DL}} G_{b',i}^{\text{BU}} + N_{\text{UE}}},
$$

$$
\text{SINR}_{b \leftarrow j}^{\mathrm{UL}} = \frac{p_{j}^{\mathrm{UL}} G_{b,j}^{\mathrm{BU}}}{\sum_{b' \in \mathbb{B}} p_{b'}^{\mathrm{DL}} G_{b',j}^{\mathrm{BU}} + p_{b}^{\mathrm{DL}} G_{b}^{\mathrm{SI}} + \mathcal{N}_{\mathrm{BS}}}.
$$

Throughput follows $R_{j}^{\mathrm{UL}} = \mathrm{BW}_{j}^{\mathrm{UL}} \log(1 + \mathrm{SINR}_{b \leftarrow j}^{\mathrm{UL}})$ and $R_{i}^{\mathrm{DL}} = \mathrm{BW}_{i}^{\mathrm{DL}} \log(1 + \mathrm{SINR}_{b \rightarrow i}^{\mathrm{DL}})$. The network scheduling problem is formulated as:

$$
\max_{\{\pi_b\}, \{p^{\mathrm{DL}}\}, \{p^{\mathrm{UL}}\}} \sum_{b \in \mathbb{B}} \sum_{i \in \mathbb{U}_b} \{w(R_i^{\mathrm{DL}}) + w(R_j^{\mathrm{UL}})\}
$$

subject to $0 \leq p^{\mathrm{DL}} \leq P_{\max}^{\mathrm{DL}}$, $0 \leq p^{\mathrm{UL}} \leq P_{\max}^{\mathrm{UL}}$, and $R \geq R_{\text{threshold}}$. The transmit power disparity is defined as $\delta = P_{\max}^{\mathrm{UL}} / P_{\max}^{\mathrm{DL}}$.

Quantitative results from proof-of-concept (PoC) studies demonstrate spectral efficiency improvements of 1.7× to 1.9× over HD baselines, with SIC performance exceeding 110 dB. System-level analyses indicate that CLI/ICI suppression targets 10–15 dB for BS-to-BS interference. Fairness metrics show that proportional fairness convergence slows at 95 dB SIC compared to perfect cancellation, while max-min fairness rates decline as SIC degrades from 110 dB to 60 dB. Machine learning approaches, including reinforcement learning and deep neural networks, are proposed to approximate NP-hard scheduling solutions, though they require extensive channel state information.

The source identifies several limitations. INI, particularly CLI and ICI, remains difficult to measure and cancel compared to SI. The scheduling optimization is non-convex, nonlinear, and NP-hard, necessitating approximation methods like successive convex approximation or geometric programming that cannot guarantee global optima. Practical deployments assume UE nodes operate in half-duplex mode due to hardware complexity and power constraints, reducing FD utilization. Backward compatibility with legacy HD nodes introduces fairness degradation, as HD nodes experience extended inter-frame space (EIFS) waiting times. Furthermore, multi-cell CLI coordination incurs prohibitive signaling overhead, and current ML-based schedulers demand substantial computational resources and CSI feedback, limiting real-time scalability.
