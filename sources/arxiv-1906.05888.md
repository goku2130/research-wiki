---
id: arxiv:1906.05888
type: paper
title: Off-Policy Deep Reinforcement Learning without Exploration
url: https://arxiv.org/abs/1906.05888
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

The paper addresses the challenge of aligning highly sparse 3D point clouds captured by depth sensors (e.g., Kinect, LiDAR) without RGB color information. In sparse regimes, traditional correspondence-based methods like ICP fail because consecutive scans rarely observe identical 3D points. The authors reframe sparse scan registration as a generalized relative pose estimation problem, leveraging line intersection constraints rather than explicit point or line correspondences.

The registration pipeline proceeds through a deterministic sequence. First, raw point clouds are organized into a regular grid structure. Horizontal and vertical scan-lines are then fitted with line segments using RANSAC, yielding H-lines and V-lines. Candidate line intersection constraints are identified by pairing H-lines from one frame with V-lines from the adjacent frame, selecting pairs within a spatial distance threshold. Plane correspondences are similarly candidate-selected based on normal vector alignment (<20°) and centroid proximity. Pose estimation is then performed using either an alternating projection algorithm or closed-form minimal solvers, embedded within a RANSAC framework. The alternating projection iteratively satisfies two constraints: intersection (moving line endpoints to force pairwise intersections) and rigidity (enforcing rigid body transformations within each scan). Minimal solvers exploit geometric canonicalization to reduce degrees of freedom. Specifically, the 1L2P solver uses one line intersection and two plane correspondences, reducing the relative motion to a single translation along the x-axis. The 3L1P solver uses three line intersections and one plane correspondence, reducing the pose to a rotation about the z-axis and two in-plane translations. After solving, inverse canonical transformations recover the full relative pose. Inlier counting weights line pairs by cluster frequency, and the pipeline is iterated multiple times using the best prior transformation as initialization.

Mathematically, lines are represented in Plücker coordinates $l, m \in \mathbb{R}^6$. The intersection constraint for a relative pose $(\mathbf{R}, \mathbf{t})$ is formulated as:
\[
m^T \begin{bmatrix} -[\mathbf{t}]_\mathrm{s} \mathbf{R} & \mathbf{R} \\ \mathbf{R} & \mathbf{0} \end{bmatrix} l = 0.
\]
The alternating projection enforces rigidity via transformations $\mathcal{T}_i$ and $\mathcal{T}'_i$:
\[
(a_i, b_i) = \mathcal{T}_i (a_i^s, b_i^s), \quad (c_i, d_i) = \mathcal{T}'_i (c_i^s, d_i^s).
\]
For the 1L2P case, canonicalization yields $\mathbf{R} = \mathbf{I}$ and $\mathbf{t} = [t_1, 0, 0]^T$, linearizing the intersection constraint into a single unknown. For 3L1P, the canonical pose becomes $\mathbf{R} = \begin{bmatrix} c\theta & -s\theta & 0 \\ s\theta & c\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$ and $\mathbf{t} = [t_1, t_2, 0]^T$, yielding a quartic polynomial in $s\theta$ solvable in closed form.

Quantitative evaluations demonstrate superior performance on sparse data. On KITTI sequences 05, 06, and 07, the proposed 7L method achieves mean translation errors of 0.0179 m, 0.0227 m, and 0.0176 m, with rotation errors of 0.0955°, 0.0792°, and 0.0838°, respectively, outperforming LOAM across all metrics. On 36× down-sampled KITTI data, the method records 7.42% translation error and 0.0234 deg/m rotation error, compared to LOAM’s 52.23% and 0.3485 deg/m. On the TUM dataset using 1/100th resolution depth data without RGB, the 7L solver achieves translation errors of 5.047 mm, 11.633 mm, and 12.043 mm, and rotation errors of 0.5212°, 0.6913°, and 0.6949°, significantly surpassing a standard 3-point RANSAC baseline that yields 24 mm and 0.740° on full-resolution RGB data. Computationally, the 7-line solver averages 0.022 s, while the 3L1P and 1L2P minimal solvers require 4.9 μs and 0.64 μs, respectively.

The authors acknowledge several limitations. The method assumes sensor motion smoothness to generate valid line constraints and does not explicitly correct for motion-induced distortions, unlike LOAM. It relies on the availability of sufficiently oriented plane correspondences to avoid degeneracy, which may not be guaranteed in sparse, noisy scans. Furthermore, the algorithm excludes boundary and edge points, potentially limiting performance in geometrically complex scenes. Finally, while alternating projection efficiently handles near-minimal constraints, it remains sensitive to initialization and requires careful threshold tuning to balance intersection satisfaction against rigidity preservation.
