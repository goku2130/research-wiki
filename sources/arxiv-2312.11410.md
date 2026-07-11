---
id: arxiv:2312.11410
type: paper
title: Process Reward Models for Mathematical Reasoning
url: https://arxiv.org/abs/2312.11410
retrieved: '2026-07-11'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

**Core Problem**
The paper addresses the active search and coverage task, where a mobile 3D sensor must optimize its trajectory to locate and scan target objects within an environment as rapidly as possible. The problem is formalized as a partially observable Markov decision process (POMDP) operating directly on 3D point clouds. Point clouds are preferred over voxel grids because they avoid the cubic computational and memory costs of scaling resolution, while providing a canonical spatial representation that facilitates sim-to-real transfer.

**Method/Recipe Step by Step**
The proposed solution is an end-to-end deep reinforcement learning pipeline structured as follows:
1. **Simulation & Data Acquisition:** A ROS/Gazebo simulator generates incremental point cloud observations using a simulated Kinect sensor (0.5–5 cell range, 640×380 resolution). New measurements are concatenated with historical data and downsampled using dual voxel filters (0.88 cells for background, 0.22 cells for targets) to eliminate duplicates and prioritize target shape fidelity.
2. **State, Action, and Reward Definition:** The state comprises the accumulated point cloud segmented into floor, wall, and target classes, plus the agent’s pose. Actions are discrete translational (forward/back/left/right) and rotational (±90°) movements. Illegal transitions (collisions) are stored in the experience replay buffer to enable collision-free exploration. The reward incentivizes discovering new target and floor points.
3. **Neural Network Architecture:** The policy network integrates four sequential stages:
   - *Embedding:* Deep hierarchical feature learning uses Farthest Point Sampling (FPS) to select representative points, followed by K-Nearest Neighbors (KNN) grouping. Local features are normalized by subtraction, concatenated, processed through linear layers, and reduced via max pooling over $K$ neighbors. This repeats hierarchically.
   - *Multi-head Attention:* Offset attention blocks from the Point Cloud Transformer (PCT) are implemented with multiple heads, concatenated, and projected back to 256 features.
   - *Spatial Preservation:* Max pooling is applied across points, and $N$ copies of the pooled result are concatenated with the preceding tensor to retain spatial relationships.
   - *RL Tail:* A distributional dueling network (Rainbow algorithm) with spectral normalization estimates return distributions across 51 discrete bins ($V_{\min}=0, V_{\max}=415$). The dueling structure decomposes $Q$-values into value and advantage streams, normalized to ensure identifiability.
4. **Training Protocol:** The agent is trained using Rainbow RL with an $\epsilon$-greedy policy decaying linearly from 1 to 0 over 200,000 steps. Episodes consist of 65 steps, with spectral normalization applied to the first fully connected layer of the value/advantage streams.

**Key Formulas**
- Success metric (target coverage over trajectory $\mathbf{q}$): $V(\mathbf{q}) = \sum_{t \ge 0} |T_{t+1} - T_t|$
- State-action value function: $Q_\pi(s, a) := \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$
- Reward function: $r_t = |T_{t+1}| - |T_t| + |F_{t+1}| - |F_t|$
- Coverage constraint ensuring all points are represented: $|P^b| < |P_{\mathrm{fps}}^b| \cdot |P_{\mathrm{knn}}^b|$
- Dueling network $Q$-value decomposition: $Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)$

**Key Quantitative Results and Numbers**
Ablation studies confirm that FPS reduces network parameters while improving performance. Multi-head attention (8 heads) accelerates initial learning convergence compared to single-head variants, though both ultimately achieve identical final performance, indicating redundant feature learning across heads. In direct comparison against a greedy baseline that maximizes immediate rewards, the optimized RL agent ($Cs256s128h8$) achieves an average of 356.9 covered points over 100 steps and 100 episodes, outperforming the greedy strategy’s 311.8 points. The RL agent also exhibits tighter confidence intervals and avoids the ~45-step stagnation observed in the greedy approach. Pre-segmentation networks referenced in the text achieve 86.4% (PCT) to 87.2% (PointStack/SPoTr) accuracy on partShapeNet.

**Stated Limitations**
The framework currently requires an a priori discriminator to pre-segment point clouds into floor, wall, and target classes, which must be replaced by an end-to-end segmentation network in future iterations. Experiments are confined to a single-room environment with two cylinders; scaling to multi-room layouts or cluttered scenes would substantially increase computational complexity and training duration. Additionally, the greedy baseline relies on an oracle to predict future observations, rendering it impractical for real-world deployment despite its use as a performance benchmark.
