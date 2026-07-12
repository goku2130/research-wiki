---
id: arxiv:2110.05457
type: paper
title: 'Legged Robots that Keep on Learning: Fine-Tuning Locomotion Policies in the
  Real World'
url: https://arxiv.org/abs/2110.05457
retrieved: '2026-07-12'
maturity: comprehensive
topic: entropy-and-exploration
---

# Legged Robots that Keep on Learning: Fine-Tuning Locomotion Policies in the Real World

### Core Problem
Designing robust controllers for legged robots is challenging because it is nearly impossible to predict and enumerate all potential real-world environmental conditions during training. While reinforcement learning (RL) can produce robust controllers, they often rely on "environment engineering" (simulating a vast range of conditions) to achieve zero-shot generalization. When a robot encounters a test-time environment that deviates significantly from its training distribution, these controllers often fail. The authors address the need for a system that allows a robot to autonomously recover from failure and fine-tune its locomotion policies in the real world.

### Method/Recipe
The authors propose a framework for autonomous real-world fine-tuning using an A1 quadruped robot. The process follows these steps:

1.  **Simulation Pre-training:** To ensure safety and sample efficiency, policies $\pi_i$ for various skills (e.g., forward/backward pacing, side-stepping) and a recovery policy are first trained in a PyBullet simulation.
2.  **Motion Imitation:** Skills are learned by imitating reference motion clips (mocap or artist-generated). The reward function encourages the robot to track target poses at each timestep.
3.  **Autonomous Recovery:** A specialized recovery policy is trained to return the robot to a standing pose from random orientations. This is achieved by modifying the imitation reward: the robot is first rewarded for rolling right-side up, and once upright, it is rewarded for imitating a standing pose.
4.  **Off-Policy RL (REDQ):** The system utilizes Randomized Ensembled Double Q-learning (REDQ). This algorithm uses an ensemble of Q-functions $Q_{\theta} = \{Q_{\theta^k}\}_{k=1}^{N_{\text{ensemble}}}$ and samples a random subset to compute target values, reducing overestimation and improving sample efficiency.
5.  **Real-World Fine-Tuning:**
    *   The pre-trained policies are deployed in the real world.
    *   Replay buffers $\mathcal{D}$ are reset when entering a new environment to account for dynamics shifts.
    *   The robot executes a skill; if it falls, the recovery policy automatically resets it.
    *   The policy is updated online using REDQ based on real-world transitions.
6.  **State Estimation:** Onboard sensors (IMU and foot contact sensors) are used via a Kalman filter to estimate linear root velocity and orientation, removing the need for external motion capture.

### Key Formulas
The reward function $r_t$ at each timestep is a weighted sum of pose, velocity, and other components:

$$
r_{t}=\boldsymbol w^{\mathfrak p}\boldsymbol r_{t}^{\mathfrak p}+\boldsymbol w^{\mathfrak v}r_t boldsymbol{{t}}^{\mathfrak v}+\boldsymbol w^{\mathfrak e}r_{t}^{\mathfrak e}+\boldsymbol w^{\mathfrak r}\boldsymbol r_{\boldsymbol{t}}^{\mathfrak r}{\mathfrak p}+\boldsymbol w^{\mathfrak r}\boldsymbol r_{\boldsymbol{t}}^{\mathfrak r}{\mathfrak r}{\mathfrak r}
$$

With weights: $w^{\mathsf{p}}=0.5,\ w^{\mathsf{v}}=0.05,\ w^{\mathsf{s}}=0.2,\ w^{\mathsf{rp}}=0.15,\ w^{\mathsf{rv}}=0.1$.

The pose reward $r_t^{\mathsf{p}}$, which encourages matching joint rotations $\hat{q}_{t}^{j}$ from the reference motion, is defined as:

$$
r_{t}^{\mathsf{p}}=\text{e x p}\left[-5\sum_{j}||\hat{q}_{t}^{j}-q_{t}^{j}||^{2}\right]
$$

### Key Quantitative Results
*   **Simulation Comparison:** In unexpected environments (rugged heightfields and low-friction surfaces), the fine-tuning method outperformed Rapid Motor Adaptation (RMA) and latent space search methods. While RMA and latent methods excelled in environments similar to training, they suffered significant performance drops in "unexpected" conditions.
*   **Algorithm Efficiency:** REDQ demonstrated superior adaptation performance compared to Soft Actor-Critic (SAC).
*   **Real-World Performance:** The robot successfully fine-tuned pacing on an outdoor lawn and side-stepping on carpet, a doormat with crevices, and 4cm-thick memory foam.
*   **Recovery Success:** The learned recovery policy was 100% successful across all experiments and was faster and more reliable on memory foam than the manufacturer's built-in rollover controller.
*   **Training Time:** The robot achieved consistent skill improvement with a modest amount of data, totaling approximately 2 hours of operation.

### Stated Limitations
*   **Workspace Drift:** The robot occasionally drifted outside the designated workspace area, requiring a human supervisor to manually reposition it.
*   **Scope of Learning:** The current experiments focused on fine-tuning in each environment separately rather than implementing a continuous, lifelong learning process across diverse, changing environments.
