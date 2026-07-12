---
id: intellisys:stellaris-staleness-aware-distributed-re
type: web
title: 'Stellaris: Staleness-Aware Distributed Reinforcement Learning with Adaptive
  Rollout Control'
url: https://intellisys.haow.us/assets/pdf/SC41406.2024.00045.pdf
retrieved: '2026-07-12'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# Stellaris: Staleness-Aware Distributed Reinforcement Learning with Serverless Computing

**Stellaris** is a generic asynchronous learning paradigm designed to accelerate distributed Deep Reinforcement Learning (DRL) training by leveraging serverless computing.

### Core Problem
Distributed DRL typically relies on serverful infrastructures or synchronous learning to maintain stability. Serverful setups suffer from high costs and low resource utilization due to idle resources. While serverless actors can reduce costs, a centralized learner often becomes a bottleneck. Transitioning to asynchronous multi-learner serverless architectures introduces two primary technical challenges:
1.  **Dynamic Staleness:** Asynchronous learners compute gradients based on potentially outdated policies, which can degrade convergence or cause divergence.
2.  **Cross-Learner Policy Drift:** Because each asynchronous learner maintains a unique policy, standard importance sampling (which assumes a single learner policy) fails to prevent unstable, "wild" policy updates during aggregation.

### Method/Recipe
Stellaris employs an actor-learner architecture coordinated via a **Distributed Cache** (e.g., Redis). The workflow consists of three primary steps:

1.  **Importance Sampling Driven Trajectory Collection:** Actors pull the latest policy from the cache, interact with the environment to sample trajectories, and submit these batches back to the cache.
2.  **On-Demand Gradient Calculation:** Serverless learner functions are invoked based on the volume of available data. To optimize efficiency, a **GPU data loader** daemon pre-loads trajectories into GPU memory, and **hierarchical data passing** (shared memory, RPC, and cache) is used to minimize communication overhead.
3.  **Staleness-Aware Gradient Aggregation:** A serverless **Parameter Function** aggregates gradients. To prevent instability, it delays aggregation until the average staleness $\bar{\delta}$ of the gradient queue falls below a dynamic threshold $\beta$.

### Key Formulas
**Global Importance Sampling Truncation:**
To prevent policy drift across multiple asynchronous learners, Stellaris truncates the importance sampling ratio $R$ using a global view:

$$
R' := \min\left(\left|\min_{i}\left(\frac{\pi_{\theta_{i}}}{\mu_{\theta}}\right)\right|, \rho\right)
$$

where $\mu_{\theta}$ is the actor policy, $\pi_{\theta_{i}}$ are the learner policies, and $\rho$ is the truncation threshold. The resulting policy gradient is:

$$
\nabla J(\pi_{\theta}) = \mathbb{E}_{t}\left[\mathbb{E}_{\tau_{t}\sim\mu_{\theta}}\left[\min\left(\left|\min_{i}\left(\frac{\pi_{\theta_{i}}}{\mu_{\theta}}\right)\right|, \rho\right)A_{t}\right]\right]
$$

**Adaptive Staleness Threshold:**
The threshold $\beta$ decays over training rounds $k$ to transition from fast initial learning to stable convergence:

$$
\beta_{k} := \delta_{\max} \times d^{k}, \quad d \in (0, 1]
$$

where $\delta_{\max}$ is the maximum staleness in a pure asynchronous environment and $d$ is the exponential decay factor.

**Staleness-Modulated Policy Update:**
The learning rate $\alpha_0$ is modulated by the staleness $\delta_c$ of the gradient:

$$
\alpha_{c} := \frac{\alpha_{0}}{\sqrt{\delta_{c}}}, \text{ if } \delta_{c} > v
$$

The final aggregated gradient $g_c$ for $H_c$ gradients is:

$$
g_{c} := \frac{1}{H_{c}}\sum_{i=1}^{H_{c}}\frac{\alpha_{0}}{\sqrt{\delta_{j}}}g_{i,j}, \quad \theta_{c+1} := \theta_{c} - g_{c}
$$

### Key Quantitative Results
Evaluations were conducted on AWS EC2 (16 NVIDIA V100 GPUs, 960 CPU cores) across MuJoCo and Atari environments:
*   **Training Quality:** Stellaris achieved up to **$2.2\times$ higher rewards** compared to state-of-the-art (SOTA) baselines.
*   **Cost Efficiency:** Training costs were reduced by up to **41%**. Specifically, costs for PPO and IMPACT were reduced by 31% and 30%, respectively; costs for RLlib and MinionsRL were reduced by 38% and 41%.
*   **HPC Scalability:** When integrated with PAR-RL on an HPC cluster, Stellaris improved final rewards by **$2.4\times$** (Hopper) and **$1.1\times$** (Qbert) while reducing costs by 19% and 34%.
*   **Overhead:** The total system overhead for the proposed components incurred **less than 5% delay** per training round.

### Stated Limitations
The authors note that existing serverless platforms lack native GPU support. Consequently, the prototype was implemented using a custom serverless container cluster on AWS EC2 accelerated instances using Docker and the NVIDIA container runtime to enable GPU functionality for the learner and parameter functions.
