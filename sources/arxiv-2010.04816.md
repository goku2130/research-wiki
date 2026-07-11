---
id: arxiv:2010.04816
type: paper
title: Characterizing Policy Divergence for Personalized Meta-Reinforcement Learning
url: https://arxiv.org/abs/2010.04816
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

# Characterizing Policy Divergence for Personalized Meta-Reinforcement Learning

### Core Problem
The author addresses the challenge of few-shot reinforcement learning (RL) in personalized settings, where a population of entities (e.g., patients) may share the same state and action spaces but possess unique transition dynamics $P^i$. In such diverse environments, standard meta-learning approaches like Model-Agnostic Meta-Learning (MAML), which seek a single optimal initialization for all tasks, may be sub-optimal. The core problem is how to effectively leverage prior experiences by identifying and utilizing the most relevant past policies based on environment similarity to accelerate adaptation to a new, unseen entity.

### Method: Cluster-Adaptive Meta-Learning (CAML)
The proposed algorithm, Cluster-Adaptive Meta-Learning (CAML), extends the meta-learning paradigm by maintaining multiple representative policy initializations (medoids) rather than a single global initialization. The process is divided into three primary phases:

**1. Policy Divergence Estimation**
To determine similarity between policies, the author uses occupancy measures. Because calculating the full occupancy measure is computationally impractical, the author proposes a sampled approach:
*   **State Sampling:** A Kernel Density Estimate (KDE) $\hat{q}$ is calculated from observed trajectories to sample a subset of states $\hat{s}_j$.
*   **Measure Estimation:** For each policy $\pi_i$, the conditional probability $\pi(a|s)$ is obtained from the model network, and the state probability density $\hat{q}^{(i)}(s)$ is calculated from the entity's specific trajectories.
*   **Divergence Calculation:** The similarity between two policies is quantified using the symmetric Jensen-Shannon (JS) divergence.

**2. Clustering**
The author employs a $K$-medoids clustering algorithm on the pairwise JS divergence distance matrix. This groups policies into $k$ clusters, identifying $k$ medoid policies that serve as representative initializations for different types of environment dynamics.

**3. Test-Time Adaptation**
When encountering a new environment, the agent must select the most appropriate initialization from the $k$ medoids. This is framed as a multi-armed bandit problem:
*   The agent performs $K$ few-shot rollouts using the available medoid policies.
*   The medoid policy that yields the highest cumulative reward is selected as the initialization.
*   The policy is then fine-tuned using Vanilla Policy Gradient (VPG).

### Key Formulas
The occupancy measure $\rho_{\pi}(s,a)$ is defined as:

$$
\rho_{\pi}(s,a)=\pi(a|s)\sum_{t=1}^{T}P(S_{t}=s|\pi)
$$

The distance between two policies is measured via the Jensen-Shannon divergence:

$$
D_{\mathrm{JS}}(\rho_{s}^{i},\rho_{s}^{j})\triangleq D_{\mathrm{KL}}\Big(\rho_{s}^{i}\,\big\vert\big\vert\,\frac{\rho_{s}^{i}+\rho_{s}^{j}}{2}\Big)+D_{\mathrm{KL}}\Big(\rho_{s}^{j}\,\big\vert\big\vert\,\frac{\rho_{s}^{i}+\rho_{s}^{j}}{2}\Big)
$$

where $D_{\mathrm{KL}}$ denotes the Kullback–Leibler divergence.

### Quantitative Results
The author evaluated the method using a 2D continuous episodic gridworld where particles must reach a target at $(1,1)$ within 100 timesteps. The environment was personalized using 24 entity types (6 latent types with 4 variants each) with remapped control directions and added variance.

*   **Few-Shot Setting:** Adaptation was tested with $K=10$ trajectory rollouts.
*   **Training:** Models were trained for 100 iterations with a batch size of 10 using VPG.
*   **Performance:** CAML consistently outperformed several baselines, including:
    *   **Reptile:** A first-order meta-learning algorithm.
    *   **Joint Training:** Pretraining by randomly sampling from all support environments.
    *   **Random Initialization:** Initializing weights randomly.
    *   **Unmatched Pretraining:** Pretraining in a different environment.
*   **Upper Bound:** The only baseline to consistently outperform or match CAML was the "matched pretrained" policy (a policy pretrained specifically in the query environment).

### Limitations
The author notes that the current results are preliminary and that further baselines and experiments are required to fully validate the approach. The evaluation was limited to a specific 2D navigation testbed, and the effectiveness of the method relies on the assumption that occupancy measures derived from trajectories are sufficient to characterize environment similarity.
