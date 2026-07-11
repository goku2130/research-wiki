---
id: arxiv:2102.03740
type: paper
title: Ensemble perspective for understanding temporal credit assignment
url: https://arxiv.org/abs/2102.03740
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Ensemble Perspective for Understanding Temporal Credit Assignment

### Core Problem
Temporal credit assignment in Recurrent Neural Networks (RNNs) is challenging due to the difficulty of capturing long-term dependencies. Traditional training via Backpropagation Through Time (BPTT) produces a point-estimate of weights, which fails to account for weight uncertainty or identify the specific relevance of individual connections to the network's macroscopic behavioral output.

### Method
The authors propose an ensemble perspective where each connection is modeled by a **spike and slab (SaS)** probability distribution rather than a fixed value. This allows the network to learn a distribution over weights, supporting stochastic synaptic plasticity and sparse architectures.

#### Step-by-Step Recipe
1.  **Weight Distribution Definition**: Each weight $w$ (input, recurrent, or output) is assigned a SaS distribution:

$$
P(w) = \pi \delta(w) + (1 - \pi) \mathcal{N}(w | m, \Xi)
$$

    where $\pi$ is the spike probability (driving sparsity), $m$ is the mean of the Gaussian slab, and $\Xi$ is the variance (representing weight uncertainty).
2.  **Mean-Field Approximation**: To train the ensemble, the pre-activation $u_i(t)$ and output $z_k(t)$ are re-parametrized as Gaussian random variables using the central-limit theorem (assuming large fan-in).
3.  **Forward Pass**: The dynamics are computed as:

$$
h_i(t+1) = (1-\alpha)h_i(t) + \alpha u_i(t+1) + \sqrt{2\alpha\sigma^2}n_i
$$

$$
u_i(t+1) = G_i^{\text{rec}}(t) + G_i^{\text{in}}(t+1) + \epsilon_i^{\text{u}}(t+1)\sqrt{(\Delta_i^{\text{in}}(t+1))^2 + (\Delta_i^{\text{rec}}(t))^2}
$$

    where $G$ represents the mean contribution and $\Delta$ represents the variance.
4.  **Generalized BPTT (gBPTT)**: The hyper-parameters $\theta = (m, \pi, \Xi)$ are updated via gradient descent. Error signals $\delta_i(t) = \frac{\partial\mathcal{L}}{\partial h_i(t)}$ are propagated backward from the final time step $T$ to $0$.
5.  **Hyper-parameter Update**: Gradients are computed for $m, \pi,$ and $\Xi$ based on the sensitivity of the pre-activation $u_i(t)$ and output $z_k(t)$ to these parameters.

### Key Quantitative Results
The method was tested on pixel-by-pixel MNIST classification ($N=100$ neurons) and a multisensory integration (MSI) task ($N=150$ neurons).

*   **Performance**: The ensemble model achieved performance comparable to or better than traditional BPTT on MNIST. In the MSI task, it successfully reproduced the cognitive benefit of combining multiple sensory sources for decision-making.
*   **VIP Weights**: The model identified "Very Important" (VIP) weights (low $\pi$, low $\Xi$). While VIP weights constitute a minority (approximately 1%), pruning them deteriorated test performance to chance levels, whereas pruning an equal number of random weights had a negligible effect.
*   **Neural Selectivity**: Using the Selectivity Index (SLI):

$$
\mathrm{SLI}(i) = \frac{1}{1 - \frac{1}{N_s}} \left[ 1 - \frac{(\frac{1}{N_s} \sum_{s=1}^{N_s} r_{i,s})^2}{\frac{1}{N_s} \sum_{s=1}^{N_s} r_{i,s}^2} \right]
$$

    the authors found that neurons emerged with mixed selectivity (responding to a few stimulus types), uni-selectivity, or non-selectivity.
*   **Low-Dimensional Dynamics**: PCA revealed that the first three modes explained $>85\%$ of the total variance in both MNIST and MSI tasks, indicating that neural activity is embedded in a low-dimensional subspace.
*   **Symmetry Breaking**: In the 28-by-28 MNIST task, continuous symmetry (Gaussian $\Xi$) broke before discrete symmetry (Bernoulli $\pi$).

### Mechanistic Analysis
*   **Lazy Learning Regime**: For infinitely large networks, the authors derived a Recurrent Neural Tangent Kernel (RNTK). In this regime, the kernel $\Theta_0$ remains constant, and the output dynamics follow a closed-form solution.
*   **Feature Learning Regime**: For finite networks, the system escapes the lazy regime. Synaptic dynamics are embedded in a low-dimensional manifold characterized by a persistent net drift velocity along the first principal component.

### Stated Limitations
*   **RNTK Applicability**: The RNTK theory only accurately predicts the early stage of learning for finite-sized networks; later stages require active feature learning.
*   **Linear Approximation**: The linear approximation used to derive output dynamics in the lazy regime requires correction via higher-order non-linear terms to account for deviations in test error.
*   **Entropy Estimation**: The calculated network entropy $S$ is an approximate estimate, as exact computation is impossible.
