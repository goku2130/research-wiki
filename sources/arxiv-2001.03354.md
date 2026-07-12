---
id: arxiv:2001.03354
type: paper
title: Learning credit assignment
url: https://arxiv.org/abs/2001.03354
retrieved: '2026-07-12'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Learning Credit Assignment

### Core Problem
The "black box" nature of deep learning makes it difficult to understand how a network coordinates a vast number of parameters to achieve a specific decision. This is known as the **credit assignment problem (CAP)**: the challenge of determining how much credit a specific component (neuron or connection) deserves for a particular macroscopic behavior. The authors seek to bridge the gap between microscopic interactions and macroscopic output by modeling the network as an ensemble of subnetworks rather than a single deterministic entity.

### Method: Generalized Backpropagation (gBP)
The authors propose a mean-field learning model that treats synaptic weights as random variables following a **Spike and Slab (SaS)** distribution.

#### 1. Weight Distribution
Each connection weight $w_{ik}^l$ from neuron $i$ (layer $l$) to neuron $k$ (layer $l+1$) is modeled as:

$$
P(w_{ik}^l) = \pi_{ik}^l \delta(w_{ik}^l) + (1-\pi_{ik}^l) \mathcal{N}(w_{ik}^l | m_{ik}^l, \Xi_{ik}^l)
$$

Where:
*   $\pi_{ik}^l$ is the "spike" probability (probability that the connection is absent).
*   $m_{ik}^l$ and $\Xi_{ik}^l$ are the mean and variance of the "slab" (Gaussian distribution).

#### 2. Network Architecture and Forward Pass
The network consists of $L$ layers with ReLU activations $h = \max(0, z)$ for hidden layers and a softmax function for the output layer. The pre-activation $z$ is defined as:

$$
z_{k}^{l+1} = \frac{1}{\sqrt{N_{l}}}\sum_{i}w_{i k}^{l}h_{i}^{l}
$$

To make the model computationally tractable, the authors use a **reparameterization trick**. Given a large layer width $N_l$, the pre-activation is approximated as a Gaussian $\mathcal{N}(G, \Delta^2)$:

$$
z_{i}^{l} = G_{i}^{l} + \epsilon_{i}^{l}\Delta_{i}^{l}
$$

where $\epsilon_i^l$ is a standard Gaussian random variable quenched for each mini-epoch.

#### 3. Learning Recipe
Instead of updating point-estimate weights (as in standard backpropagation), the model updates the SaS hyperparameters $\theta_{ik}^l = (\pi_{ik}^l, m_{ik}^l, \Xi_{ik}^l)$ using gradient descent to minimize the cross-entropy objective function:

$$
\mathcal{C} = -\sum_{i}\hat{h}_{i}\ln{h_{i}}
$$

The gradients are computed via a generalized backpropagation (gBP) process:
1.  **Forward Pass:** Compute activations using the reparameterized $z_i^l$.
2.  **Backward Pass:** Compute the error signal $\mathcal{K}_i^{l+1} = \frac{\partial \mathcal{C}}{\partial z_i^{l+1}}$ using the chain rule.
3.  **Hyperparameter Update:** Update $\theta_{ik}^l$ using the gradient $\Delta\theta_{ki}^l = -\eta \mathcal{K}_i^{l+1} \frac{\partial z_i^{l+1}}{\partial \theta_{ki}^l}$.

### Key Quantitative Results
*   **Performance:** gBP achieves test accuracies similar to or better than standard backpropagation (BP).
*   **Weight Categorization:** The model identifies three distinct roles for synaptic weights:
    1.  **Very Important (VIP):** $\pi \approx 0$ and $\Xi \approx 0$ (deterministic and essential).
    2.  **Unimportant (UIP):** $\pi \approx 1$ (can be pruned).
    3.  **Variability:** Weights with broad distributions that may encode nuisance factors.
*   **Sparsity and Entropy:** The sparsity (fraction of $\pi=1$) and connection entropy $S_\ell$ exhibit a **non-monotonic behavior** across layers. Both grow in the middle stages and decrease toward the output. This suggests a process of **encoding $\to$ recoding $\to$ decoding**, where the middle stage provides the most degrees of freedom to manipulate the hypothesis space.
*   **Robustness:** Perturbing VIP weights significantly impacts test performance, whereas perturbing UIP weights does not.

### Limitations and Future Work
The authors note that the point solution ($\pi=0, \Xi=0$) is not a stable attractor for gBP. The current framework assumes weights are factorized across connections; future directions include considering correlations among weights and extending the framework to temporal credit assignment in recurrent neural networks.
