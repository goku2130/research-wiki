---
id: link:simple-statistical-gradient-following-al
type: web
title: Simple statistical gradient-following algorithms for connectionist reinforcement
  learning
url: https://link.springer.com/article/10.1007/BF00992696
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

This article introduces REINFORCE algorithms, a class of associative reinforcement learning algorithms for connectionist networks with stochastic units.

**Core Problem:**
The central problem addressed is how to enable connectionist networks with stochastic units to learn to maximize expected reinforcement in both immediate and certain delayed reinforcement tasks. This learning must occur without explicitly computing or storing information for gradient estimates.

**Method/Recipe:**
REINFORCE algorithms operate by adjusting network weights in a direction that follows the gradient of expected reinforcement. The core idea is that the expected value of the product of the reinforcement signal and the "eligibility" of a weight change is proportional to the gradient of the expected reinforcement with respect to that weight.

The general form of the weight update rule for a weight $w_{ij}$ connecting unit $i$ to unit $j$ is:
1.  **Define the Reinforcement Signal:** Let $r$ be the reinforcement signal received by the network.
2.  **Define the Probability of Output:** For each stochastic unit $k$, let $y_k$ be its output and $p_k$ be the probability of that output given the unit's input and weights.
3.  **Define the Eligibility of a Weight:** The eligibility of a weight $w_{ij}$ is defined as the partial derivative of the logarithm of the probability of the network's output with respect to that weight. This is represented as $\frac{\partial \ln p}{\partial w_{ij}}$.
4.  **Update Rule (General Form):** The change in weight $\Delta w_{ij}$ is given by:
    $\Delta w_{ij} = \alpha (r - b) \frac{\partial \ln p}{\partial w_{ij}}$
    where:
    *   $\alpha$ is a positive learning rate.
    *   $r$ is the reinforcement signal.
    *   $b$ is a "baseline" term, which can be any function that does not depend on the weights $w_{ij}$ and serves to reduce variance without changing the expected value of the update.

**Key Formulas (in LaTeX):**

The fundamental property exploited by REINFORCE algorithms is:
$\nabla_w E[r] = E[r \nabla_w \ln p]$
where $E[\cdot]$ denotes the expected value, $r$ is the reinforcement signal, $p$ is the probability of the network's output given its weights $w$, and $\nabla_w$ is the gradient with respect to the weights.

The general weight update rule is:
$\Delta w_{ij} = \alpha (r - b) \frac{\partial \ln p}{\partial w_{ij}}$

For specific types of stochastic units, the term $\frac{\partial \ln p}{\partial w_{ij}}$ takes different forms:

*   **Bernoulli Stochastic Units (binary output $y \in \{0, 1\}$):**
    If the probability of output $y=1$ is $p_1 = f(\text{net})$, where $\text{net}$ is the net input to the unit and $f$ is a differentiable activation function (e.g., logistic sigmoid), then:
    $\frac{\partial \ln p}{\partial w_{ij}} = (y_j - p_j) \frac{\partial \text{net}_j}{\partial w_{ij}}$
    where $p_j$ is the probability of unit $j$ outputting 1, and $\frac{\partial \text{net}_j}{\partial w_{ij}}$ is typically $y_i$ (the output of the preceding unit $i$).

*   **Gaussian Stochastic Units (real-valued output $y$):**
    If the output $y$ is drawn from a Gaussian distribution with mean $\mu = f(\text{net})$ and fixed variance $\sigma^2$, then:
    $\frac{\partial \ln p}{\partial w_{ij}} = \frac{(y_j - \mu_j)}{\sigma^2} \frac{\partial \mu_j}{\partial w_{ij}}$
    where $\mu_j$ is the mean output of unit $j$, and $\frac{\partial \mu_j}{\partial w_{ij}}$ is typically $f'(\text{net}_j) y_i$.

**Key Quantitative Results and Numbers:**
The article states that REINFORCE algorithms make weight adjustments in a direction that lies along the gradient of expected reinforcement. This is a theoretical result concerning the *expected* direction of weight change, not a specific numerical performance metric. The algorithms are applicable to both immediate-reinforcement tasks and "certain limited forms of delayed-reinforcement tasks." No specific numerical performance results (e.g., accuracy, convergence rates, or task-specific scores) are provided in the abstract. The paper also mentions that some specific examples of these algorithms bear a close relationship to existing algorithms, while others are novel.

**Stated Limitations:**
The abstract explicitly mentions that the algorithms are applicable to "certain limited forms of delayed-reinforcement tasks." This implies that their applicability to general delayed reinforcement scenarios might be restricted or require further development. The article also notes that while the algorithms follow the gradient of expected reinforcement, they do so "without explicitly computing gradient estimates or even storing information from which such estimates could be computed," which could be seen as a limitation in terms of direct gradient computation but an advantage in terms of computational simplicity. The discussion section is stated to cover "what is known about their limiting behaviors," suggesting that full theoretical understanding or practical stability guarantees might still be an area of ongoing research.
