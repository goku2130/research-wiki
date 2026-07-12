---
id: arxiv:2402.09345
type: paper
title: 'InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward
  Modeling'
url: https://arxiv.org/abs/2402.09345
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-hacking
---

# InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward Modeling

### Core Problem
Reinforcement Learning from Human Feedback (RLHF) often suffers from **reward hacking** (or reward overoptimization), where a policy model optimizes for a proxy reward model (RM) but diverges from true human objectives. The authors attribute this to **reward misgeneralization**, where RMs rely on spurious features (e.g., response length bias) that correlate with training labels but are irrelevant to actual human preferences. This leads to poor generalizability when the policy model's response distribution shifts during the RL stage.

### Method
InfoRM addresses reward misgeneralization by applying the **Information Bottleneck (IB)** principle to reward modeling. The goal is to learn a compact latent representation $\boldsymbol{S}$ that retains information useful for predicting human preferences $\boldsymbol{Y}$ while filtering out irrelevant information from the input $\boldsymbol{X}$.

#### Step-by-Step Recipe:
1.  **Architecture**: The framework utilizes an LLM backbone as an encoder $f_\phi$ and an MLP as a decoder $g_\psi$.
2.  **Latent Representation**: The encoder $f_\phi$ outputs a mean $f_\phi^\mu(\boldsymbol{x})$ and a diagonal covariance $f_\phi^\sigma(\boldsymbol{x})$, modeling the latent representation $\boldsymbol{S}$ as a multivariate Gaussian.
3.  **Reparameterization**: To allow backpropagation, the latent variable is sampled using $h_\phi(\boldsymbol{x}, \epsilon) = f_\phi^\mu + f_\phi^\sigma(\boldsymbol{x})\epsilon$, where $\epsilon \sim \mathcal{N}(0,1)$.
4.  **Optimization**: The model optimizes a variational lower bound of the IB objective, balancing preference utility and information compression.
5.  **Overoptimization Detection**: The authors introduce the **Cluster Separation Index (CSI)** to detect reward hacking. They observed that overoptimized samples manifest as outliers in the IB latent space. The CSI is computed by:
    *   Clustering RLHF model outputs in the latent space using DBSCAN.
    *   Calculating the geometric centroid $\mathbf{c}_i$ for each cluster $C_i$.
    *   Finding the Euclidean distance $d_i$ from each centroid to the nearest SFT model output $\mathbf{s} \in S$.
    *   Summing the weighted distances: $\text{CSI} = \sum_{i=1}^n |C_i| \cdot d_i$.

### Key Formulas
The core objective function $J(\boldsymbol{\theta})$ is defined as:

$$
\max_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \max_{\boldsymbol{\theta}} I(\boldsymbol{S}; Y) - \beta I(\boldsymbol{X}; \boldsymbol{S}|Y)
$$

where $\beta$ is a trade-off parameter. The variational approximation used for training is:

$$
\max_{\{\phi,\psi\}} \mathbb{E}_{(x^w,x^l)\sim\mathcal{D}} [L_{\text{preference}} - \beta L_{\text{bottleneck}}]
$$

The preference loss $L_{\text{preference}}$ is based on the Bradley-Terry model:

$$
L_{\text{preference}} = \log \sigma \left( g_{\psi}(h_{\phi}(x^w,\epsilon^w)) - g_{\psi}(h_{\phi}(x^l,\epsilon^l)) \right)
$$

The bottleneck loss $L_{\text{bottleneck}}$ is the KL divergence between the latent distribution and a prior $r(S)$ (a centered isotropic multivariate Gaussian):

$$
L_{\text{bottleneck}} = \text{KL} \left[ p_{\phi}(S|x^w), r(S) \right] + \text{KL} \left[ p_{\phi}(S|x^l), r(S) \right]
$$

### Key Quantitative Results
InfoRM was tested across RM scales of 70M, 440M, 1.4B, and 7B.

*   **Simulation**: In experiments with a "gold RM," InfoRM prevented the decline of gold scores typically seen in Standard RMs during later RL stages, even with 25% label noise.
*   **Real-World Performance**: Evaluated by GPT-4, InfoRM outperformed several baselines. On the **Anthropic-Helpful** dataset, InfoRM achieved a win ratio of **54.5%**, tie ratio of **33.5%**, and lose ratio of **12.0%** against the Standard RM. It also outperformed Standard RM w/ KL, Ensemble RM, and WARM.
*   **Generalization**: On out-of-distribution (OOD) benchmarks, InfoRM showed superior accuracy. For example, on **Truthful QA (MC)**, InfoRM achieved **46.87%** accuracy compared to the Standard RM's **40.63%**.
*   **CSI Effectiveness**: The CSI indicator showed abrupt increases (e.g., between 600-700 training steps on Anthropic-Helpful) coinciding with the emergence of reward hacking, whereas InfoRM maintained consistently lower CSI values.

### Stated Limitations
1.  **Scalability**: The framework has only been evaluated on models up to 7B parameters; scaling to significantly larger state-of-the-art models remains unexplored.
2.  **Detection Latency**: The CSI monitoring mechanism requires inference on test datasets and exhibits some latency, necessitating the development of more lightweight, real-time metrics.
3.  **Evaluation Bias**: The authors noted that win rates computed by GPT-4 are influenced by the structure of the prompts used for evaluation.
