---
id: arxiv:2405.19107
type: paper
title: Offline Regularised Reinforcement Learning for Large Language Models Alignment
url: https://arxiv.org/html/2405.19107v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

The paper introduces Direct Reward Optimisation (DRO), a framework for aligning Large Language Models (LLMs) using single-trajectory datasets, which are more abundant and cheaper to collect than traditional pairwise preference data. DRO aims to overcome the limitations of existing methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimisation (DPO) that rely on scarce and expensive pairwise preference data.

**Core Problem:**
Current LLM alignment methods (RLHF, DPO) primarily rely on pairwise preference data, where a human annotator chooses a preferred response between two options for a given prompt. This data is expensive to collect, and its quality can degrade as LLMs improve, making distinctions between good responses harder. In contrast, single-trajectory data (prompt, response, scalar human feedback like thumbs-up/down) is naturally more abundant and easier to obtain from user logs. The core problem is to develop an effective and theoretically sound method for LLM alignment that can leverage this single-trajectory data.

**Method/Recipe Step by Step (DRO-V):**
DRO is presented as a framework, with DRO-V (Direct Reward Optimisation with Value) being a practical instantiation. DRO-V combines offline policy learning with value function learning.

1.  **Data Collection:** Utilize a single-trajectory dataset of the form $(x_i, y_i, r_i)_{i=1}^N$, where $x_i$ is a prompt, $y_i$ is a generated response, and $r_i$ is a scalar reward (e.g., 1 for thumbs-up, 0 for thumbs-down). This data is collected by an unknown behavior policy and remains static during optimization (offline setting).
2.  **Objective Formulation:** DRO aims to approximate the optimal Kullback-Leibler (KL) regularized policy $\pi^*$ by minimizing a mean-squared objective function. This objective simultaneously learns a policy $\pi_\theta$ and a value function $V_\phi$.
3.  **Loss Function:** The DRO loss for a pair of policy and value functions $(\pi, V)$ is defined as:

$$
\mathcal{L}_{\mathsf{D R O}}(\pi,V)\stackrel{\text{d e f}}{=}\frac{1}{2}\mathbb{E}_{x\sim\rho,y\sim\mu(\cdot|x)}\left[\left(r(x,y)-V(x)-\tau\text{l o g}\frac{\pi(y|x)}{\pi_{\mathsf{r e f}}(y|x)}\right)^{2}\right]
$$

    where $\rho$ is the prompt distribution, $\mu(\cdot|x)$ is the data distribution for responses given a prompt, $\pi_{\tt ref}$ is a reference policy (e.g., from supervised fine-tuning), and $\tau$ is a regularization scalar.
4.  **Practical Implementation (DRO-V Algorithm):**
    *   Initialize a parameterized policy $\pi_\theta$ and a parameterized value function $V_\phi$.
    *   For a specified number of training steps $K$:
        *   Sample a batch of $(x_i, y_i, r_i)$ triplets from the dataset.
        *   Compute the gradient updates for the value function parameters $\phi$ using:

$$
\nabla_{\varphi}\widehat{\mathcal{L}}(\theta,\varphi)=\frac{1}{B}\sum_{i=1}^{B}\left(V_{\varphi}(x_{i})-r(x_{i},y_{i})+\tau\text{l o g}\frac{\pi_{\theta}(y_{i}|x_{i})}{\pi_{\mathtt{r e f}}(y_{i}|x_{i})}\right)\nabla_{\varphi}V_{\varphi}(x_{i})
$$

        *   Compute the gradient updates for the policy parameters $\theta$ using:

$$
\nabla_{\theta}\widehat{\mathcal{L}}(\theta,\varphi)=-\tau\sum_{i=1}^{n}\left(\nabla_{\theta}\text{l o g}\pi_{\theta}(y_{i}|x_{i})\left(r(x_{i},y_{i})-V_{\varphi}(x_{i})\right)-\frac{\tau}{2}\nabla_{\theta}\left(\text{l o g}\frac{\pi_{\theta}(y_{i}|x_{i})}{\pi_{\mathsf{r a i i j}}(x_{i})}\right)^2\right)
$$

            The policy gradient is rescaled by $1/\tau$ in practice.
        *   Update $\theta$ and $\phi$ using an optimizer (e.g., AdaFactor).
5.  **Architectural Choices:** DRO-V uses two separate neural networks for $\pi_\theta$ and $V_\phi$, and computes multiple value numbers across the batch, as these choices were found to be empirically beneficial.

**Key Formulas in LaTeX:**

*   **Optimal KL-regularized policy:**

$$
\pi^{*}(x)\stackrel{\text{d e f}}{=}\text{a r g}\text{m a x}_{\pi}\quad\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x)}[r(x,y)-\tau\cdot\text{K L}(\pi(\cdot|x)\mid|\pi_{\mathtt{r e f}}(\cdot|x))]
$$

*   **Optimality condition relating reward, value, and policy:**

$$
r(x,y)-V(x)=\tau\log\frac{\pi(y|x)}{\pi_{\mathtt{r e f}}(y|x)}
$$

*   **DRO Loss Function:**

$$
\mathcal{L}_{\mathsf{D R O}}(\pi,V)\stackrel{\text{d e f}}{=}\frac{1}{2}\mathbb{E}_{x\sim\rho,y\sim\mu(\cdot|x)}\left[\left(r(x,y)-V(x)-\tau\text{l o g}\frac{\pi(y|x)}{\pi_{\mathsf{r e f}}(y|x)}\right)^{2}\right]
$$

*   **Gradient for Value Function Parameters ($\phi$):**

$$
\nabla_{\varphi}\widehat{\mathcal{L}}(\theta,\varphi)=\frac{1}{B}\sum_{i=1}^{B}\left(V_{\varphi}(x_{i})-r(x_{i},y_{i})+\tau\text{l o g}\frac{\pi_{\theta}(y_{i}|x_{i})}{\pi_{\mathtt{r e f}}(y_{i}|x_{i})}\right)\nabla_{\varphi}V_{\varphi}(x_{i})
$$

*   **Gradient for Policy Parameters ($\theta$):**

$$
\nabla_{\theta}\widehat{\mathcal{L}}(\theta,\varphi)=-\tau\sum_{i=1}^{n}\left(\nabla_{\theta}\text{l o g}\pi_{\theta}(y_{i}|x_{i})\left(r(x_{i},y_{i})-V_{\varphi}(x_{i})\right)-\frac{\tau}{2}\nabla_{\theta}\left(\text{l o g}\frac{\pi_{\theta}(y_{i}|x_{i})}{\pi_{\mathsf{r a i i j}}(x_{i})}\right)^2\right)
$$

*   **Approximation error bound for $\pi_V$ (policy optimized against approximate V):**

$$
\left|\text{l o g}\frac{\pi_{V}(y|x)}{\pi^{*}(y|x)}\right|\leq\frac{2}{\tau}\text{m a x}_{y}\left|(V^{\pi_{V}}(x)-V(x))\left(1-\frac{\pi_{V}(y|x)}{\mu(y|x)}\right)\right|
$$

**Key Quantitative Results and Numbers:**

*   **Outperformance against KTO:** DRO-V significantly outperforms Kahneman-Tversky Optimization (KTO) on the UltraFeedback dataset using T5 encoder-decoder models.
    *   **T5-L (770M parameters):**
        *   DRO-V vs SFT: 78.9% ± 0.3% winrate
        *   KTO vs SFT: 67.5% ± 0.7% winrate
        *   DRO-V vs KTO: 63.4% ± 1.0% winrate
    *   **T5-XL (3B parameters):**
        *   DRO-V vs SFT: 81.5% ± 1.0% winrate
        *   KTO vs SFT: 78.2% ± 0.7% winrate
        *   DRO-V vs KTO: 57.5% ± 0.8% winrate
*   **Hyperparameter Impact (T5-L):**
    *   **Learning Rate:** DRO-V performance remained stable within an order of magnitude change for learning rates. Jointly varying $\pi$ and $V$ learning rates: 73.7% (1e-5) to 78.9% (1e-4). Varying $V$ learning rate only: 76.8% (1e-5) to 79.1% (5e-4).
    *   **Regularization Parameter $\tau$:**
        *   DRO-V: Optimal at $\tau=1.0$ (78.9% winrate vs SFT), with $\tau=0.1$ yielding 70.5% and $\tau=5.0$ yielding 76.6%.
        *   KTO: Optimal at $\tau=5.0$ (67.5% winrate vs SFT), with $\tau=0.1$ yielding 63.5% and $\tau=1.0$ yielding 61.9%.
*   **Parameter Sharing (T5-L, 10,000 steps):**
    *   Using two separate networks for $\pi$ and $V$ (Double Net) significantly outperformed a single shared network (Single Net).
    *   Double Net Multiple Values: 76.6% winrate vs SFT.
    *   Double Net Single Value: 72.1% winrate vs SFT.
    *   Single Net Multiple Values: 55.5% winrate vs SFT.
    *   Single Net Single Value: 57.6% winrate vs SFT.
    *   Direct comparison of Double Net Multiple Values vs Double Net Single Value yielded 54.9% in favor of the multiple value version.

**Stated Limitations:**

*   **Empirical Study Scope:** The empirical study is limited in terms of the number of tasks and the scale of the models tested.
*   **Generalizability:** Further work is required to broadly establish the performance gains of DRO when considering the largest language models, especially as it leverages large amounts of user-generated data.
