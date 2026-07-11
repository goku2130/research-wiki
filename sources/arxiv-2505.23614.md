---
id: arxiv:2505.23614
type: paper
title: Inference-time Scaling of Diffusion Models through Classical Search
url: https://arxiv.org/abs/2505.23614
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Inference-time Scaling of Diffusion Models through Classical Search

### Core Problem
The authors address the challenge of **inference-time control** in diffusion models, where generated outputs must be adapted to meet specific test-time objectives (e.g., physical validity in robotics or human intent in image generation) without retraining the model. The core problem is navigating the vast generative space to find high-quality samples that optimize a verifier function $f(x_0)$ while remaining on the generative manifold.

### Method
The proposed framework orchestrates a combination of **global search** (to identify high-score modes within the base distribution) and **local search** (to optimize samples beyond the base model's capabilities).

#### 1. Global Search
The sampling process is viewed as traversing a search tree.
*   **Unified BFS-style Linear Search:** Denoises $N$ particles in parallel. At each noise level $t$, particles are scored using $\widehat{f}(x_t^k)$ (options include *Current*, *Difference*, or *Max* rewards along the path). Compute is reallocated via resampling (Multinomial or Srinivasan sampling process [SSP]) and tempering schedules (Constant, Increase, or Infinite).
*   **Adaptive DFS:** Iteratively denoises a single particle. If the verifier score falls below a threshold $\delta_t$, the algorithm performs **adaptive backtracking** by reintroducing noise (with a depth $\Delta_T \geq T/4$) to explore diverse modes.

#### 2. Local Search
To sample from the compositional distribution $\tilde{p}_{0}(x_{0})\propto p_{0}(x_{0})f(x_{0})^{\lambda}$, the authors employ **annealed Langevin MCMC**.
*   **Mechanism:** The search follows the gradient flow of KL-divergence. The update is implemented via the unadjusted Langevin algorithm (ULA):

$$
\boldsymbol{x}^{i+1}=\boldsymbol{x}^{i}+\eta\nabla_{\boldsymbol{x}^{i}}\log\nu(\boldsymbol{x}^{i})+\sqrt{2\eta}\boldsymbol{\epsilon}^{i}
$$

*   **Theoretical Unification:** The authors prove (Proposition 1) that in the continuous limit ($T \to \infty$), training-free guidance with recurrence is equivalent to running Langevin MCMC on a series of annealed distributions.

#### 3. Double-Verifier Strategy
To mitigate "reward hacking" (where gradient guidance produces out-of-distribution samples that trick the verifier), the authors propose using separate verifiers for local and global search.

### Key Formulas
*   **Denoised Estimate:** Used for intermediate scoring during global search:

$$
x_{0|t}=\mathbb{E}[x_{0}\big|x_{t}]=\frac{x_{t}-\sigma_{t}\epsilon_{\theta}(x_{t},t)}{\alpha_{t}}
$$

*   **Training-Free Guidance Update:**

$$
\tilde{\boldsymbol{x}}_t = \boldsymbol{x}_t + \Delta_t, \quad \Delta_t = \rho_t \nabla_{\boldsymbol{x}_t} \log f(\boldsymbol{x}_{0|t}) + \mu_t \alpha_t \nabla_{\boldsymbol{x}_{0|t}} \log f(\boldsymbol{x}_{0|t})
$$

### Quantitative Results
*   **Image Generation (BFS):** On SDv1.5 ($N=8$), the improved BFS (**Increase**, **Max**, **SSP**) achieved a score of $1.087 \pm 0.031$, outperforming DAS ($1.052 \pm 0.033$) and SVDD ($0.775 \pm 0.087$). On SDXL ($N=8$), it achieved $1.291 \pm 0.018$ vs. DAS's $1.265 \pm 0.019$.
*   **Efficiency (DFS):** DFS outperformed Best-of-N and BFS, requiring up to $2\times$ less computational cost.
*   **Offline RL (D4RL):** The Test-Time Search (TTS) method achieved an average locomotion score of $86.1$, comparable to training-based baselines like D-QL ($86.3$) and QGPO ($86.6$), and significantly better than DAS ($80.2$) and TFG ($82.1$).
*   **Policy Distillation:** Fine-tuning a model on TTS-generated actions outperformed SOTA online RL (DPPO) in Halfcheetah ($51.6 \pm 0.7$ vs $47.8$).
*   **Double-Verifier (ImageNet):** BFS-double ($N=8$) improved FID to $118.2$ and accuracy to $55.9\%$, compared to BFS-single's FID of $133.3$ and accuracy of $46.5\%$.

### Limitations
The authors state that inference-time scaling still requires the tuning of additional hyperparameters. They suggest that future work could employ heuristics like evolutionary search for hyperparameter optimization.
