---
id: arxiv:2310.15815
type: paper
title: Rejection Sampling CoT (Yan et al., 2023)
url: https://arxiv.org/abs/2310.15815
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

**Core Problem**
Imitation learning (IL) operates under the assumption that offline demonstrations originate from expert or optimal policies. In real-world scenarios, however, datasets are frequently contaminated by noisy, suboptimal actions collected from non-expert demonstrators. These corrupt demonstrations misguide policy training, degrading robustness and applicability. Existing mitigation strategies often depend on unreliable human annotations or lack interpretability, creating a critical need for an unsupervised, self-motivated framework that automatically identifies and filters inferior demonstrations without auxiliary signals or reward supervision.

**Methodology and Recipe**
The proposed Self-Motivated Imitation LEarning (SMILE) framework addresses noisy demonstrations through a Policy-wise Diffusion mechanism combined with a self-paced filtering strategy. The training and inference recipe proceeds as follows:
1. **Forward Diffusion:** Expert actions are progressively corrupted by adding Gaussian noise across $T$ diffusion steps, simulating a gradual decline in policy expertise.
2. **Noise Approximation:** A neural network $\epsilon_\theta$ is trained to predict the injected noise at each timestep using a mean squared error objective, capturing the information that triggers expertise deterioration.
3. **One-Step Generation:** To mitigate the $\mathcal{O}(T)$ latency of standard multi-step denoising, the reverse process is approximated by a single-step policy $\pi_\phi$. This generator is trained to match the posterior mean of the forward diffusion process, reducing decision complexity to $\mathcal{O}(1)$ while preserving generative quality.
4. **Self-Motivated Filtering:** The trained noise approximator is repurposed to estimate the expertise gap between the current policy and the behavior policy that generated each demonstration. By computing a conditioned Q-function, the method predicts the diffusion step $t$ required to recover a clean policy from a noisy one. Demonstrations where the predicted step $t(\tau) = 0$ are retained, while those with $t(\tau) > 0$ are filtered out, ensuring the agent continuously imitates superior trajectories.

**Key Formulas**
The forward diffusion kernel adds Gaussian perturbations conditioned on state $s$:

$$
q(a_t|a_{t-1}, s) := \mathcal{N}(a_t; a_{t-1}, \beta_t^2\mathbf{I})
$$

Using the reparameterization trick, actions at step $t$ are generated directly from the initial action $a_0$ via:

$$
q(a_t|a_0, s) := \mathcal{N}(a_t; a_0, \sigma_t^2\mathbf{I}), \quad \sigma_t = \sqrt{\sum_{k=1}^t \beta_k^2}
$$

The noise predictor minimizes:

$$
L(\theta) = \mathbb{E}_{s,a_0,t,\epsilon}[\|\epsilon - \epsilon_\theta(s, a_t, t)\|_2^2]
$$

The one-step generator policy $\pi_\phi$ is optimized via:

$$
L(\phi) = \mathbb{E}_{(s,a_0)\sim\mathcal{D},a'_0\sim\pi_{\phi},t,\epsilon}[||\mu_t(a_t,a'_0) - \mu_t(a_t,a_t - \sigma_t\epsilon_{\theta}(s,a_t,t))||^2]
$$

Filtering relies on a conditioned Q-function derived from energy-based modeling:

$$
Q_{\theta,t}(\pi|\tilde{\pi}) = -\frac{1}{2\sigma_t^2} \mathbb{E}[\|a - (\tilde{a} - \sigma_t \epsilon_\theta(s, \tilde{a}, t))\|^2]
$$

The filtering decision for a trajectory $\tau$ is determined by:

$$
t(\tau) = \arg \max_{t' \in [0,T]} \frac{1}{|\tau|} \sum_i^{|\tau|} Q_{\theta,t'}(a^{(i)}|a^{\pi_\phi}, s^{(i)})
$$

**Quantitative Results**
Evaluated on continuous-control MuJoCo tasks (HalfCheetah, Walker2d, Hopper), SMILE consistently converges to expert-level performance with lower variance than baselines including Behavior Cloning, GAIL, ILEED, COIL, and RILCO. Ablation studies confirm the filtering module's necessity, as SMILE without filtering exhibits higher performance variance. Predicted diffusion steps correlate strongly with demonstration quality; for instance, in HalfCheetah, higher-return demonstration bins yield larger predicted diffusion steps as the agent improves. Compared to naive self-paced learning, SMILE achieves substantially higher returns (e.g., 3523.80 vs. 280.38 on Hopper). Additionally, the one-step generator reduces decision-making latency by approximately tenfold compared to naive multi-step reverse processes without compromising learning efficacy on most tasks.

**Stated Limitations**
The framework is currently restricted to continuous action spaces. The Gaussian-based diffusion process may inadvertently improve suboptimal actions when applied to finite or discrete action spaces. Furthermore, the method underutilizes the ordinal relationships among demonstrations, treating filtering as a binary expert-vs-non-expert decision rather than leveraging nuanced quality hierarchies. Concurrent training of the noise approximator and policy generator can also introduce optimization fluctuations, requiring careful hyperparameter tuning and exponential moving averages for stability.
