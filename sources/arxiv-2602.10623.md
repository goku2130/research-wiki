---
id: arxiv:2602.10623
type: paper
title: Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling
url: https://arxiv.org/abs/2602.10623
retrieved: '2026-07-11'
maturity: comprehensive
topic: length-and-format-bias
---

# Bayesian Non-negative Reward Modeling (BNRM)

### Core Problem
Reward models (RMs) in Reinforcement Learning from Human Feedback (RLHF) are susceptible to **reward hacking** (or reward over-optimization). This occurs when a policy exploits spurious correlations—such as response length, stylistic artifacts, or formatting—encoded in the proxy RM rather than the true human objective. These failures stem from noisy human annotations and the tendency of deep neural networks to rely on "shortcut" features, leading to poor generalization under distribution shifts.

### Method
The **Bayesian Non-negative Reward Model (BNRM)** mitigates reward hacking by integrating non-negative factor analysis (NFA) into the Bradley-Terry (BT) preference model. It replaces deterministic reward functions with a stochastic generative process that enforces sparsity and non-negativity to disentangle semantic intent from nuisance factors.

#### Step-by-Step Recipe:
1.  **Generative Process**: The model represents rewards as a product of two non-negative latent variables:
    *   **Local Latent Factors ($\theta$):** Instance-specific variables that induce disentangled representations of semantic preference.
    *   **Global Reward Dictionary ($\Phi$):** A shared basis across all data points that acts as a population-level regularizer to suppress systematic biases.
2.  **Prior Specification**: To enforce sparsity and non-negativity, Gamma priors are placed on both $\Phi$ and $\theta$:

$$
\Phi \sim \text{Gamma}(\gamma_0, \delta_0), \quad \theta \sim \text{Gamma}(\alpha_0, \beta_0)
$$

3.  **Reward Calculation**: The scalar reward for a prompt-response pair $(x, y)$ is computed as:

$$
r(x, y) = \theta^\top \Phi
$$

4.  **Preference Modeling**: The probability of preference between two responses $y_1$ and $y_2$ is modeled via the Bradley-Terry likelihood:

$$
p(y_1 \succ y_2 \mid r_1, r_2) = \sigma(r_1 - r_2)
$$

5.  **Amortized Variational Inference**: To scale to LLMs, BNRM uses the LLM backbone as an inference network (encoder) to approximate the posterior distributions $q(\theta \mid x, y)$ and $q(\Phi)$. These are parameterized using a reparameterizable **Weibull distribution** to model sparse, positive random variables.
6.  **Training Objective**: The model is trained by maximizing the Evidence Lower Bound (ELBO):

$$
\max \mathcal{L}(\mathcal{D}) = \mathbb{E}_{q(\theta|x,y)q(\Phi)}[\log p(\mathcal{D} \mid \theta, \Phi)] - \eta KL(q(\theta|x,y) \| p(\theta)) - \eta KL(q(\Phi) \| p(\Phi))
$$

    where $\eta$ is a trade-off coefficient balancing the likelihood and KL divergence.

### Key Quantitative Results
*   **Robustness and Generalization**: Using a Gemma-2B-it backbone with 40K Unified-Feedback samples, BT-BNRM improved accuracy over the BT baseline by **5.4%** (Unified-Feedback), **13.3%** (HHH Alignment), **6.1%** (MT-Bench), and **8.0%** (RewardBench).
*   **High-Scale Performance**: When refining the Skywork-Reward-Llama-3.1-8B model, BNRM achieved an overall RewardBench score of **93.6**, outperforming the base model (93.1).
*   **RLHF Alignment**: PPO-tuned policies using BNRM achieved higher benchmark accuracy (e.g., **74.98%** for Llama-3.1-8B-Instruct). On Arena-Hard-v0.1, the BNRM-aligned model achieved a **50% win rate** and **28% tie rate** against the base model.
*   **Debiasing**: BNRM significantly reduced length bias; the Pearson correlation between response length and reward score dropped from **0.488** (BT) to **0.123** (BNRM).
*   **Efficiency**: The computational overhead is negligible; full-parameter fine-tuning of Llama-3.1-8B-Instruct saw a training time increase of only **1.3%** (from 11.70h to 11.86h).
*   **Noise Tolerance**: At a 40% label noise rate, BNRM improved over the BT baseline by up to **16.7%**.

### Stated Limitations
*   **Hyperparameter Sensitivity**: The model introduces the KL coefficient $\eta$, which requires tuning to balance prior regularization and posterior adaptation (optimal $\eta = 10^{-5}$ in experiments).
*   **Scope of Evaluation**: While effective in current benchmarks, the authors note that open-ended RLHF settings may expose more diverse and adaptive forms of reward hacking, particularly in multi-turn, tool-use, or long-horizon alignment scenarios.
