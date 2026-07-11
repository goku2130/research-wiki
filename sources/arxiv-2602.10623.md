---
id: arxiv:2602.10623
type: paper
title: Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling
url: https://arxiv.org/html/2602.10623v2
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-modeling
---

The paper "Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling" introduces the Bayesian Non-Negative Reward Model (BNRM) to address reward hacking in Reinforcement Learning from Human Feedback (RLHF).

**Core Problem:**
Reward models (RMs) in RLHF, learned from human preferences, are susceptible to reward hacking. This occurs when policies exploit spurious correlations (e.g., response length, stylistic artifacts) in the RM, leading to behavior that scores high but deviates from true human objectives. This misgeneralization stems from noisy human annotations and the tendency of deep neural networks to learn shortcut features. Existing methods like ensembles incur high computational costs, information-theoretic methods implicitly suppress features, and supervised interventions for specific biases generalize poorly.

**Method/Recipe:**
BNRM integrates non-negative factor analysis (NFA) into the Bradley-Terry (BT) preference model, formulating reward learning as a stochastic generative process over sparse, non-negative latent factors.

1.  **Generative Process Formulation:**
    *   **Modeling Aleatoric Uncertainty:** Replaces deterministic latent representation $z$ with a stochastic latent variable $\theta$ to capture inherent randomness in human preferences. The preference probability is marginalized over $\theta$:

$$
p(y_1 \succ y_2 \thinspace|\thinspace x, y_1, y_2) = \int_{\theta_1, \theta_2} P(y_1 \succ y_2 \thinspace|\thinspace \theta_1, \theta_2) p(\theta_1 \thinspace|\thinspace y_1, x) \thinspace P(\theta_2 \thinspace|\thinspace y_2, x) \thinspace d\theta_1 \thinspace d\theta_2
$$

    *   **Modeling Epistemic Uncertainty:** Treats the final layer weights (denoted as $\Phi$) as a global stochastic variable, accounting for model uncertainty about global reward factors:

$$
p(y_1 \succ y_2 \thinspace|\thinspace x, y_1, y_2) = \int_{\theta_1, \theta_2, \Phi} p(y_1 \succ y_2 \thinspace|\thinspace \theta_1, \theta_2, \Phi) p(\theta_1 \thinspace|\thinspace y_1, x) \thinspace p(\theta_2 \thinspace|\thinspace y_2, x) \thinspace p(\Phi) \thinspace d\theta_1 \thinspace d\theta_2 \thinspace d\Phi
$$

    *   **Latent Variables:**
        *   **Instance-Specific Latent Factors ($\theta$):** Non-negative, sparse latent variables for each prompt-response pair $(x,y)$, promoting disentangled reward representations.
        *   **Global Reward Dictionary ($\Phi$):** A global dictionary of non-negative reward factors, shared across all data points, acting as a population-level regularizer to suppress spurious correlations.
    *   **Priors:** Gamma priors are placed on both $\Phi$ and $\theta$ to enforce non-negativity and sparsity:

$$
\Phi \sim \text{Gamma}(\gamma_0, \delta_0), \qquad \theta \sim \text{Gamma}(\alpha_0, \beta_0)
$$

    *   **Reward Calculation:** The scalar reward for a response $(x,y)$ is generated as:

$$
r(x,y) = \theta^{\top}\Phi
$$

    *   **Preference Likelihood:** Human preference for $(y_1, y_2)$ with rewards $(r_1, r_2)$ is modeled via a Bradley-Terry likelihood:

$$
p(y_1 \succ y_2 \thinspace|\thinspace r_1, r_2) = \sigma(r_1 - r_2)
$$

        where $\sigma(\cdot)$ is the logistic sigmoid function.

2.  **Variational Inference and Training Objective:**
    *   Since exact posteriors are intractable, variational inference (VI) is used with tractable variational distributions $q(\theta|x,y)$ and $q(\Phi)$.
    *   The LLM backbone $f$ is repurposed as an inference network (encoder) for amortized inference of local latent variables.
    *   **Weibull Distribution for Posteriors:** A Weibull distribution is adopted for its reparameterization and ability to model sparse, positive random variables:

$$
q(\theta \mid x, y) = \text{Weibull}(k, \lambda)
$$

$$
(k, \lambda) = \text{Activation}(z W_{\mathrm{vi}})
$$

        where $W_{\mathrm{vi}} \in \mathbb{R}^{d_{\mathrm{model}} \times 2K}$, $k, \lambda \in \mathbb{R}_{+}^{K}$ are shape and scale parameters. Softplus activation is used for $k$ and ReLU for $\lambda$.
    *   The variational distribution for $\Phi$, $q(\Phi)$, is similarly parameterized as a Weibull distribution.
    *   **Training Objective (ELBO Maximization):** The model is trained by maximizing the Evidence Lower Bound (ELBO):

$$
\max_{W_{\mathrm{llm}}, W_{\mathrm{vi}}, \Phi} \mathcal{L}(\mathcal{D}) = \max_{W_{\mathrm{llm}}, W_{\mathrm{vi}}, \Phi} \left[ \mathbb{E}_{q(\theta|x,y)q(\Phi)} \left[ \log p(\mathcal{D}|\theta, \Phi) \right] - \eta \mathrm{KL}(q(\theta|x,y)\|p(\theta)) - \eta \mathrm{KL}(q(\Phi)\|p(\Phi)) \right]
$$

        where $\eta$ is a trade-off coefficient balancing likelihood and KL divergence.

**Key Quantitative Results and Numbers:**

*   **ID and OOD Evaluation (40K UF training examples, Gemma 2B it):**
    *   BT-BNRM improved accuracies over BT baseline:
        *   Unified-Feedback: 74.2% (↑5.4%)
        *   HHH Alignment: 83.6% (↑13.3%)
        *   MT-Bench: 75.2% (↑6.1%)
    *   GRM-BNRM improved accuracies over GRM-SFT baseline:
        *   Unified-Feedback: 74.1% (↑2.6%)
        *   HHH Alignment: 82.4% (↑3.7%)
        *   MT-Bench: 75.1% (↑2.1%)
*   **RewardBench Performance (Full Parameter Training on Skywork-Preference-v0.2, Llama-3.1-8B-Instruct):**
    *   BNBT-Reward-Llama-3.1-8B achieved an overall score of **93.6%** (↑0.5% over Skywork-Reward-Llama-3.1-8B).
    *   Chat-Hard subset: **89.7%** (↑1.3%)
    *   Safety subset: **92.6%** (↓0.1%)
*   **Low-Resource Settings (Gemma-2B-it, RewardBench):** BNRM trained on 1K examples matched the performance of BT trained on 20K examples.
*   **Label Noise (Gemma-2B-it, 40K samples, 40% noise):** BNRM improved BT by up to 16.7% and rivaled BT trained with only 10-20% noise.
*   **RLHF Evaluation (PPO-tuned Llama-3.1-8B-Instruct):**
    *   BNRM-fine-tuned policies achieved average accuracy of **74.98%** (↑12.15% over Base Llama3.1-8B-Instruct) and **62.25%** (↑6.57% over OpenRLHF-Llama3-8B-SFT).
*   **Arena-Hard-v0.1 (GPT-4.1 as judge, BNRM-aligned Llama-3.1-8B-Instruct vs. Base Llama-3.1-8B-Instruct):**
    *   Win rate: **50%**
    *   Tie rate: **28%**
    *   Outperformed baseline (22% win rate).
*   **Human Assessment (BNRM-aligned Llama-3.1-8B-Instruct vs. Base Llama-3.1-8B-Instruct):**
    *   Average human win rate: **51%**
    *   Average human tie rate: **18%**
*   **Reward Debiasing (RM-Bench Hard subset, Pearson correlation between response length and reward score):**
    *   Vanilla BT model: r = 0.488
    *   BNRM: r = 0.123 (substantially lower)
*   **Training Time Overhead:**
    *   Gemma-2B-it: 7.7% increase (from 3.88h to 4.18h)
    *   Llama-3.1-8B-Instruct: 1.3% increase (from 11.70h to 11.86h)
*   **Convergence:** BNRM achieved 71.75% validation accuracy within 0.25 epochs, surpassing standard BT and GRM variants even after 1.5 epochs.

**Stated Limitations:**
1.  **Hyperparameter Tuning:** BNRM introduces an additional hyperparameter, the KL coefficient $\eta$, which controls the strength of Bayesian regularization. While robust, achieving optimal performance may require tuning $\eta$ to balance prior regularization and posterior adaptation. In experiments, $\eta=10^{-5}$ provided a favorable balance.
2.  **Scope of Reward Hacking Mitigation:** Although BNRM shows evidence of mitigating reward hacking in OOD, noisy-label, Arena-Hard, and human evaluations, open-ended RLHF settings may expose more diverse and adaptive forms of reward hacking. Evaluating BNRM under broader multi-turn, tool-use, and long-horizon alignment scenarios is an important direction for future work.
