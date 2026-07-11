---
id: arxiv:2405.14734
type: paper
title: 'SimPO: Simple Preference Optimization with a Reference-Free Reward'
url: https://arxiv.org/abs/2405.14734
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

**Core Problem**
Direct Preference Optimization (DPO) implicitly defines rewards using the log-ratio between the policy model and a reference model. This formulation introduces two critical drawbacks: it requires storing and computing with a reference model during training, increasing memory and compute overhead, and it creates a fundamental mismatch between the reward optimized during training and the average log-likelihood metric used to guide generation at inference. Consequently, satisfying DPO’s reward ranking does not guarantee a higher average log-likelihood for winning responses, leading to inefficient preference learning and susceptibility to length bias exploitation.

**Method/Recipe Step by Step**
SimPO resolves these issues through a streamlined, reference-free optimization procedure:
1. **Construct Preference Data:** Assemble a dataset $\mathcal{D} = (x, y_w, y_l)$ containing prompts, winning responses, and losing responses.
2. **Define Length-Normalized Reward:** Replace DPO’s log-ratio reward with the average log probability of all tokens in a response, directly aligning the training objective with the generation metric.
3. **Introduce Target Margin:** Incorporate a target reward margin $\gamma > 0$ into the Bradley-Terry ranking objective to enforce a minimum separation between winning and losing rewards.
4. **Minimize Loss:** Directly optimize the policy model $\pi_\theta$ by minimizing the SimPO loss without KL regularization or a reference model.
5. **Tune Hyperparameters:** Set $\beta \in [2.0, 2.5]$ and $\gamma \in [0.3, 1.6]$ based on the specific model setup, utilizing a small learning rate and diverse preference data to maintain stability and prevent catastrophic forgetting.

**Key Formulas**
The average log-likelihood metric is defined as:

$$
p_\theta(y|x) = \frac{1}{|y|} \log \pi_\theta(y|x) = \frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i|x, y_{<i})
$$

This serves as the length-normalized reward:

$$
r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)
$$

Incorporating the target margin $\gamma$ into the Bradley-Terry model yields:

$$
p(y_w \succ y_l|x) = \sigma(r(x, y_w) - r(x, y_l) - \gamma)
$$

The complete SimPO objective is expressed as:

$$
\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma \right) \right]
$$

**Key Quantitative Results**
SimPO consistently outperforms DPO and its variants across multiple architectures. It improves length-controlled win rates by up to 6.4 points on AlpacaEval 2 and by up to 7.5 points on Arena-Hard. The top-performing Gemma-2-9B-it-SimPO model achieves a 72.4% length-controlled win rate on AlpacaEval 2, a 59.1% win rate on Arena-Hard, and ranks first among all sub-10B models on Chatbot Arena. SimPO effectively mitigates length exploitation, reducing the Spearman correlation between average log-likelihood and response length to 0.34, compared to 0.59 for DPO and 0.82 without length normalization. Computationally, SimPO reduces training runtime by approximately 20% and GPU memory usage by roughly 10% relative to standard DPO by eliminating reference model forward passes.

**Stated Limitations**
The authors explicitly identify several constraints. SimPO requires manual tuning of the target reward margin $\gamma$, and a rigorous theoretical analysis of its convergence remains absent. The objective is primarily designed to optimize helpfulness and does not explicitly incorporate safety or honesty constraints. Furthermore, preference optimization with SimPO consistently degrades performance on reasoning-heavy downstream tasks such as GSM8K. While empirical results show low KL divergence without explicit regularization, the authors caution that SimPO could theoretically suffer from reward hacking or model degeneration if hyperparameters are improperly set, necessitating careful implementation to maintain generalization.
