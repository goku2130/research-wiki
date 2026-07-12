---
id: arxiv:2307.04964
type: paper
title: 'Secrets of RLHF in Large Language Models Part I: PPO'
url: https://arxiv.org/abs/2307.04964
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

The paper "Secrets of RLHF in Large Language Models Part I: PPO" by Zheng et al. (2023) investigates the challenges of stable training in Reinforcement Learning with Human Feedback (RLHF) for Large Language Models (LLMs), particularly focusing on the Proximal Policy Optimization (PPO) algorithm.

**Core Problem:**
The primary problem addressed is the instability and sensitivity to hyperparameters of the PPO algorithm when applied to RLHF for LLMs. This instability leads to "pattern collapse," where SFT models become over-optimized, exhibiting highly biased behavior and generating responses that "cheat" the reward model for anomalous high scores, rather than genuinely aligning with human preferences. This issue is exacerbated by the difficulty of reward design, environment interaction, agent training, and the high trial-and-error cost associated with LLMs.

**Method/Recipe Step by Step:**
The authors dissect the RLHF framework, re-evaluate PPO, and explore how its components impact policy agent training. They identify policy constraints as a key factor for effective PPO implementation and propose "PPO-max," an advanced version of PPO, to improve training stability.

The overall RLHF training process involves three stages:
1.  **Supervised Fine-Tuning (SFT):** The model learns human-like dialogues from human-annotated examples.
2.  **Reward Model (RM) Training:** A reward model is trained to compare and assign scalar reward values to responses based on human feedback.
    *   **RM Architecture:** Pre-trained transformer-based language models with the last unembedding layer removed and an additional linear layer.
    *   **RM Training Loss:** For a pair of preferred ($y_w$) and dispreferred ($y_l$) samples given prompt $x$:
        $\mathcal{L}(\psi) = -\log \left( \sigma(r(x, y_w) - r(x, y_l)) \right)$
        where $r(x,y)$ is the reward assigned by the RM.
3.  **Proximal Policy Optimization (PPO):** The policy model is updated based on feedback from the reward model to maximize cumulative reward.
    *   **Total Reward:** An additional KL divergence penalty is incorporated into the reward function to prevent the RL policy from deviating drastically from the SFT model:
        $r_{\mathrm{total}}(x,y) = r(x,y) - \eta \mathbf{KL}(\pi_{\phi}^{\mathrm{RL}}(y|x), \pi^{\mathrm{SFT}}(y|x))$
        where $\eta$ is the KL reward coefficient.
    *   **Policy Gradient Methods:** PPO is a policy gradient method that optimizes the policy $\pi(a|s, \theta)$ parameterized by $\theta$. The update rule is $\theta \leftarrow \theta + \alpha \nabla_{\theta}J(\theta)$.
    *   **Generalized Advantage Estimation (GAE):** GAE is used to estimate the advantage function $A(s_t, a_t)$, which balances bias and variance in reward estimation.
        *   TD-k return: $\hat{R}_{t}^{k} = r_{t} + \gamma r_{t+1} + \ldots + \gamma^{(k-1)}r_{t+k-1} + \gamma^{k}V(s_{t+k})$
        *   k-step advantage: $\hat{A}_{t}^{k} = \hat{R}_{t}^{k} - V(s_{t})$
        *   GAE($\gamma, \lambda$): $\hat{A}_{t}^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}$, where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.
    *   **PPO Variants:**
        *   **PPO-Penalty:** $\mathcal{L}_{\text{ppo-penalty}}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})}\hat{A}_{t}\right]-\beta\text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s_{t}),\pi_{\theta}(\cdot|s_{t}))$
        *   **PPO-Clip:** $\mathcal{L}_{\text{ppo-clip}}(\theta)=\hat{\mathbb{E}}_{t}\left[\text{min}\left(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})}\hat{A}_{t},\text{clip}\left(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})},1-\epsilon,1+\epsilon\right)\hat{A}_{t}\right)\right]$
        *   **PPO-ptx (with pretraining gradients):** $\mathcal{L}_{\mathrm{ppo-ptx}}(\theta)=\mathcal{L}_{\mathrm{ppo-clip}}(\theta)+\lambda_{\mathrm{ptx}}\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{pretrain}}}\left[\log(\pi_{\phi}^{\mathrm{RL}}(x))\right]$
    *   **PPO-max Setup:** Incorporates effective implementations including:
        *   Reward model initialization and pre-training for the critic model.
        *   Global gradient clipping.
        *   Small experience buffer size.
        *   Pre-train language model loss in policy optimization (to reduce alignment tax).
        *   Clipping of the value function loss.
    *   **Monitoring Metrics:** Perplexity, KL divergence between policy and reference models, and average length of generation responses are used to monitor training stability, as reward scores and training losses alone are insufficient.

**Key Formulas in LaTeX:**
*   Reward Model Training Loss: $\mathcal{L}(\psi) = -\log \left( \sigma(r(x, y_w) - r(x, y_l)) \right)$
*   Total Reward with KL Penalty: $r_{\mathrm{total}}(x,y) = r(x,y) - \eta \mathbf{KL}(\pi_{\phi}^{\mathrm{RL}}(y|x), \pi^{\mathrm{SFT}}(y|x))$
*   Policy Gradient Update Rule: $\theta \leftarrow \theta + \alpha \nabla_{\theta}J(\theta)$
*   GAE($\gamma, \lambda$): $\hat{A}_{t}^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}$
*   PPO-Penalty Objective: $\mathcal{L}_{\text{ppo-penalty}}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})}\hat{A}_{t}\right]-\beta\text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s_{t}),\pi_{\theta}(\cdot|s_{t}))$
*   PPO-Clip Objective: $\mathcal{L}_{\text{ppo-clip}}(\theta)=\hat{\mathbb{E}}_{t}\left[\text{min}\left(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})}\hat{A}_{t},\text{clip}\left(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})},1-\epsilon,1+\epsilon\right)\hat{A}_{t}\right)\right]$
*   PPO-ptx Objective: $\mathcal{L}_{\mathrm{ppo-ptx}}(\theta)=\mathcal{L}_{\mathrm{ppo-clip}}(\theta)+\lambda_{\mathrm{ptx}}\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{pretrain}}}\left[\log(\pi_{\phi}^{\mathrm{RL}}(x))\right]$
*   Reward Normalization and Clipping: $\tilde{r}(x,y)=\text{clip}\left(\frac{r_{n}(x,y)-\overline{r(x,y)}}{\sigma(r(x,y))},-\delta,\delta\right)$
*   Token Level KL-Penalty: $r_{\text{total}}(x,y_{i})=r(x,y_{i})-\eta\text{K L}(\pi_{\theta}^{\text{R L}}(y_{i}|x),\pi^{\text{S F T}}(y_{i}|x))$

**Key Quantitative Results and Numbers:**
*   **Reward Model Accuracy:** RM trained on Chinese dataset showed higher accuracy than English due to greater dissimilarity between responses in Chinese pairs. Accuracy significantly slowed improvement after 200 steps (approx. 0.2 epochs).
*   **PPO-max Performance:** Evaluated on 7B and 13B SFT models, demonstrating comparable alignment performance with ChatGPT.
*   **Human Evaluation:** RLHF-trained models showed strong preference over SFT models.
    *   English dataset: RLHF model achieved 62% rating on Harmless held-out dataset vs. 5% for SFT. Helpful dataset showed 44% for RLHF vs. 30% for SFT.
    *   Chinese dataset: RLHF model enhanced performance on both Helpful and Harmless datasets.
*   **GPT-4 as Judge Evaluation:** Results mirrored human evaluation, with RLHF models showing significant advantages, especially in harmful prompts.
*   **Mitigating Defeats to ChatGPT:** RLHF models significantly reduced the defeat rate against ChatGPT (gpt-3.5-turbo-0613).
    *   English: Defeat rate decreased from 45% (SFT) to 24% (RLHF).
    *   Chinese: Defeat rate decreased from 37% (SFT) to 29% (RLHF).
*   **Language Understanding (C-eval):** PPO-ptx effectively mitigated the decline in NLU capabilities caused by PPO.
*   **Hyperparameters:**
    *   Learning rate for policy model: 5e-7.
    *   Learning rate for critic model: 1.65e-6.
    *   Warmup over first 10% steps.
    *   Batch size for environment sampling: 128.
    *   Batch size for policy/critic training: 32.
    *   GAE $\lambda = 0.9$.
    *   Nucleus sampling: p=0.9, temperature $\tau=0.8$.
    *   Repetition penalty: $\beta=1.1$.
    *   Maximum token length: 2048.

**Stated Limitations:**
*   **Performance Indicator:** The focus during the PPO phase was more on achieving stability rather than enhancing final performance. Stability is crucial but does not guarantee improved outcomes. Reward scores cannot reliably predict RLHF performance during training, indicating a need for more suitable performance indicators.
*   **Scaling Law:** The study primarily focused on a 7-billion-parameter model, and the impact of model size and data scale on RLHF performance has not yet been investigated.
