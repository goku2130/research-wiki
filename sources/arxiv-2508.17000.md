---
id: arxiv:2508.17000
type: paper
title: 'KL-Regularised Q-Learning: A Token-level Action-Value perspective on Online
  RLHF'
url: https://arxiv.org/html/2508.17000v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

The paper "KL-Regularised Q-Learning: A Token-level Action-Value perspective on Online RLHF" introduces KL-regularised Q-Learning (KLQ), an action-value method for Language Model Reinforcement Learning from Human Feedback (LM-RLHF) that operates at the token level. It addresses the heuristic nature of Proximal Policy Optimisation (PPO), a widely used policy gradient algorithm in LM-RLHF, particularly its ad-hoc handling of KL-divergence constraints.

**Core Problem:**
The core problem is the heuristic motivation and ad-hoc KL-divergence constraint handling of PPO in LM-RLHF, despite its empirical success. The authors aim to develop a theoretically grounded online action-value method for token-level LM-RLHF that can perform comparably or better than PPO.

**Method/Recipe Step-by-Step:**

1.  **KL-Regularised RL Framework:** The method operates within a KL-regularised RL setting, where the objective is to maximize the expected discounted, KL-regularised return. A reference policy $\pi_b$ (typically the SFT policy) and a temperature parameter $\tau > 0$ are used to control the regularization strength.
2.  **Token-level View:** KLQ adopts a token-level view of the LM-RLHF problem, treating partial completions as states $s_t = (x, y_{1:t})$ and next tokens as actions $a_t = y_{t+1}$. Rewards are administered at each time step as weighted negative token-wise KL-divergence, $-\tau D(\pi_\theta||\pi_b)(s_{t+1})$, with a final reward $R_\phi(x,y)$ from the reward model at the End-Of-Sequence (EOS) token.
3.  **$\lambda$-Return Value Estimator:** KLQ uses $\lambda$-returns to construct regression targets for the action-value function. This addresses the sparsity of the reward signal in LM-RLHF, as the reward model evaluates full completions only at the end of a trajectory. The $\lambda$-return $G_t^{\lambda,\alpha}[Q_\theta]$ is a weighted combination of n-step returns, with $\lambda$ acting as a truncation rate and $\alpha$ scaling the error term.
4.  **Action-Value Decomposition:** To enable initialization from a pre-trained policy (SFT model), KLQ leverages an analytic correspondence between optimal policies and action-values in the KL-regularised setting. The action-value function $Q_\theta(s,a)$ is parameterized in terms of the policy $\pi_\theta(a|s)$ and a Boltzmann state-value function $V_\theta(s)$.
5.  **Training:**
    *   **Experience Gathering:** Rollouts are generated from the current policy $\pi(a|s; \theta)$ by sampling initial prompts and auto-regressively generating completions. Rewards are obtained from the reward model, with $r_T^{(i)} = R_i$ and $r_t^{(i)} = 0$ for $t < T$.
    *   **Regression Target Computation:** For each trajectory, $\lambda$-return regression targets $\hat{G}_t$ are computed. This involves calculating the action value $Q(s_t, a_t; \theta)$, the TD-error $\delta_t$, and an error term $\Delta_t$ recursively.
    *   **Action-Value Function Training:** The action-value function is trained by minimizing an $\ell^2$ loss between the parameterized $Q_\theta(s,a)$ and the computed $\hat{G}_t$. This loss is optimized using a gradient-based optimizer over multiple epochs per batch.

**Key Formulas (in LaTeX):**

*   **KL-Regularised Return Objective:**
    $G_t=\sum_{k=t}^{T-1}\gamma^{k-t}\left(r_{k+1}-\tau\gamma D(\pi_{\theta}||\pi_{b})\left(s_{k+1}\right)\right)$ (1)
*   **KL-Divergence:**
    $D(\pi_{\theta}||\pi_{b})(s): = \sum_{a}\pi_{\theta}(a|s)\text{l o g}\left(\frac{\pi_{\theta}(a|s)}{\pi_{b}(a|s)}\right)$ (2)
*   **Modified LM-RLHF Objective (Reward Model):**
    $\bar{R}_{\phi}(x,y)=R_{\phi}(x,y)-\tau\log{\left(\frac{\pi_{\theta}(y|x)}{\pi_{b}(y|x)}\right)}$ (4)
*   **$\lambda$-Return Regression Target:**
    $G_{t}^{\lambda,\alpha}[Q_{\theta}]=\alpha\sum_{k=t}^{T-1}(\lambda\gamma)^{k-t}\delta_{k}+Q_{\theta}(s_{t},a_{t})$ (7)
*   **TD-Error for KLQ:**
    $\delta_{t}=r_{t+1}+\gamma V_{\theta}(s_{t+1})-Q_{\theta}(s_{t},a_{t})$ (14)
*   **Action-Value Function Parameterization:**
    $Q_{\theta}(s,a)=\tau\log\left(\frac{\pi_{\theta}(a|s)}{\pi_{b}(a|s)}\right)+V_{\theta}(s)$ (13)
*   **KLQ Loss Function:**
    $\mathcal{L}(\theta)=\mathbb{E}\left[\left(Q_{\theta}(s,a)-\hat{G}\right)^{2}\right]$ (6)
*   **Equivalence Parameter $\beta$ (KLQ to PPO-penalty):**
    $\beta=\tau\left(\frac{1-\alpha}{\alpha}\right)$ (18)

**Key Quantitative Results and Numbers:**

*   **Performance on LM-RLHF Objective:** KLQ performs "on-par" with PPO in optimizing the LM-RLHF objective, achieving similar final rewards on both TL;DR and Anthropic-HH datasets (Figure 1).
*   **Compute Cost:** KLQ has an "equal per-update compute cost" compared to PPO, with both algorithms completing a similar number of training episodes in approximately the same wall-clock time (Table 3).
*   **LLM-as-a-Judge Win-Rate:** KLQ achieves a "consistently higher win-rate against PPO" in LLM-as-a-judge evaluations using GPT-4o mini. This was observed across various KL-penalty coefficients on both TL;DR and Anthropic-HH datasets (Figure 3). For example, on TL;DR, KLQ shows particularly strong performance at higher KL-penalty coefficients.
*   **Hyperparameters:** Default hyperparameters used include $\gamma = 1$, $\tau = 0.05$, $\lambda = 0.95$, initial learning rate of $1.41 \times 10^{-5}$, 4 training epochs per batch, 48 rollouts per batch per device, and a total of 192 rollouts per batch on 4 GPUs. Each training run consisted of 75,000 episodes, taking approximately 5 hours.

**Stated Limitations:**

*   **Preliminary Experiments:** The experiments are relatively preliminary compared to recent works in LM-RLHF.
*   **Computational Resources:** Limited computational resources (5 hours on 4 A100 GPUs per run) prevented longer experiments, large hyper-parameter sweeps, or many repeats to reduce noise.
*   **Model Quality:** Existing SFT and reward models from Huggingface Hub were used, making it difficult to evaluate or guarantee their quality and properties. A more robust evaluation would involve training custom SFT and reward models.
*   **Task Scope:** The methodology was not evaluated on multi-step reasoning tasks, which are an exciting application area for language model RL.
*   **Theoretical Equivalence Aspects:** Certain aspects of the theoretical equivalence between KLQ and PPO (Section 4) require further investigation.
*   **Advantage Normalization:** It is unclear how to establish an analogue of advantage whitening (used in TRL's PPO implementation) for KLQ due to its combined $\pi,V$ objective, potentially limiting empirical improvements.
