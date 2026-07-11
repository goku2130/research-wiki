---
id: arxiv:2310.05199
type: paper
title: 'Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from
  Human Feedback'
url: https://arxiv.org/abs/2310.05199
retrieved: '2026-07-11'
maturity: comprehensive
topic: length-and-format-bias
---

# Summary: Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback

### Core Problem
In Reinforcement Learning from Human Feedback (RLHF), reward models (RM) often develop a "length bias," where they erroneously associate longer response lengths with higher quality. This creates a shortcut for the model to maximize rewards without actually increasing the helpfulness or accuracy of the content. This phenomenon, a form of "reward gaming" or "reward hacking," leads to verbose, low-quality outputs and can cause model degradation during the Proximal Policy Optimization (PPO) phase.

### Method
The authors propose a **Product-of-Experts (PoE)** framework to decouple the learning of human intent from the influence of sequence length. The method consists of the following steps:

1.  **Dual-Expert Architecture**: The reward modeling phase is split into two distinct experts:
    *   **Main Reward Expert ($r_{\phi}$)**: A larger model (e.g., 7B LLaMA) with a standard learning rate, designed to capture complex human intents.
    *   **Bias-only Expert ($r_{\psi}$)**: A smaller model (e.g., 560M BLOOMZ) with a higher learning rate (typically $3\times$ that of the main expert). Smaller models with higher learning rates are used because they tend to capture coarser, simpler patterns like sequence length.
2.  **Semantic Disruption (Noise Injection)**: To prevent the bias-only expert from learning semantic information, Gaussian noise $N$ is added to the token embeddings of the input $X$:

$$
X' = X + N
$$

3.  **Joint Training**: The two experts are jointly optimized to predict human preferences. The combined reward $\hat{r}(x,y)$ is formulated as:

$$
\hat{r}(x,y) = \text{Softmax}(\log(r_{\phi}(x,y)) + \log(r_{\psi}(x,y)))
$$

    Alternatively expressed as $\hat{r}(x,y) \propto r_{\phi}(x,y) \circ r_{\psi}(x,y)$. The training objective maximizes the likelihood:

$$
-\mathbb{E}_{(x,y)\sim\mathcal{D}}[\log(\sigma(\hat{r}(x,y_{i})-\hat{r}(x,y_{1-i})))]
$$

4.  **PPO Inference**: During the RL optimization stage, the bias-only expert is discarded. Only the main expert $r_{\phi}$ is used to provide the reward signal for the policy $\pi_{\phi}$, which is optimized using the objective:

$$
\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[r_{\theta}(x,y)-\beta\text{log}\left(\pi_{\phi}^{\text{RL}}(y\mid x)/\pi^{\text{SFT}}(y\mid x)\right)\right]
$$

    where $\beta$ is the KL-penalty coefficient.

### Key Quantitative Results
The method was evaluated on the HH-RLHF and rm-static datasets using various models (BLOOMZ, LLaMA, Alpaca).

*   **Win Rates**: On the HH-RLHF dataset using the Alpaca model, the proposed method achieved win rates against baselines as follows:
    *   **Vs. SFT**: 54.23%
    *   **Vs. PPO (Vanilla)**: 57.47%
    *   **Vs. Answer$_{Chosen}$**: 67.69% (evaluated via humans, AlpacaFarm, and GPT-4).
*   **Length Reduction**: The method significantly reduced average output length while maintaining or increasing reward scores. For the Alpaca model on HH-RLHF, the average length dropped from **689 tokens** (Vanilla-PPO) to **586 tokens** (Ours-PPO). On rm-static, it dropped from **721** to **617 tokens**.
*   **Perplexity (PPL)**: On HH-RLHF, the proposed method achieved a lower (better) PPL of **8.82** compared to **9.67** for Vanilla-PPO-Alpaca.
*   **Correlation**: Spearman/Pearson coefficients between reward and length were reduced. For BLOOMZ, the coefficients dropped from **0.3865/0.3932** (Vanilla) to **0.2354/0.2990** (PoE-RM).
*   **Summarization (TL;DR)**: The method achieved a **58% win rate** with an output length of **51 tokens**, compared to **220 tokens** when the KL penalty ($\beta$) was 0.

### Limitations
The authors note that improper reward model overoptimization can lead to "negative optimization," resulting in the policy model collapsing and engaging in "self-talking." This suggests a potential loss of instruction-following ability in certain edge cases.
