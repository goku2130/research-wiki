---
id: arxiv:2404.10719
type: paper
title: Is DPO Superior to PPO for LLM Alignment? A Theoretical and Empirical Study
url: https://arxiv.org/abs/2404.10719
retrieved: '2026-07-12'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

# Summary: Is DPO Superior to PPO for LLM Alignment?

### Core Problem
The authors investigate the performance discrepancy between reward-based Reinforcement Learning from Human Feedback (RLHF), specifically Proximal Policy Optimization (PPO), and reward-free methods, specifically Direct Preference Optimization (DPO). While DPO is popular in academic benchmarks for its simplicity, PPO is widely used in industry (e.g., ChatGPT, Claude). The study aims to determine if DPO is truly superior, identify the fundamental limitations of DPO, and uncover the critical factors that allow PPO to achieve state-of-the-art (SOTA) performance.

### Theoretical Analysis of DPO
The paper argues that DPO is susceptible to distribution shifts and out-of-distribution (OOD) data. The authors present **Theorem 4.1**, stating that the class of policies induced by PPO ($\Pi_{\text{PPO}}$) is a proper subset of those induced by DPO ($\Pi_{\text{DPO}}$):

$$
\Pi_{\text{PPO}} \subsetneq \Pi_{\text{DPO}}
$$

This implies that any solution found by PPO also minimizes the DPO objective, but DPO can discover solutions that do not maximize the RL objective. Specifically, DPO may assign high probabilities to OOD responses that were not present in the preference dataset, leading to biased policies and unpredictable behavior.

### Method and Recipe
The study compares PPO and DPO across dialogue and code generation tasks. To optimize PPO, the authors identify a "recipe" of three critical factors:
1.  **Advantage Normalization:** Stabilizes training and improves performance.
2.  **Large Batch Size:** Particularly critical for code generation tasks.
3.  **Reference Model EMA:** Updating the reference model parameters using an Exponential Moving Average (EMA) to prevent the policy from being overly regularized toward a stale reference model.

#### Key Formulas
The general RLHF objective is defined as:

$$
J_{r}(\pi_{\theta})=\mathbb{E}_{\mathbf{x}\sim p_{\text{d a t a}},\mathbf{y}\sim\pi_{\theta}}\left[r(\mathbf{x},\mathbf{y})-\beta\text{l o g}\frac{\pi_{\theta}(\mathbf{y}\mid\mathbf{x})}{\pi_{\text{r e f}}(\mathbf{y}\mid\mathbf{x})}\right]
$$

PPO utilizes a reward model $r_\phi$ trained via the Bradley-Terry model:

$$
\mathbb{P}_{\phi}(\mathbf{y}_{w}\succ\mathbf{y}_{l}\mid\mathbf{x}) = \sigma\left(r_{\phi}(\mathbf{x},\mathbf{y}_{w})-r_{\phi}(\mathbf{x},\mathbf{y}_{l})\right)
$$

DPO optimizes the policy directly using the following loss:

$$
\mathcal{L}_{\text{D P O}}(\pi_{\theta})=-\mathbb{E}_{(\mathbf{x},\mathbf{y}_{w},\mathbf{y}_{l})\sim\mathcal{D}}\left[\text{l o g}\sigma\left(\beta{\left(\text{l o g}\frac{\pi_{\theta}(\mathbf{y}_{w}\mid\mathbf{x})}{\pi_{\text{r e f}}(\mathbf{y}_{w}\mid\mathbf{x})}-\text{l o g}\frac{\pi_{\theta}(\mathbf{y}_{l}\mid\mathbf{x})}{\pi_{\text{r e f}}(\mathbf{y}_{l}\mid\mathbf{x})}\right)}\right)\right]
$$

### Key Quantitative Results
PPO consistently outperformed DPO and iterative DPO (DPO-Iter) across all benchmarks:

*   **Code Generation (CodeContest):** Using CodeLlama-34B, PPO achieved SOTA results, improving the $10@1k$ pass rate from 16.4% (AlphaCode-41B) to **22.4%**. In contrast, DPO failed to generate any correct codes (pass rate of 0) after one epoch.
*   **Code Generation (APPS):** PPO (CodeLlama-34B) achieved a pass@5 of **44.4%**, significantly higher than DPO-Iter (**34.2%**) and the SFT baseline (**38.6%**).
*   **Dialogue (HH-RLHF):** PPO achieved a higher OpenAssistant reward (**0.718**) compared to DPO (**0.611**) and DPO-Iter (**0.678**). GPT-4 evaluations showed a preference for PPO over DPO (PPO win rate: 42%, Tie: 28%, DPO win: 30%).
*   **Safety (SafeRLHF):** With an SFT(Alpaca) base, DPO achieved a safety rate of **55.4%**, whereas PPO achieved **99.5%**.

### Stated Limitations
The authors acknowledge that for the code competition tasks (APPS and CodeContest), they utilized ground-truth rewards for PPO training and for labeling the preference pairs in DPO-Iter, rather than a learned reward model. They state that while this is a limitation, it does not affect the overall conclusions regarding the relative superiority of PPO over DPO.
