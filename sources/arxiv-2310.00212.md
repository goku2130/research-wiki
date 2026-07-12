---
id: arxiv:2310.00212
type: paper
title: Pairwise Proximal Policy Optimization (P3O)
url: https://arxiv.org/abs/2310.00212
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

The paper "Pairwise Proximal Policy Optimization (P3O): Harnessing Relative Feedback for LLM Alignment" addresses limitations of Proximal Policy Optimization (PPO) in aligning Large Language Models (LLMs) with human preferences, particularly when using comparison-based reward models.

**Core Problem:**
The dominant approach for LLM alignment, Reinforcement Learning with Human Feedback (RLHF), uses PPO as its RL optimizer. However, PPO exhibits two main limitations when optimizing rewards derived from comparison-based losses like the Bradley-Terry Loss (BTL):
1.  **Reward Invariance:** PPO is not invariant to equivalent reward functions that contain identical preference information but differ by a constant shift. BTL is invariant to such shifts, leading to inconsistencies when PPO optimizes rewards trained with BTL.
2.  **Token-wise vs. Trajectory-wise Optimization:** PPO requires token-wise updates, which introduces complexity in function approximation (e.g., learning a value function V) and algorithmic design (e.g., using Generalized Advantage Estimation (GAE) to redistribute sparse rewards). Reward models, trained on human comparisons, typically provide a single scalar reward for an entire response (trajectory-wise).

**Method/Recipe Step by Step (P3O):**
P3O is a novel trajectory-wise policy gradient algorithm that operates directly on comparative rewards, designed within a new framework called Reinforcement Learning with Relative Feedback.

1.  **Initialization:** Initialize policy parameters $\theta_0$.
2.  **Loop (for k = 0, 1, 2 ...):**
    a.  **Collect Pairwise Trajectories:** Run the current policy $\pi_{\theta_k}$ from a batch of prompts and generate *two* trajectories (responses) for each prompt. This results in a dataset $D_k = \{\tau_i\}$.
    b.  **Compute Trajectory-wise Rewards:** For each pair of generated responses $(y_1, y_2)$ for a prompt $x$, compute a final reward $\hat{r}_{\mathrm{final}}$ that incorporates both the preference reward and a KL penalty:
        $\hat{r}_{\mathrm{final}} = \hat{r}_{\mathrm{preference}} - \beta D_{\mathrm{KL}}(\pi_{\theta_k}(y|x) \| \pi^{\mathrm{SFT}}(y|x))$
        where $\hat{r}_{\mathrm{preference}}$ is derived from the reward model, and $D_{\mathrm{KL}}$ is the KL-divergence from the Supervised Fine-Tuning (SFT) model $\pi^{\mathrm{SFT}}$, with $\beta$ as a regularization coefficient.
    c.  **Estimate Clipped Pairwise Policy Gradient:** Compute the policy gradient using one of two versions of clipping (separately or jointly for actions $y_1$ and $y_2$) as described in Section 4.2 of the paper. The core idea is to use the difference in rewards $(r(y_1|x) - r(y_2|x))$ to update the policy.
    d.  **Apply Gradient Updates:** Update the policy parameters $\theta_k$ using gradient descent based on the estimated clipped pairwise policy gradient.

**Key Formulas in LaTeX:**
*   **Bradley-Terry Loss (BTL) for human preference distribution:**
    $p^{*}(\mathbf{y}_{1}\succ\mathbf{y}_{2}|\mathbf{x})=\frac{\text{e x p}(r^{*}(\mathbf{y}_{1}|\mathbf{x}))}{\text{e x p}(r^{*}(y_{1}|\mathbf{x}))+\text{e x p}(r^{*}(y_{2}|\mathbf{x}))}$
*   **RL Fine-Tuning Objective (PPO's modified reward):**
    $\max \mathbb{E} r_{\phi}(\mathbf{y}|\mathbf{x}) - \beta D_{\mathrm{KL}}(\pi_{\theta}(\cdot|\mathbf{x})\|\pi^{\mathrm{SFT}}(\cdot|\mathbf{x}))$
*   **Pairwise Policy Gradient (PPG) (Equation 6, contextual bandit generalization in Theorem 1):**
    $\nabla\mathcal{L}^{\mathrm{VPG}}(\mathbf{x}) = \mathbb{E}_{\mathbf{y}_{1},\mathbf{y}_{2}\sim\pi_{\theta_{\mathrm{old}}}}\left(r(\mathbf{y}_{1}|\mathbf{x})-r(\mathbf{y}_{2}|\mathbf{x})\right)\nabla\left(\log\frac{\pi_{\theta}(\mathbf{y}_{1}|\mathbf{x})}{\pi_{\theta}(\mathbf{y}_{2}|\mathbf{x})}\right)/2$
    (This is the simplified form; the full form in Theorem 1 includes importance sampling ratios.)
*   **Reward Equivalence Definition (Definition 1):**
    Two reward functions $r(y|x)$ and $r'(y|x)$ are equivalent ($r \sim r'$) if there exists a function $\delta(x)$ such that for every $(x, y)$, $r(y|x) - r'(y|x) = \delta(x)$.
*   **PPO's objective (no clip, contextual bandit setting):**
    $\mathcal{L}_{\mathrm{no\,clip}}^{\mathrm{PPO}}=-\mathbb{E}_{y\sim\pi_{\theta_{\mathrm{old}}}(\cdot|x)}(r(y|x)-V_{\phi}(x))\frac{\pi_{\theta}(y|x)}{\pi_{\theta_{\mathrm{old}}}(y|x)}$
*   **DPO's gradient (simplified):**
    $\nabla\mathcal{L}^{\mathsf{DPO}}(\mathbf{x},\mathbf{y}_{w},\mathbf{y}_{l})=-\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_{l}|\mathbf{x})}{\pi^{\mathsf{SFT}}(y_{l}|\mathbf{x})}-\beta\log\frac{\pi_{\theta}(\mathbf{y}_{w}|\mathbf{x})}{\pi^{\mathsf{SFT}}(y_{w}|\mathbf{x})}\right)\nabla\left(\log\frac{\pi_{\theta}(\mathbf{y}_{w}|\mathbf{x})}{\pi_{\theta}(y_{l}|\mathbf{x})}\right)/2$

**Key Quantitative Results and Numbers:**
*   **KL-Reward Frontier:** P3O consistently outperforms PPO and DPO in terms of KL-Reward trade-off on both TL;DR summarization and Anthropic Helpful and Harmless (HH) question-answering datasets.
    *   On TL;DR, P3O-V2 achieves almost the same highest reward as PPO while exhibiting 25% lower KL-divergence than DPO for the same reward.
    *   On HH, P3O-V1 and P3O-V2 show strictly dominant frontiers over PPO and DPO, respectively, for both 1B and 6B model sizes, delivering a "substantial higher reward in the range of 0.1-0.3".
*   **Head-to-Head Comparisons (HH dataset, 6B model):**
    *   **GPT-4 Evaluation (proxy for human preference):**
        *   P3O vs. PPO: P3O has a 57.0% win rate.
        *   P3O vs. SFT: P3O has a 69.3% win rate.
        *   DPO vs. P3O: DPO has a 45.4% win rate (P3O wins more often).
    *   **Reward Model Evaluation:**
        *   DPO vs. P3O: DPO has a 49.5% win rate (marginally higher reward for DPO).
*   **Additional Statistics (HH dataset, 6B model, highest reward checkpoints):**
    *   **Reward:** P3O: -0.302, DPO: -0.298, PPO: -0.613, SFT: -1.195. (DPO slightly higher reward than P3O).
    *   **KL % sample:** P3O: 9.83, DPO: 12.01, PPO: 7.02, SFT: 0. (P3O has lower KL than DPO for similar reward).
    *   **Token num (average response length):** P3O: 80.46, DPO: 88.84, PPO: 109.03, SFT: 112.70. (P3O and DPO generate shorter responses).

**Stated Limitations:**
*   The paper is "Under review as a conference paper" at the time of publication.
*   The empirical evaluations are conducted on two specific tasks (summarization and question-answering) and specific datasets (TL;DR and HH).
*   The reward model used is a proxy for human preference, not direct human feedback during training.
*   The study focuses on the KL-Reward trade-off and GPT-4 evaluation, which is a proxy for human evaluation.
*   Future work aims to understand the impacts of reward over-optimization on different RL algorithms, generalize the algorithm to more than two ranked responses, and explore applications beyond LLM alignment.
