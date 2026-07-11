---
id: arxiv:2410.16033
type: paper
title: 'TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search Decoding'
url: https://arxiv.org/abs/2410.16033
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search Decoding

### Core Problem
Inference-time alignment via Best-of-N (BoN) sampling improves large language model (LLM) performance by selecting the highest-reward response from $N$ candidates. However, BoN is computationally expensive, as inference FLOPs scale linearly with $N$. While existing acceleration methods like Speculative Best-of-N (SBoN) and Cascade Reward Sampling (CARDS) attempt to prune low-quality responses early, they rely on the assumption that partial rewards correlate with final rewards. The authors demonstrate that traditional reward models (RMs), typically trained on complete responses, produce "chaotic" and poorly correlated scores for partial completions, rendering early pruning unreliable.

### Method
TreeBoN integrates a speculative tree-search strategy into the BoN framework, iteratively branching promising paths and pruning low-quality ones. Instead of traditional RMs, it uses a weighted implicit reward derived from a Direct Preference Optimization (DPO) policy model to guide the search.

**Step-by-Step Recipe:**
1. **Initialization:** The total maximum response length $l_{\max}$ is divided into $N_{\text{layer}}$ equal segments, where each segment length $l_i = l_{\max} / N_{\text{layer}}$.
2. **Root Generation:** The base policy $\pi_{\text{base}}$ generates $N$ initial candidate responses of length $l_1$, forming the first-layer candidate set $C_1$.
3. **Iterative Expansion (for layers $i = 1$ to $N_{\text{layer}}-1$):**
   - **Partial Evaluation:** A partial reward function $r$ computes scores for all candidates $y \in C_i$.
   - **Pruning:** The top candidates are selected based on these scores to form an active set $P_i$. To maintain a constant computational budget, the size of $P_i$ is set to $N / N_{\text{children}}$.
   - **Branching:** For every parent response in $P_i$, the model samples $N_{\text{children}}$ child responses of length $l_{i+1}$, forming the next candidate set $C_{i+1}$.
4. **Final Selection:** After the final layer $C_{N_{\text{layer}}}$ is generated, a reward model evaluates the complete responses, and the one with the highest reward is returned:

$$
\mathbf{y}^{\star}=\text{a r g m a x}_{\mathbf{y}\in C_{N_{\text{l a y c r}}}}r(\mathbf{y}|\mathbf{x})
$$

**Implicit Reward Guidance:**
To evaluate partial responses $\mathbf{y}_{:K}$, TreeBoN utilizes a weighted sum of the log-likelihood ratios from a DPO model:

$$
r_{\text{p a r t i a l}}(\mathbf{y}_{:K}|\mathbf{x})=\sum_{k=0}^{K-1}w_{k}\text{l o g}\frac{\pi^{*}(y_{k}|\mathbf{x},\mathbf{y}_{:k})}{\pi(y_{k}|\mathbf{x},\mathbf{y}_{:k})}
$$

where $w_k = \frac{1}{|\mathbf{y}_{:k}|}$ serves as a weighting factor to adjust the contribution of each token-level ratio.

### Key Quantitative Results
TreeBoN consistently outperforms standard BoN and other accelerated methods across multiple benchmarks:

*   **Win Rates (GPT-4 Evaluation):** Against BoN, TreeBoN achieved the highest win rate of 65% on TutorEval and approximately 60% across AlpacaFarm, HH-RLHF, and UltraFeedback.
*   **Computational Efficiency:** TreeBoN is highly scalable. With only 6.3% of the compute budget (using $N=8$ root nodes compared to BoN's $N=128$), TreeBoN maintained a 55% win rate against BoN.
*   **Reasoning Performance:** On the GSM8K dataset, TreeBoN increased the pass@1 solve rate by 9% over BoN at a maximum response length of 576 tokens.
*   **Comparison to Baselines:** Under identical compute constraints (total tokens generated), TreeBoN significantly outperformed SBoN and CARDS. At a max length of 192, TreeBoN achieved a 63.21% win rate, compared to 49.66% for SBoN and 51.01% for CARDS.
*   **Ablation Findings:** The authors found that using a traditional RM for partial rewards in a tree structure ("RM TreeBoN") was less effective than using the weighted implicit DPO reward, and that the tree structure itself provided a significant advantage over standard BoN even when using the same DPO reward.

### Limitations
The authors state that TreeBoN suffers from the difficulty of training.
