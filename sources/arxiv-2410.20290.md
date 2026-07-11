---
id: arxiv:2410.20290
type: paper
title: Fast Best-of-N Decoding via Speculative Rejection
url: https://arxiv.org/html/2410.20290v2
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# Fast Best-of-N Decoding via Speculative Rejection

### Core Problem
Inference-time alignment via the **Best-of-N (BoN)** method involves generating $N$ independent responses for a prompt and selecting the one with the highest score from a reward model. While BoN is competitive with post-training alignment (like RLHF or DPO), it is computationally prohibitive for large $N$ (e.g., $N > 1000$). The primary bottleneck is the memory required for the KV cache and the computational cost of generating $N$ full sequences, often requiring dozens or hundreds of GPUs to avoid Out-of-Memory (OOM) errors and maintain reasonable latency.

### Method: Speculative Rejection
The authors propose **Speculative Rejection**, an algorithm that simulates a large $N$ on a single accelerator by dynamically reducing the batch size during auto-regressive generation. The method is based on the observation that partial rewards (scores of incomplete utterances) are positively correlated with final rewards, allowing the system to halt unpromising trajectories early.

**Step-by-Step Recipe:**
1. **Initial Batching:** Determine an initial batch size $b_{init}$ based on available GPU memory and prompt length.
2. **Early Generation:** Generate tokens for all $b$ sequences until the GPU approaches its memory limit (OOM) or the end-of-sequence (EOS) token is reached.
3. **Partial Reward Evaluation:** Use the reward model $s$ to score the concatenation of the prompt and the partial response for all active sequences.
4. **Threshold Calculation:** Compute a prompt-dependent cutoff threshold $r_{cut}$ as the $\alpha$-th lower quantile of the partial rewards, where $\alpha \in (0,1)$ is a hyperparameter representing the rejection rate.
5. **Speculative Rejection:** Terminate all sequences with partial rewards below $r_{cut}$. Only the top $(1-\alpha)$ proportion of sequences are accepted for the next round.
6. **Iterative Refinement:** Repeat the generation and rejection phases until all remaining sequences are completed.
7. **Final Selection:** Return the completed response with the highest final reward.

**Key Formulas:**
The set of partial rewards is defined as:

$$
\mathcal{R}_{\text{partial}} := \{s(Y_k^{\le \tau_k}) : k = 1, 2, \dots, b\}
$$

The cutoff threshold is:

$$
r_{cut} := q_\alpha(\mathcal{R}_{\text{partial}})
$$

The index set for accepted sequences is:

$$
\mathcal{I}_{\text{accepted}} = \{k : 1 \le k \le b, s(Y_k^{\le \tau_k}) \ge r_{cut}\}
$$

The final output is:

$$
Y_{\mathrm{SR}} = Y_{k^*}, \quad \text{where} \quad k^* := \arg \max_{k \in \mathcal{I}} \{ s(Y_k) \mid Y_k \sim p(\cdot \mid X) \}
$$

### Key Quantitative Results
The method was evaluated on the AlpacaFarm dataset using various models (e.g., Llama-3-8B, Mistral-7B).

*   **Computational Efficiency:** Speculative Rejection on a single GPU achieves reward scores that would require Best-of-N to use between **16 and 32 GPUs** to match, while maintaining similar latency.
*   **Generation Quality:** Using GPT-4-Turbo as an evaluator, Speculative Rejection ($\alpha = 0.5$) achieved an average **win-rate (WR) of 66.17%** and a **length-controlled win-rate (LC-WR) of 70.01%**, outperforming Bo120 (50% WR/LC-WR) and Bo3840 (62.89% WR).
*   **Probability Maximization:** When used to maximize the probability of generated utterances (using perplexity as the metric), the method achieved an average **speedup of 39.9$\times$** compared to Bo120 and produced responses with lower average perplexity (**1.554** vs. **2.407**).

### Stated Limitations
*   **Prompt-Dependent Stopping:** The correlation between partial and final rewards varies by prompt. The authors note that a fixed rejection rate $\alpha$ may be suboptimal and suggest that an adaptive rejection rate could further improve speedup and scores.
*   **Reward Model Utility:** Current reward models are typically trained to evaluate full responses. The authors suggest that training reward models specifically as **value functions** (to predict expected final scores from partial sequences) would lead to more accurate early stopping and optimal speedup.
