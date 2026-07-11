---
id: arxiv:2511.03827
type: paper
title: 'STARS: Segment-level Token Alignment with Rejection Sampling in Large Language
  Models'
url: https://arxiv.org/abs/2511.03827
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# STARS: Synchronous Token Alignment for Robust Supervision

**STARS (Synchronous Token Alignment for Robust Supervision)** is a decoding-time alignment algorithm designed to improve the safety and system efficiency of Large Language Models (LLMs). It replaces dynamic, uncertainty-based verification with a fixed-horizon supervision schedule to eliminate hardware underutilization and mitigate the risks of miscalibrated model confidence.

### Core Problem
Existing inference-time alignment methods (e.g., CARDS) often use model uncertainty (entropy) to determine when to segment text for verification by a reward model (RM). The authors identify two critical failures in this approach:
1. **Miscalibrated Confident Hallucinations:** LLMs frequently assign high probability to incorrect or toxic tokens. Because uncertainty remains low, verification is delayed, allowing errors to cascade and pollute the context window.
2. **The Straggler Problem:** In batched inference, dynamic segmentation creates a "ragged frontier." The entire batch must wait for the longest segment (the straggler) to complete before the RM forward pass can occur, leading to significant GPU idle time and reduced throughput.

### Method
STARS decouples the verification schedule from the model's internal confidence by imposing a fixed segment horizon $K$. The process follows these steps:
1. **Fixed-Horizon Generation:** The model generates exactly $K$ tokens for all requests in a batch.
2. **Synchronous Verification:** The entire batch pauses simultaneously to undergo a parallel forward pass through the reward model.
3. **Rejection Sampling:** Based on the RM score, trajectories are accepted or rejected. If rejected, the model rewinds and re-samples.
4. **Iteration:** This "supervision heartbeat" repeats every $K$ tokens until the generation is complete.

This approach ensures that any "confident hallucination" is detected and pruned within at most $K$ tokens, strictly bounding the "compute-at-risk."

### Key Formulas
The authors describe the inefficiency of dynamic batching where $L_i$ is the segment length for the $i$-th request. The batch latency is governed by the maximum segment length:

$$
L_{batch} = \max_i(L_i)
$$

When a request $j$ has a shorter segment ($L_j < L_{batch}$), the GPU cores assigned to that request remain idle, creating pipeline bubbles. STARS eliminates this by ensuring $L_i = K$ for all $i$.

### Quantitative Results
STARS was evaluated on the HH-RLHF benchmark using Llama-7B and Mistral-7B backbones with a Llama-7B-RM.

**Alignment Quality (Win-Tie vs. Vanilla):**
STARS achieved competitive alignment quality compared to the dynamic CARDS method:
* **Llama-7b:** STARS (60.2%) vs. CARDS (64.5%) vs. Vanilla (50.0%).
* **Mistral-7b:** STARS (64.5%) vs. CARDS (69.8%) vs. Vanilla (50.0%).

**System Efficiency (Batch Size = 64):**
STARS significantly outperformed CARDS in throughput and waste reduction:
* **Throughput:** STARS ($K=15$) achieved **185.0 T/s**, a $\approx 53.5\%$ improvement over CARDS (**120.5 T/s**). STARS ($K=30$) reached **195.0 T/s**.
* **Rejection Waste:** STARS ($K=15$) reduced the average number of discarded tokens per rejection to **15.0**, compared to **45.2** for CARDS.

### Limitations
The authors note that reward models (RMs) are typically trained on complete human-written responses and can be miscalibrated when evaluating incomplete text sequences. While STARS uses traditional item-level RMs to evaluate fixed-size blocks, the inherent difficulty of scoring partial text remains a challenge for reward-guided decoding.
