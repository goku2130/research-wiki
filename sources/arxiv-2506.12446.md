---
id: arxiv:2506.12446
type: paper
title: 'From Outcomes to Processes: Guiding PRM Learning from ORM for Inference-Time
  Alignment'
url: https://arxiv.org/abs/2506.12446
retrieved: '2026-07-12'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# SP-PRM: Guiding Process Reward Model Learning from Outcome Reward Models

### Core Problem
Inference-time alignment often utilizes **Reward-Guided Search (RGS)** to steer Large Language Model (LLM) generation. However, most RGS frameworks rely on **Outcome Reward Models (ORMs)**, which are trained to evaluate complete responses. When ORMs are used to provide process rewards for partial sequences, a **granularity mismatch** occurs. This leads to inconsistent scoring—where a high-scoring complete response may have low-scoring partial prefixes—resulting in "myopic" decoding decisions and suboptimal alignment with human preferences.

### Method: The SP-PRM Framework
To resolve this, the authors propose **SP-PRM**, a dual-consistency framework that derives a **Process Reward Model (PRM)** from an ORM without requiring new human annotations. The framework optimizes for two objectives: **Score Consistency** (ensuring partial sequences align with the final outcome score) and **Preference Consistency** (ensuring partial evaluations align with human preferences).

#### Step-by-Step Recipe:
1.  **Partial Sequence Dataset Construction ($\mathcal{D}_{partial}$):**
    *   Extract incomplete sequences from a preference dataset $\mathcal{D}$.
    *   **Token-Level Truncation (TLT):** Generates partial sequences at every token position.
    *   **Stochastic Sampling Truncation (SST):** An adaptive strategy that samples $K$ truncation points $t_i \sim U(1, T)$ per pair to improve efficiency and mitigate overfitting.
2.  **Score Consistency Training:**
    *   Train a reward model $r_\theta$ using a Bradley-Terry objective on partial pairs $(x, y_{<t}^w, y_{<t}^l)$, where $y^w$ is the preferred response.
3.  **Preference Consistency Integration:**
    *   **Reference Guidance:** Use a pre-trained ORM $r_\phi$ as a reference. Samples are retained in $\mathcal{D}_{partial}$ only if the preference predicted by $r_\phi$ aligns with the score consistency requirement.
    *   **Confidence Weighting:** Calculate the Shannon entropy $H_t$ of the reward gap using $r_\phi$. Since longer prefixes typically yield higher confidence (lower entropy), weights are assigned as $w_t = 1/H_t$.
4.  **Final Optimization:**
    *   The model is trained using a weighted Bradley-Terry loss to balance long-term outcome optimization with semantic understanding.

### Key Formulas
The standard reward model loss is defined as:

$$
\mathcal{L}_{\text{R M}}=-\mathbb{E}_{(x,y^{w},y^{l})\sim\mathcal{D}} \log \sigma\left(r(x,y^{w})-r(x,y^{l})\right)
$$

To implement preference-based weighting, the probability $p_t^w$ and entropy $H_t$ are calculated via the reference model $r_\phi$:

$$
p_{t}^{w}=\sigma\big(|r_{\phi}(x,y_{<t}^{w})-r_{\phi}(x,y_{<t}^{l})|\big)
$$

$$
H_{t}=-(p_{t}^{w}\log p_{t}^{w}+p_{t}^{l}\log p_{t}^{l})
$$

The final **SP-PRM loss function** integrates these weights $w$:

$$
\mathcal{L}_{\mathrm{S P-P R M}}=-\mathbb{E}_{(x, y_{<t}^{w}, y_{<t}^{l}) \sim \mathcal{D}_{\mathrm{p a r t i a l}}} w \log \left(\sigma\left(r_{\theta}\left(x, y_{<t}^{w}\right)-r_{\theta}\left(x, y_{<t}^{l}\right)\right)\right)
$$

where $w = 1/H_t$ if $r_\phi(x,y_{<t}^w) > r_\phi(x,y_{<t}^l)$, and $w=0$ otherwise.

### Key Quantitative Results
SP-PRM was evaluated across dialogue (HH-RLHF), summarization (TL;DR), and reasoning (GSM8K) tasks using models from 1B to 8B parameters.

*   **General Performance:** Achieved a **3.6%–10.3% improvement** in GPT-4 evaluation scores across all tasks.
*   **Summarization (TL;DR):** Reward scores increased by **3-7%**. Specifically, Best-of-N (BoN-16) saw an **11.7% improvement** in average reward (0.60 $\rightarrow$ 0.67).
*   **Reasoning (GSM8K):** Integrating SP-PRM with Chunk-level Beam Search (CBS) increased accuracy by **3.5%** (1B model) and **2.5%** (3B model). BoN-16 reached accuracies of **65.5% (1B)** and **69.5% (3B)**.
*   **Safety (AdvBench):** Reduced the Attack Success Rate (ASR) by **20%** compared to base RGS methods.
*   **Consistency Gain:** The Agreement Rate for score consistency ($\text{AR}_{\text{RM-SC}}$) improved from $\approx 60\%$ (original ORMs) to over **55% at 5 tokens** and up to **64.7% at 50 tokens**.

### Stated Limitations
*   **Model Scale:** Experiments were limited to the Llama-3 series (1B to 8B parameters); the authors note that findings may not necessarily apply to larger models.
*   **Inference Speed:** The RGS method requires further inference-time optimization to enhance generation speed.
