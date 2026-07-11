---
id: arxiv:2605.14366
type: paper
title: Reinforcement Learning with Semantic Rewards Enables Low-Resource Language
  Expansion without Alignment Tax
url: https://arxiv.org/abs/2605.14366
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

# Research Summary: Reinforcement Learning with Semantic Rewards for Low-Resource Language Expansion

## Core Problem: The Alignment Tax
Expanding Large Language Models (LLMs) to low-resource languages typically relies on supervised fine-tuning (SFT) using token-level likelihood objectives. The authors argue that this approach enforces rigid surface-form imitation on narrow, biased data distributions, leading to "alignment tax"—a phenomenon where gains in the target low-resource language come at the cost of catastrophic forgetting of general capabilities and pretrained knowledge.

## Proposed Method: Semantic-Space Alignment
The authors propose a semantic-space alignment paradigm that prioritizes meaning preservation over token-level matching. This is operationalized through a two-stage training process using **Group Relative Policy Optimization (GRPO)**.

### Step-by-Step Recipe
1.  **Cold-start SFT:** The base model (Qwen3-4B) is fine-tuned on a small subset (5k instances) for two epochs to bootstrap minimal competence in the target language (e.g., correct script usage), ensuring the RL stage does not collapse into uninformative exploration.
2.  **Reinforcement Learning (GRPO):** Starting from the cold-start checkpoint, the model is optimized for one epoch. For each prompt $x$, the model samples a group of $K$ candidate outputs $\{y^{(k)}\}_{k=1}^K$. The policy $\pi_\theta(\cdot|x)$ is updated based on relative rewards within the group, utilizing trust-region-style constraints to limit policy drift.
3.  **Reward Calculation:** The model is guided by a composite reward function consisting of semantic similarity and language consistency.

### Key Formulas
The **Semantic Similarity Reward** ($R_{\text{sim}}$) uses a multilingual sentence embedding model $f(\cdot)$ to calculate the cosine similarity $s(y, y^*)$ between the generated output $y^*$ and the reference $y$. To focus on meaningful improvements, a threshold-and-rescale shaping function is applied:

$$
R_{\text{sim}}(y,y^{*})=\begin{cases}{0,}&{s(y,y^{*})\leq\tau,}\\ {\frac{s(y,y^{*})-\tau}{1-\tau},}&{s(y,y^{*})>\tau,}\\ \end{cases}
$$

where $\tau$ is the minimal semantic adequacy level from the cold-start phase.

The **Language Consistency Reward** ($R_{\text{lang}}$) uses rule-based script checks to prevent reward hacking (e.g., mixed-language outputs):

$$
R_{\mathrm{l a n g}}(y)=\begin{cases}{0,}&{\mathrm{l a n g u a g e\;m i x e d},}\\ {1,}&{\mathrm{l a n g u a g e\;c o n s i s t e n t}.}\ \ \end{cases}
$$

The **Final Reward** is a weighted sum:

$$
R(y,y^{*})=\lambda_{\mathrm{s i m}}R_{\mathrm{s i m}}(y,y^{*})+\lambda_{\mathrm{l a n g}}R_{\mathrm{l a n g}}(y)
$$

The authors set $\lambda_{\text{sim}} = 1.5$ and $\lambda_{\text{lang}} = 1.0$.

## Key Quantitative Results
The method was evaluated on Tibetan–Chinese machine translation (MT) and Tibetan headline generation (HG).

*   **Mitigation of Alignment Tax (MT):** While "Strong SFT" achieved higher surface metrics (BLEU-4: 0.6006 vs. RL: 0.4519), it suffered significant degradation in general capabilities. On the CMRC benchmark, RL outperformed SFT by **+5.15 points** in Average score (46.97 vs. 41.82) and **+2.80 points** in F1 (65.79 vs. 62.99).
*   **Semantic Quality (HG):** In open-ended generation, RL was preferred by LLM judges with a **+16.1% higher win rate** (51.2% vs. 35.1%) compared to SFT, despite having lower ROUGE-L scores (0.2530 vs. 0.3095).
*   **Few-Shot Transfer:** When transferring from MT to HG with only 1,000 samples, the RL-initialized model achieved higher semantic similarity (0.5690) than the SFT-initialized model (0.5456).
*   **Mechanistic Analysis:** On an OOD CMRC set, RL showed smaller distributional drift. The mean Negative Log-Likelihood (NLL) increased by only **+0.24** for RL, compared to **+0.64** for SFT. The 90th-percentile KL divergence was also lower for RL (0.0839) than for final SFT (0.0932).

## Stated Limitations
The authors note that the limited and domain-narrow nature of available low-resource data (such as translation corpora) may lead SFT to achieve artificially high in-domain metrics that do not reflect real-world generalization. Additionally, they acknowledge that residual pretrained biases may persist despite the use of semantic RL.
