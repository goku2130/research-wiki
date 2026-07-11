---
id: aclanthology:regularized-best-of-n-sampling-with-mini
type: web
title: Regularized Best-of-N Sampling with Minimum Bayes Risk Decoding
url: https://aclanthology.org/2025.naacl-long.472.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# Regularized Best-of-N Sampling with Minimum Bayes Risk Decoding

### Core Problem
Best-of-N (BoN) sampling is a decoding-time alignment strategy that selects the response with the highest score from $N$ samples based on a proxy reward model. However, BoN is susceptible to **reward hacking**, where the system over-optimizes a misspecified proxy reward model that does not perfectly reflect true human preferences. This often leads to performance degradation as $N$ increases (e.g., when $N > 16$ or $N > 32$), as the model exploits inaccuracies in the reward proxy rather than optimizing for the intended objective.

### Method
The authors propose **MBR-BoN**, which incorporates the Minimum Bayes Risk (MBR) objective as a proximity regularizer to mitigate reward hacking. The MBR objective penalizes responses that deviate significantly from the center of the sample distribution of the reference policy.

**Step-by-Step Recipe:**
1. **Sampling:** Generate a set $Y_{\text{ref}}$ of $N$ responses from the reference language model $\pi_{\text{ref}}(\cdot|x)$ for a given input $x$.
2. **Reward Scoring:** Compute the proxy reward $R(x, y)$ for every response $y \in Y_{\text{ref}}$.
3. **Utility Computation:** For each response $y$, calculate its average utility (similarity) relative to all other sampled responses $y' \in Y_{\text{ref}}$. In experiments, this is implemented as the cosine similarity of embeddings:

$$
U(y, y') = \cos(\text{emb}(y), \text{emb}(y'))
$$

4. **Selection:** Select the response that maximizes the combined reward and regularized MBR objective:

$$
y_{\text{MBR-BoN}}(x) = \arg\max_{y \in Y_{\text{ref}}} R(x, y) + \beta \sum_{y' \in Y_{\text{ref}}} \frac{1}{N} U(y, y')
$$

   where $\beta$ is a hyperparameter controlling the trade-off between reward optimization and proximity to the reference policy.

**Theoretical Justification:**
The authors analytically demonstrate that maximizing the MBR objective is equivalent to minimizing the **Wasserstein Distance (WD)** between the selected output's distribution and the empirical distribution of the reference policy. They argue that WD is a more robust regularizer than KL-divergence for inference-time algorithms because KL-divergence requires sample sizes exponential to the divergence value to be reliable and is overly sensitive to minor textual variances (e.g., "I will" vs "I'll").

### Key Quantitative Results
The method was evaluated using Mistral-7B and Dolly-v2-3B on the AlpacaFarm and Anthropic’s hh-rlhf datasets, using Eurus as the gold reference reward model.

*   **Decoding Performance:** MBR-BoN consistently outperformed both vanilla BoN and pure MBR decoding across various proxy reward models (SHP-Large, SHP-XL, OASST).
*   **Hyperparameter Tuning:** The optimal $\beta$ varies by dataset and reward model (e.g., $\beta=0.5$ for AlpacaFarm with SHP-Large, but $\beta=20.0$ with OASST). Notably, a development set of as few as **10 instances** was sufficient to find a near-optimal $\beta$.
*   **Preference Dataset Generation:** MBR-BoN was used to generate pairwise datasets for Direct Preference Optimization (DPO), where the MBR-BoN selection served as the "chosen" response and the lowest-reward sample as the "rejected" response. Models trained via DPO on MBR-BoN datasets outperformed those trained on vanilla BoN datasets.
*   **Computational Overhead:** For $N=128$ on an NVIDIA T4 GPU, computing MBR values added approximately **2 seconds** to the total pipeline, compared to $0.1$ seconds for reward value computation.

### Stated Limitations
*   **Computational Complexity:** The utility function computation is quadratic relative to the number of samples $N$, which may limit scalability.
*   **Evaluation Metrics:** The study relied on automated reward models (Eurus) and LLM-as-a-judge (GPT-4o); the authors note that human evaluation is still desirable.
*   **Algorithmic Scope:** The application to preference learning was limited specifically to DPO; the effectiveness of MBR-BoN for other preference optimization algorithms remains unexplored.
