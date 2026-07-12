---
id: arxiv:2407.14477
type: paper
title: Data-Centric Human Preference with Rationales for Direct Preference Alignment
url: https://arxiv.org/abs/2407.14477
retrieved: '2026-07-12'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Data-Centric Human Preference with Rationales for Direct Preference Alignment

### Core Problem
Standard preference datasets used for aligning Large Language Models (LLMs) typically consist of triplets: a prompt $x$, a preferred response $y_w$, and a rejected response $y_l$. These datasets are often "opaque" because they lack explicit explanations regarding *why* one response is preferred over another. This ambiguity can lead to data inefficiency, the learning of spurious correlations (such as a bias toward longer responses), and reward hacking, requiring vast amounts of costly human-annotated data to achieve robust alignment.

### Method
The authors propose a data-centric framework that augments standard preference pairs with **rationales** ($r$)—machine-generated explanations that detail the reasoning behind the human preference. This approach is designed to be compatible with various direct preference optimization algorithms.

**Step-by-Step Recipe:**
1. **Dataset Augmentation:** Transform the standard dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ into an enriched dataset $\mathcal{D}' = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}, r^{(i)}\}_{i=1}^N$ by generating rationales that explain the preference of $y_w$ over $y_l$.
2. **Loss Function Adaptation:** Integrate a rationale prediction term into the optimization objective. For Direct Preference Optimization (DPO), this results in **Rationale-DPO (RDPO)**.
3. **Optimization:** Train the policy $\pi_\theta$ by minimizing the combined loss of the preference margin and the likelihood of the rationale.

**Key Formula:**
The RDPO loss function is defined as:

$$
\mathcal{L}_{\mathsf{RDPO}}(\pi_{\theta};\pi_{\mathrm{ref}})=-\mathbb{E}_{(x,y_w,y_l,r)\sim\mathcal{D}'}\left[\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\beta\log\frac{\pi_{\theta}(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)+\gamma\log\pi_{\theta}(r|x,y_w\succ y_l)\right]
$$

Where:
*   $\beta$ is the divergence hyperparameter between the policy and reference model.
*   $\gamma$ is a hyperparameter weighting the impact of the rationales.
*   $\sigma$ is the sigmoid function.

### Key Quantitative Results
The method was evaluated using Mistral-7B and Llama-3/3.1-8B models across datasets including Orca DPO Pairs, UltraFeedback, and Anthropic HH.

*   **Learning Efficiency:** RDPO accelerates convergence. On the Orca dataset, RDPO achieved a $\sim 60\%$ winrate against the SFT baseline using only 3,000 training samples, whereas standard DPO required 9,000 samples to reach the same rate.
*   **Model Performance:** 
    *   RDPO consistently outperformed DPO, achieving winrates above $60\%$ against DPO-trained models on both Orca and UltraFeedback datasets.
    *   On the AlpacaEval 2.0 benchmark (Mistral-7B), RDPO achieved a length-controlled (LC) winrate of $22.42\%$, compared to $19.52\%$ for DPO.
    *   **RSimPO** (SimPO adapted with rationales) achieved winrates of $68\%$ on UltraFeedback and $65\%$ on Anthropic HH against the vanilla SimPO model.
*   **Ablation (Rationale-Only):** In an extreme case where the DPO alignment loss was set to zero, training on rationales alone achieved a $61.8\%$ winrate against SFT on the Orca dataset, suggesting rationales provide a rich enough signal to drive alignment independently.
*   **Regularization:** Analysis of log probabilities indicated that RDPO acts as a regularizer, mitigating the reward hacking and length exploitation often seen in DPO.

### Theoretical Insights
The authors use information theory to show that rationales provide additional value when the reward modeling based on query-response pairs alone is biased or overconfident. They derive a generalization bound for preference learning:

$$
\mathrm{gen}(\mu,\mathcal{A}_{ra})\leq\sqrt{\frac{2\sigma^{2}}{n}\cdot\big(I(\theta_{ra};Z)+\delta+\eta_{1}\big)}
$$

This suggests that sample efficiency improves when the rationale contains minimal irrelevant information ($\eta_1$ is small) and the learning process focuses on preference-predictive content.

### Stated Limitations
*   **Model Scale:** The study focused on models up to 8 billion parameters; the impact on larger models remains to be explored.
*   **Data Format:** The current framework is designed for pairwise preferences; the authors note the need to extend this to unpaired preference learning (e.g., KTO).
*   **Reward Modeling:** The specific impact of rationales on training explicit reward models (as opposed to direct policy optimization) requires further investigation.
